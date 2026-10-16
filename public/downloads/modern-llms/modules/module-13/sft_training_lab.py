"""CPU teaching experiment: serialize, pack, train, generate, and reload.

Synthetic symbol reversal exposes the data contract; it is not evidence about LLM quality.
"""
import copy
import importlib.util
import io
import subprocess
import sys

if importlib.util.find_spec("torch") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
torch.manual_seed(130)
VOCAB, WIDTH, BOS, USER, ASSISTANT, END = 16, 32, 0, 1, 2, 3


def dialogue(a, b):
    # Reverse two symbols. Role headers are inputs; answer and END are targets.
    ids = torch.tensor([BOS, USER, a, b, END, ASSISTANT, b, a, END])
    labels = ids.clone()
    labels[:6] = -100
    return ids, labels


def pack(records):
    ids, labels, positions, segments = [], [], [], []
    for i, (x, y) in enumerate(records):
        target = y.clone()
        target[0] = -100  # No prediction across an independent example boundary.
        ids.append(x); labels.append(target)
        positions.append(torch.arange(len(x)))
        segments.append(torch.full_like(x, i))
    return tuple(torch.cat(xs) for xs in (ids, labels, positions, segments))


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(VOCAB, WIDTH)
        self.position = nn.Embedding(16, WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.proj = nn.Linear(WIDTH, WIDTH)
        self.norm = nn.LayerNorm(WIDTH)
        self.ff = nn.Sequential(nn.Linear(WIDTH, 64), nn.GELU(), nn.Linear(64, WIDTH))
        self.head = nn.Linear(WIDTH, VOCAB)

    def forward(self, ids, positions, segments):
        h = self.token(ids) + self.position(positions)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        idx = torch.arange(len(ids))
        allowed = (segments[:, None] == segments[None, :]) & (idx[:, None] >= idx[None, :])
        scores = q @ k.T / WIDTH**.5
        h = self.norm(h + self.proj(scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ v))
        return self.head(h + self.ff(h))


def loss_sum(model, batch):
    x, y, pos, seg = batch
    logits = model(x, pos, seg)
    return F.cross_entropy(logits[:-1], y[1:], ignore_index=-100, reduction="sum")


train = [dialogue(a, b) for a in range(4, 12) for b in range(4, 12) if a != b]
heldout = [dialogue(a, a) for a in range(4, 12)]
model = TinyCausalLM()

# Packing preserves logits, summed loss and gradients when positions and boundaries match.
records = train[:3]
packed_batch = pack(records)
with torch.no_grad():
    packed_logits = model(packed_batch[0], packed_batch[2], packed_batch[3])
    individual = torch.cat([model(b[0], b[2], b[3]) for b in map(lambda r: pack([r]), records)])
torch.testing.assert_close(packed_logits, individual, rtol=1e-5, atol=1e-6)
packed_loss = loss_sum(model, packed_batch)
separate_loss = sum(loss_sum(model, pack([r])) for r in records)
torch.testing.assert_close(packed_loss, separate_loss, rtol=1e-6, atol=1e-6)
gp = torch.autograd.grad(packed_loss, tuple(model.parameters()))
gs = torch.autograd.grad(separate_loss, tuple(model.parameters()))
for left, right in zip(gp, gs):
    torch.testing.assert_close(left, right, rtol=1e-4, atol=3e-6)

def score(model, records):
    with torch.no_grad():
        return sum(float(loss_sum(model, pack([r]))) for r in records) / (3 * len(records))

before = {"train": score(model, train), "heldout": score(model, heldout)}
optimizer = torch.optim.AdamW(model.parameters(), lr=.006)
history = []
for step in range(240):
    # Eight independent examples per packed row; all 56 rotate through training.
    subset = [train[(step * 8 + j) % len(train)] for j in range(8)]
    batch = pack(subset)
    loss = loss_sum(model, batch) / int((batch[1][1:] != -100).sum())
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    history.append(float(loss.detach()))
after = {"train": score(model, train), "heldout": score(model, heldout)}
assert after["train"] < before["train"] / 5

@torch.no_grad()
def generate(model, prompt, max_new=4):
    sequence = prompt.clone()
    output = []
    for _ in range(max_new):
        logits = model(sequence, torch.arange(len(sequence)), torch.zeros_like(sequence))
        token = int(logits[-1].argmax())
        output.append(token)
        sequence = torch.cat([sequence, torch.tensor([token])])
        if token == END:
            break
    return output

# Generate once per held-out example, then derive every metric from stored outputs.
generated = [generate(model, ids[:6]) for ids, _ in heldout]
exact = sum(pred == ids[6:].tolist() for pred, (ids, _) in zip(generated, heldout)) / len(heldout)
stop_rate = sum(bool(pred) and pred[-1] == END for pred in generated) / len(generated)
buffer = io.BytesIO()
torch.save(model.state_dict(), buffer); buffer.seek(0)
restored = TinyCausalLM()
restored.load_state_dict(torch.load(buffer, weights_only=True))
with torch.no_grad():
    probe = pack(heldout[:2])
    torch.testing.assert_close(model(probe[0], probe[2], probe[3]), restored(probe[0], probe[2], probe[3]), rtol=0, atol=0)
print("token NLL before / after:", before, after)
print("held-out exact / stop rate:", exact, stop_rate, "stored outputs:", generated)
print("packed logits, summed loss, gradients and checkpoint reload: PASS")
