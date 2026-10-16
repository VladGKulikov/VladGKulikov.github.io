"""Capstone data: a fallback tokenizer, packing, and deterministic batches.

This module deliberately has no extra dependencies. The byte tokenizer is not
a practical course choice—module 2 explains why byte-level sequences without
learned merges are unacceptably long. It merely lets training and tests run in
any environment. The pretraining notebook replaces it with the BBPE trained in
module 2 without changing the rest of the pipeline: both implement the same
`encode(str) -> list[int]` contract.

Shape contract: ``seq_len`` is the number of trainable transitions. The stream
returns a window of length ``seq_len + 1``; MiniLLM shifts it inside the loss,
thereby training every transition, including boundaries between adjacent windows.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


class ByteTokenizer:
    """Use UTF-8 bytes as IDs; the vocabulary is 256 bytes plus control tokens."""

    BOS, EOS, PAD = 256, 257, 258
    vocab_size = 259

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        return [self.BOS, *ids, self.EOS] if add_special else ids

    def decode(self, ids: list[int]) -> str:
        payload = bytes(i for i in ids if i < 256)
        return payload.decode("utf-8", errors="replace")


def toy_corpus(repeats: int = 400) -> str:
    """A small corpus with learnable structure.

    The point is not language quality but testability. This text has low
    entropy, so a few hundred CPU steps must drive loss well below uniform
    ln(V). If it does not, the training loop is broken; “too little data” is
    not the explanation.
    """
    templates = [
        "коты спят на окне, а собаки спят на полу.\n",
        "утром светло, вечером темно, ночью совсем темно.\n",
        "один плюс один равно два, два плюс два равно четыре.\n",
        "the cat sleeps on the mat and the dog sleeps on the floor.\n",
        "one plus one is two, two plus two is four.\n",
    ]
    return "".join(templates[i % len(templates)] for i in range(repeats * len(templates)))


@dataclass
class TokenStream:
    """A flat token stream with deterministic window slicing."""

    tokens: torch.Tensor          # [T_total], dtype=long
    seq_len: int

    @classmethod
    def from_text(cls, text: str, tokenizer: ByteTokenizer, seq_len: int) -> "TokenStream":
        ids = tokenizer.encode(text)
        if len(ids) < seq_len + 1:
            raise ValueError(f"корпус короче окна: {len(ids)} < {seq_len + 1}")
        return cls(torch.tensor(ids, dtype=torch.long), seq_len)

    def __len__(self) -> int:
        return (self.tokens.numel() - 1) // self.seq_len

    def checksum(self) -> str:
        """Stream SHA-256: the data contract from NOTEBOOK_POLICY §3."""
        return hashlib.sha256(self.tokens.numpy().tobytes()).hexdigest()[:16]

    def batch(self, indices: torch.Tensor) -> torch.Tensor:
        """indices [B] -> windows [B, seq_len+1] containing seq_len targets."""
        starts = indices * self.seq_len
        offsets = torch.arange(self.seq_len + 1)
        return self.tokens[starts[:, None] + offsets[None, :]]

    def loader(self, batch_size: int, seed: int = 0, shuffle: bool = True,
               start_batch: int = 0):
        """An infinite deterministic batch generator.

        `start_batch` resumes at an arbitrary position without reading earlier
        batches. Each epoch's permutation is derived from (seed, epoch), so the
        global batch number fully determines the position. This is the cursor
        from NOTEBOOK_POLICY §3: an interrupted run must resume at the same
        location in the data.
        """
        n = len(self)
        per_epoch = n // batch_size
        if per_epoch == 0:
            raise ValueError(f"batch_size={batch_size} больше числа окон {n}")
        global_batch = start_batch
        while True:
            epoch, within = divmod(global_batch, per_epoch)
            if shuffle:
                gen = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
                order = torch.randperm(n, generator=gen)
            else:
                order = torch.arange(n)
            for i in range(within, per_epoch):
                yield self.batch(order[i * batch_size:(i + 1) * batch_size])
                global_batch += 1


# =============================================================================
# Sharded stream for profile L
# =============================================================================
def write_shards(tokens, out_dir: str | Path, shard_tokens: int = 1_000_000,
                 dtype: str = "uint16") -> dict:
    """Write a flat token stream to .bin shards and a checksummed index.json.

    Why shards? A profile-L corpus with billions of tokens does not fit in
    memory, and a single file is inconvenient for resuming downloads and
    verifying integrity. uint16 is enough for vocabularies up to 65,536. It
    uses half the space of uint32—a 40 GB difference at 20B tokens.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_dtype = np.dtype(dtype)
    if not np.issubdtype(target_dtype, np.integer):
        raise ValueError(f"dtype шардов должен быть целочисленным, получен {dtype}")
    source = np.asarray(tokens)
    if source.ndim != 1 or source.size == 0:
        raise ValueError("tokens должен быть непустым одномерным массивом")
    if not np.issubdtype(source.dtype, np.integer):
        raise ValueError(f"идентификаторы должны быть целыми, получен {source.dtype}")
    limits = np.iinfo(target_dtype)
    smallest, largest = int(source.min()), int(source.max())
    # Check the range BEFORE casting dtype. Otherwise a NumPy array containing
    # ID 70000 would silently turn it into 4464 when written as uint16.
    if smallest < limits.min or largest > limits.max:
        raise ValueError(f"идентификаторы не помещаются в {dtype}")
    array = source.astype(target_dtype, copy=False)

    shards = []
    for i, start in enumerate(range(0, array.size, shard_tokens)):
        chunk = array[start:start + shard_tokens]
        name = f"shard_{i:05d}.bin"
        (out_dir / name).write_bytes(chunk.tobytes())
        shards.append({
            "name": name,
            "tokens": int(chunk.size),
            "sha256": hashlib.sha256(chunk.tobytes()).hexdigest()[:32],
        })
    index = {"dtype": dtype, "total_tokens": int(array.size), "shards": shards}
    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index


class ShardStream:
    """Read shards lazily through memmap with the TokenStream interface.

    Data is not loaded into memory: NumPy maps the file and the operating system
    pages it in. SHA-256 integrity checks are optional and disabled by default,
    because on tens of gigabytes they require a full disk pass.
    """

    def __init__(self, directory: str | Path, seq_len: int, verify: bool = False) -> None:
        self.directory = Path(directory)
        self.seq_len = seq_len
        self.index = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        self.dtype = np.dtype(self.index["dtype"])
        self.maps, offsets, total = [], [], 0
        for shard in self.index["shards"]:
            path = self.directory / shard["name"]
            data = np.memmap(path, dtype=self.dtype, mode="r")
            if verify:
                digest = hashlib.sha256(data.tobytes()).hexdigest()[:32]
                if digest != shard["sha256"]:
                    raise RuntimeError(f"контрольная сумма шарда {shard['name']} не сошлась")
            self.maps.append(data)
            offsets.append(total)
            total += data.size
        self.offsets = np.array(offsets)
        self.total_tokens = total

    def __len__(self) -> int:
        return (self.total_tokens - 1) // self.seq_len

    def close(self) -> None:
        """Release file mappings explicitly, including on Windows.

        On Unix a temporary directory can usually be removed while a memmap is
        alive; Windows locks the file until the mapping closes. An explicit
        lifecycle makes behavior consistent and does not leave cleanup to the
        garbage collector.
        """
        for data in self.maps:
            mapping = getattr(data, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self.maps.clear()

    def __enter__(self) -> "ShardStream":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # A destructor must not mask the interpreter's original error.
            pass

    def checksum(self) -> str:
        return hashlib.sha256("".join(s["sha256"] for s in self.index["shards"])
                              .encode()).hexdigest()[:16]

    def _read(self, start: int, length: int) -> np.ndarray:
        """Read a window that may cross shard boundaries."""
        out = np.empty(length, dtype=self.dtype)
        written = 0
        shard = int(np.searchsorted(self.offsets, start, side="right") - 1)
        position = start - self.offsets[shard]
        while written < length:
            data = self.maps[shard]
            take = min(length - written, data.size - position)
            out[written:written + take] = data[position:position + take]
            written += take
            shard += 1
            position = 0
            if shard >= len(self.maps) and written < length:
                raise IndexError("окно выходит за конец потока")
        return out

    def batch(self, indices: torch.Tensor) -> torch.Tensor:
        rows = [self._read(int(i) * self.seq_len, self.seq_len + 1) for i in indices]
        return torch.from_numpy(np.stack(rows).astype(np.int64))

    loader = TokenStream.loader
