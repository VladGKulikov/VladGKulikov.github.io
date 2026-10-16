"""Small NumPy embedding lab; the source is embedded in both learner notebooks.

This is an operation-level example with chosen weights, not a trained language model.
No files, downloads, GPU, or repository imports are required by this source.
"""
import numpy as np


def softmax(logits):
    """Normalize the final axis, subtracting its maximum for numerical stability."""
    values = np.asarray(logits, dtype=np.float64)
    exp_values = np.exp(values - values.max(axis=-1, keepdims=True))
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def embedding_budget(vocab_size, dim, *, tied, bytes_per_parameter=2):
    """Count unique weight parameters and payload, excluding bias and training state."""
    if any(type(x) is not int or x <= 0 for x in (vocab_size, dim, bytes_per_parameter)):
        raise ValueError("dimensions and bytes_per_parameter must be positive integers")
    if type(tied) is not bool:
        raise ValueError("tied must be a boolean")
    parameters = vocab_size * dim * (1 if tied else 2)
    payload = parameters * bytes_per_parameter
    return {"parameters": parameters, "bytes": payload, "GiB": payload / 2**30}


E = np.array([[1., 0.], [2., 0.], [0., 1.]])
ids = np.array([0, 2, 0])
X = E[ids]
assert X.shape == (3, 2)
np.testing.assert_array_equal(X, [[1, 0], [0, 1], [1, 0]])

h = np.array([1., 0.])  # A chosen state, not a computed Transformer output.
U_untied = np.array([[0., 1., -1.], [2., 0., 1.]])
z_tied = h @ E.T
z_untied = h @ U_untied
p_tied = softmax(z_tied)
np.testing.assert_array_equal(z_tied, [1, 2, 0])
np.testing.assert_array_equal(z_untied, [0, 1, -1])
np.testing.assert_allclose(p_tied, [0.24472847105479764, 0.6652409557748218, 0.09003057317038046])
assert int(p_tied.argmax()) == 1  # Tied projection is not an inverse lookup.
np.testing.assert_allclose(softmax(z_tied + 10_000), p_tied)
assert np.isclose(softmax(z_untied).sum(), 1.)
print("lookup:", X.tolist())
print("tied / untied logits:", z_tied.tolist(), z_untied.tolist())
print("tied probabilities:", p_tied.round(6).tolist())

# BEGIN GRADIENT
# One learning step on a simple quadratic objective, not next-token cross-entropy.
targets = np.zeros_like(X)


def lookup_loss(table):
    return 0.5 * np.sum((table[ids] - targets) ** 2)


grad = np.zeros_like(E)
np.add.at(grad, ids, X - targets)  # Repeated ID 0 must accumulate twice.
np.testing.assert_array_equal(grad, [[2, 0], [0, 0], [0, 1]])

# Advanced indexing writes through a temporary; repeated IDs do not accumulate.
wrong_grad = np.zeros_like(E)
wrong_grad[ids] += X - targets
np.testing.assert_array_equal(wrong_grad, [[1, 0], [0, 0], [0, 1]])
assert not np.array_equal(wrong_grad, grad)

# Independent finite-difference check catches incorrect repeated-index assignment.
epsilon = 1e-6
numeric_grad = np.zeros_like(E)
for index in np.ndindex(E.shape):
    plus, minus = E.copy(), E.copy()
    plus[index] += epsilon
    minus[index] -= epsilon
    numeric_grad[index] = (lookup_loss(plus) - lookup_loss(minus)) / (2 * epsilon)
np.testing.assert_allclose(grad, numeric_grad, atol=1e-9)
updated_E = E - 0.1 * grad
np.testing.assert_allclose(updated_E, [[0.8, 0], [2, 0], [0, 0.9]])
assert np.isclose(lookup_loss(E), 1.5)
assert np.isclose(lookup_loss(updated_E), 1.045)
assert lookup_loss(updated_E) < lookup_loss(E)
print("gradient:", grad.tolist())
print("loss before / after:", lookup_loss(E), lookup_loss(updated_E))

# BEGIN BUDGETS
# Change these inputs for your experiment; fixed model examples stay below.
vocab_size, dim, tied = 16384, 512, True
budgets = {
    "gpt-oss-120b": embedding_budget(201088, 2880, tied=False),
    "Llama-3.1-8B": embedding_budget(128256, 4096, tied=False),
    "expansion 100000 to 262000": embedding_budget(262000-100000, 4096, tied=False),
    "your configuration": embedding_budget(vocab_size, dim, tied=tied),
}
assert budgets["gpt-oss-120b"]["parameters"] == 1_158_266_880
assert budgets["Llama-3.1-8B"]["bytes"] == 2_101_346_304
assert budgets["expansion 100000 to 262000"]["parameters"] == 1_327_104_000
if (vocab_size, dim, tied) == (16384, 512, True):
    assert budgets["your configuration"]["bytes"] == 16 * 2**20
for name, budget in budgets.items():
    print(name, budget)
