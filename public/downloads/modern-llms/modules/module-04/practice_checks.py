"""Behavioral checks embedded in the M4 notebook after its NumPy core."""
import numpy as np


def check_cached_positions():
    """Compare full causal attention with cached steps; require wrong IDs to fail."""
    rng = np.random.default_rng(404)
    q, k, v = [rng.standard_normal((7, 8)) for _ in range(3)]
    pos = np.arange(7, dtype=np.int64)
    inv = rope_inverse_frequencies(8, 10_000.0)
    qr = apply_rope_adjacent(q, pos, inv)
    kr = apply_rope_adjacent(k, pos, inv)

    def attend(queries, keys, values, allowed):
        scores = (queries @ keys.T) / np.sqrt(queries.shape[-1])
        scores = np.where(allowed, scores, -np.inf)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        return weights @ values

    full = attend(qr, kr, v, np.tril(np.ones((7, 7), dtype=bool)))
    cache_k, cache_v, steps, wrong = [], [], [], []
    for t in range(7):
        cache_k.append(apply_rope_adjacent(k[t:t+1], [t], inv))
        cache_v.append(v[t:t+1])
        keys, values = np.concatenate(cache_k), np.concatenate(cache_v)
        query = apply_rope_adjacent(q[t:t+1], [t], inv)
        steps.append(attend(query, keys, values, np.ones((1, t+1), dtype=bool)))
        reset = apply_rope_adjacent(q[t:t+1], [0], inv)
        wrong.append(attend(reset, keys, values, np.ones((1, t+1), dtype=bool)))
    cached = np.concatenate(steps)
    np.testing.assert_allclose(cached, full, rtol=1e-12, atol=1e-12)
    wrong_gap = float(np.max(np.abs(np.concatenate(wrong)-full)))
    assert wrong_gap > .01, 'The negative control must expose the reset-position bug'
    # Changing frequencies for Q alone leaves cached K in a different geometry.
    new_inv = inv / 4
    q_new = apply_rope_adjacent(q[-1:], [6], new_inv)
    new_keys = apply_rope_adjacent(k, pos, new_inv)
    mixed = attend(q_new, kr, v, np.ones((1, 7), dtype=bool))
    consistent = attend(q_new, new_keys, v, np.ones((1, 7), dtype=bool))
    spectrum_gap = float(np.max(np.abs(mixed-consistent)))
    assert spectrum_gap > .01
    return {'full_vs_cached_max_error': float(np.max(np.abs(cached-full))),
            'reset_query_position_gap': wrong_gap, 'mixed_spectrum_gap': spectrum_gap}


def check_axis_and_scaling_contracts():
    """Probe one M-RoPE axis and the full-versus-partial amplitude distinction."""
    inv = rope_inverse_frequencies(8, 10_000.0)
    x = np.ones((1, 8))
    first = apply_mrope_adjacent(x, [[0, 1, 2]], (2, 1, 1), inv)
    changed = apply_mrope_adjacent(x, [[0, 1, 3]], (2, 1, 1), inv)
    np.testing.assert_array_equal(first[:, :6], changed[:, :6])
    assert not np.allclose(first[:, 6:], changed[:, 6:])
    q = np.array([1., 2., 3., 4.]); k = np.array([2., -1., 1., 2.]); a = 1.3
    np.testing.assert_allclose((a*q) @ (a*k), a*a*(q@k), rtol=1e-14)
    partial_q, partial_k = q.copy(), k.copy()
    partial_q[2:] *= a; partial_k[2:] *= a
    expected = q[:2]@k[:2] + a*a*(q[2:]@k[2:])
    np.testing.assert_allclose(partial_q@partial_k, expected, rtol=1e-14)
    # Nonzero unrotated contribution is essential for the counterexample.
    k[0] = 3.; partial_k = k.copy(); partial_k[2:] *= a
    assert not np.isclose(partial_q@partial_k, a*a*(q@k))
    return {'only_width_pairs_changed': True, 'partial_rope_is_not_global_scaling': True}
