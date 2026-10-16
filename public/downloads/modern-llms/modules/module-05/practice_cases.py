"""Self-contained experiments embedded in RU/EN notebooks, also run by validation."""

TEXT = {
    'ru': {
        'cache': '## 7. Запрос с заполненным кэшем\n\nНомер строки запроса может отличаться от позиции токена. Для запроса в позиции 5 ключи 0–5 уже доступны. Сверим целую последовательность, отдельный токен и блок новых токенов; затем воспроизведём ошибку маски без смещения.',
        'mla': '## 8. MLA: перенести проекцию на сторону запроса\n\n`C` хранит общий латент, `A` восстанавливает ключи, `B` — значения. Сравним восстановление K/V с вычислением в пространстве латента, включая отдельную позиционную ветвь и выходную проекцию. Это проверка алгебры MLA на малых массивах; размеры выбраны для опыта.',
    },
    'en': {
        'cache': '## 7. Queries after a cached prefix\n\nA query row number need not equal its token position. A query at position 5 can see keys 0–5. Compare full-sequence, single-token and chunked execution, then reproduce the missing-offset mask bug.',
        'mla': '## 8. MLA: moving a projection to the query side\n\n`C` holds the shared latent; `A` and `B` expand keys and values. Compare explicit K/V reconstruction with latent-space computation, including a separate positional branch and an output projection. This small experiment tests MLA algebra rather than a complete model.',
    },
}

CASES = {
'cache': r'''
rng_cache = np.random.default_rng(505)
Qp = rng_cache.normal(size=(9, 6))
Kp = rng_cache.normal(size=(9, 6))
Vp = rng_cache.normal(size=(9, 4))
for window in (None, 3):
    complete = sdpa(Qp, Kp, Vp, make_mask(9, window))
    for prefix, count in ((0, 9), (5, 1), (4, 3)):
        end = prefix + count
        distance = np.arange(prefix, end)[:, None] - np.arange(end)[None, :]
        allowed = distance >= 0
        if window is not None:
            allowed &= distance < window
        mask = np.where(allowed, 0.0, -np.inf)
        direct = sdpa(Qp[prefix:end], Kp[:end], Vp[:end], mask)
        stream = tiled_attention(Qp[prefix:end], Kp[:end], Vp[:end],
            causal=True, query_start=prefix, window=window, tile_size=2)
        np.testing.assert_allclose(direct, complete[prefix:end], atol=1e-12)
        np.testing.assert_allclose(stream, direct, atol=1e-12)
wrong = sdpa(Qp[5:6], Kp[:6], Vp[:6], np.where(np.tri(1, 6), 0.0, -np.inf))
assert not np.allclose(wrong, sdpa(Qp[5:6], Kp[:6], Vp[:6]))
# Empty tiles must be skipped without forming -inf - (-inf).
late_mask = np.full((1, 9), -np.inf)
late_mask[0, -1] = 0
np.testing.assert_allclose(tiled_attention(Qp[:1], Kp, Vp, late_mask, tile_size=2), Vp[-1:])
print('Cached prefix, chunked decode, window, empty tiles: PASS')
''',
'mla': r'''
rng_mla = np.random.default_rng(506)
H, Tq, Tk, dc, dk, dv, dr = 2, 3, 7, 4, 6, 5, 2
C = rng_mla.normal(size=(Tk, dc))
A = rng_mla.normal(size=(H, dk, dc))
B = rng_mla.normal(size=(H, dv, dc))
q = rng_mla.normal(size=(H, Tq, dk))
qr = rng_mla.normal(size=(H, Tq, dr))  # already rotated query branch
kr = rng_mla.normal(size=(Tk, dr))     # shared, already rotated key branch
Wout = rng_mla.normal(size=(H, dv, 8))
explicit, absorbed = [], []
for h in range(H):
    expanded_k, expanded_v = C @ A[h].T, C @ B[h].T
    scores = (q[h] @ expanded_k.T + qr[h] @ kr.T) / np.sqrt(dk + dr)
    latent_scores = ((q[h] @ A[h]) @ C.T + qr[h] @ kr.T) / np.sqrt(dk + dr)
    np.testing.assert_allclose(scores, latent_scores, atol=1e-12)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    explicit.append((p @ expanded_v) @ Wout[h])
    absorbed.append((p @ C) @ (B[h].T @ Wout[h]))
np.testing.assert_allclose(sum(explicit), sum(absorbed), atol=1e-12)
# A rectangular A makes R @ A and A @ R different *shapes*, not a commutator.
# The correct obstruction: A.T @ R(n-m).T @ q changes with key position n.
def R(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
A_small = np.array([[1., 2., 0.], [0., 1., 3.]])
q_small = np.array([1., .5])
c_small = np.array([.2, -.4, .7])
m, n = .3, 1.1
rotated_score = (R(m) @ q_small) @ (R(n) @ A_small @ c_small)
effective_q = A_small.T @ R(n-m).T @ q_small
np.testing.assert_allclose(rotated_score, effective_q @ c_small, atol=1e-12)
assert not np.allclose(effective_q, A_small.T @ R(.8-m).T @ q_small)
print('MLA key/value/output absorption and position dependence: PASS')
''',
}
