"""Additional executable experiments embedded in the bilingual M6 notebooks."""
CASES = {
    'scale_and_axis': r'''
# With eps>0, scale invariance is approximate and fails near zero.
small = np.array([[1e-5, -2e-5]])
assert np.linalg.norm(rms_norm(100*small)-rms_norm(small)) > 0.5
zero = np.zeros((2,4), dtype=np.float64)
np.testing.assert_array_equal(rms_norm(zero), zero)
constant = np.full((2,4), 3.0)
np.testing.assert_array_equal(layer_norm(constant), zero)
assert np.all(rms_norm(constant) > 0.99)

# Two heads, two features each. Changing head 1 changes head 0 only with a joint denominator.
q = np.array([[1.,2.],[3.,4.]])
changed = q.copy(); changed[1] *= 10
per_head = rms_norm(q); per_head_changed = rms_norm(changed)
joint = rms_norm(q.reshape(1,-1)).reshape(2,2)
joint_changed = rms_norm(changed.reshape(1,-1)).reshape(2,2)
np.testing.assert_array_equal(per_head[0], per_head_changed[0])
assert not np.allclose(joint[0], joint_changed[0])
print('eps and normalization-axis counterexamples: PASS')
''',
    'mixing': r'''
logits = np.array([[5.,1.,-2.],[0.,3.,-1.],[2.,0.,1.]])
records = []
for steps in (1,20,100):
    a = sinkhorn_mix(logits, steps)
    row_error = float(np.max(np.abs(a.sum(1)-1)))
    column_error = float(np.max(np.abs(a.sum(0)-1)))
    records.append((steps,row_error,column_error))
a = sinkhorn_mix(logits,100)
np.testing.assert_allclose(a.sum(0),1,atol=1e-10)
np.testing.assert_allclose(a.sum(1),1,atol=1e-10)
assert np.all(a>=0)
streams = np.array([[2.,-3.],[4.,7.],[-1.,5.]])
mixed = a @ streams
assert np.all(mixed <= streams.max(0)+1e-9)
assert np.all(mixed >= streams.min(0)-1e-9)
assert not np.allclose(mixed,streams)  # Doubly stochastic does not mean identity.
# This tests a fixed mixing matrix, not a full mHC layer or its input-dependent Jacobian.
print('Sinkhorn (iterations, row error, column error):', records)
''',
}
