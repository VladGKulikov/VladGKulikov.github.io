"""Two explicit AdamW steps, with supplied gradients and no training framework."""
import numpy as np


def adamw_trace():
    theta = np.array([1., -2.])
    m, v = np.zeros(2), np.zeros(2)
    eta, decay, beta1, beta2, eps = .01, .1, .9, .999, 1e-8
    history = [theta.copy()]
    for step, gradient in enumerate((np.array([.2, -.4]), np.array([-.1, .2])), 1):
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2
        m_hat, v_hat = m / (1 - beta1**step), v / (1 - beta2**step)
        theta = (1 - eta * decay) * theta - eta * m_hat / (np.sqrt(v_hat) + eps)
        history.append(theta.copy())
    return np.array(history)


trace = adamw_trace()
np.testing.assert_allclose(trace[1], [.9890000005, -1.98800000025], atol=1e-12, rtol=0)
# The second gradient reverses, but accumulated momentum retains its old sign.
assert trace[2, 0] < trace[1, 0] and trace[2, 1] > trace[1, 1]
print("AdamW parameter trace:", trace)
