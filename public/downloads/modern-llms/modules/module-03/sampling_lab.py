"""A transparent scalar reference for next-token selection policies."""
import math
from numbers import Integral, Real


def token_distribution(logits, temperature=1.0, top_k=None, top_p=None):
    """Return probabilities for one finite logit vector; apply temperature, k, then p."""
    values = list(logits)
    if not values or any(isinstance(x, bool) or not isinstance(x, Real)
                         or not math.isfinite(x) for x in values):
        raise ValueError('logits must be a nonempty vector of finite real numbers')
    if isinstance(temperature, bool) or not isinstance(temperature, Real) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, Integral) or not 1 <= top_k <= len(values)):
        raise ValueError('top_k must be an integer in [1, vocabulary size]')
    if top_p is not None and (isinstance(top_p, bool) or not isinstance(top_p, Real) or not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise ValueError('top_p must lie in (0, 1]')
    maximum = max(values)
    weights = [math.exp((x - maximum) / temperature) for x in values]
    ranked = sorted(range(len(values)), key=lambda i: (-values[i], i))
    candidates = ranked if top_k is None else ranked[:top_k]
    if top_p is not None and top_p < 1:
        mass = math.fsum(weights[i] for i in candidates)
        cumulative = 0.0
        kept = []
        for i in candidates:
            kept.append(i)
            cumulative += weights[i] / mass
            if cumulative >= top_p:
                break
        candidates = kept
    total = math.fsum(weights[i] for i in candidates)
    selected = set(candidates)
    return [weight / total if i in selected else 0.0 for i, weight in enumerate(weights)]


def sampling_scenario():
    """Compute four policies on the same illustrative logits, with filters applied separately."""
    logits = [3.0, 2.0, 1.0, 0.0]
    return {
        'logits': logits,
        'temperature_1': token_distribution(logits),
        'temperature_2': token_distribution(logits, temperature=2),
        'top_k_2': token_distribution(logits, top_k=2),
        'top_p_09': token_distribution(logits, top_p=0.9),
    }
