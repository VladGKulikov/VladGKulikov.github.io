"""Coverage, noisy selection and scoped inference accounting."""
import math
from dataclasses import dataclass


def positive_count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(name + " must be a positive integer")


def probability(value, name):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name + " must be finite and lie in [0, 1]")


def oracle_coverage(p_correct, n):
    """At least one success in n independent, identically distributed attempts."""
    positive_count(n, "n")
    probability(p_correct, "p_correct")
    if p_correct == 1:
        return 1.0
    return -math.expm1(n * math.log1p(-p_correct))


def noisy_selection(p, n, tpr, fpr):
    """Uniform accepted candidate; uniform pool fallback if none are accepted."""
    positive_count(n, "n")
    for name, value in (("p", p), ("tpr", tpr), ("fpr", fpr)):
        probability(value, name)
    acceptance = p*tpr + (1-p)*fpr
    if acceptance == 0 or n == 1:
        return p
    precision = p*tpr/acceptance
    if acceptance == 1:
        return precision
    log_survival = (n-1)*math.log1p(-acceptance)
    # A non-negative mixture avoids cancellation for tiny probabilities.
    return p*math.exp(log_survival) + precision*(-math.expm1(log_survival))


@dataclass(frozen=True)
class ReasoningCost:
    parameters: float = 8.03e9
    tokens_per_second: float = 209.4
    layers: int = 32
    kv_width: int = 1024
    bytes_per_element: int = 2

    def __post_init__(self):
        for name in ("parameters", "tokens_per_second"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in ("layers", "kv_width", "bytes_per_element"):
            positive_count(getattr(self, name), name)

    @property
    def kv_bytes_per_token(self):
        return 2*self.layers*self.kv_width*self.bytes_per_element

    def estimate(self, tokens, prompt_tokens=0):
        positive_count(tokens, "tokens")
        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens < 0:
            raise ValueError("prompt_tokens must be a non-negative integer")
        return dict(decode_seconds=tokens/self.tokens_per_second,
                    parameter_matmul_flops=2*self.parameters*tokens,
                    generated_kv_gib=self.kv_bytes_per_token*tokens/2**30,
                    total_kv_gib=self.kv_bytes_per_token*(tokens+prompt_tokens)/2**30)


scenario = ReasoningCost()
assert abs(noisy_selection(.4, 17, .9, .05)-.923)<.001
assert oracle_coverage(1e-18, 1) > 0  # direct 1-(1-p)**n loses this value
assert noisy_selection(1e-18, 1, 1, 0) == 1e-18
assert math.isclose(noisy_selection(1e-18, 2, 1, 0), 2e-18, rel_tol=1e-15)
