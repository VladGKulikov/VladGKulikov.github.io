"""Independent tensor and enumeration checks for the scalar teaching sampler."""
import math
import random
from collections import Counter

import torch
from sampling_lab import token_distribution, sampling_scenario


def check_sampling_contract():
    scenario = sampling_scenario()
    logits = torch.tensor(scenario['logits'], dtype=torch.float64)
    expected = {
        'temperature_1': torch.softmax(logits, dim=0),
        'temperature_2': torch.softmax(logits / 2, dim=0),
        'top_k_2': torch.softmax(torch.tensor([3., 2., -math.inf, -math.inf], dtype=torch.float64), dim=0),
        'top_p_09': torch.softmax(torch.tensor([3., 2., 1., -math.inf], dtype=torch.float64), dim=0),
    }
    for key, vector in expected.items():
        torch.testing.assert_close(torch.tensor(scenario[key], dtype=torch.float64), vector, atol=1e-14, rtol=0)
    # A known rational distribution checks the inclusive nucleus boundary.
    assert token_distribution([0., 0., 0., 0.], top_p=0.5) == [0.5, 0.5, 0., 0.]
    assert token_distribution([0., 0., 0.], top_k=1) == [1., 0., 0.]
    assert token_distribution([0.], top_p=0.01) == [1.]
    assert token_distribution([3., 2., 1.], top_p=1) == token_distribution([3., 2., 1.])
    assert token_distribution([1e308, -1e308]) == [1., 0.]
    # Top-p acts on the distribution renormalized after top-k in this implementation.
    assert token_distribution([3., 2., 1., 0.], top_k=2, top_p=0.7) == [1., 0., 0., 0.]
    probabilities = scenario['top_p_09']
    rng = random.Random(7)
    draws = rng.choices(range(4), weights=probabilities, k=20_000)
    counts = Counter(draws)
    frequencies = [counts[i] / len(draws) for i in range(4)]
    assert max(abs(a-b) for a, b in zip(frequencies, probabilities)) < 0.015
    invalid = [([], {}), ([math.nan], {}), ([math.inf], {}), ([True], {}),
               ([0., 1.], {'temperature': 0}), ([0., 1.], {'temperature': math.inf}),
               ([0., 1.], {'top_k': 0}), ([0., 1.], {'top_k': 3}),
               ([0., 1.], {'top_k': True}), ([0., 1.], {'top_p': 0}),
               ([0., 1.], {'top_p': 1.1}), ([0., 1.], {'top_p': math.nan})]
    for values, options in invalid:
        try:
            token_distribution(values, **options)
        except ValueError:
            pass
        else:
            raise AssertionError((values, options))
    return {'status': 'PASS', 'scenario': scenario, 'draws': len(draws),
            'empirical_top_p': frequencies, 'invalid_inputs_rejected': len(invalid)}


if __name__ == '__main__':
    import json
    print(json.dumps(check_sampling_contract(), indent=2))
