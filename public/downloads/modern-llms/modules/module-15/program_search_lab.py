"""Generate, check, select and stop on a finite polynomial-program task.

No LLM or API is used. The generator samples two unknown coefficients; a visible
example fixes the constant term. Tests execute the candidates rather than drawing
precomputed correctness flags. This isolates selection and accounting.
"""
from dataclasses import dataclass, asdict
import itertools
import json
import random
import re

@dataclass(frozen=True)
class ProgramTask:
    task_id: int
    a: int
    b: int
    c: int
    def target(self, x):
        return self.a*x*x+self.b*x+self.c

TASKS = [ProgramTask(i, *abc) for i, abc in enumerate(itertools.product(range(-2, 3), repeat=3))]
CALIBRATION = [t for t in TASKS if t.c == 0]
HELD_OUT = [t for t in TASKS if t.c != 0]
assert len(CALIBRATION) == 25 and len(HELD_OUT) == 100
assert not {t.task_id for t in CALIBRATION} & {t.task_id for t in HELD_OUT}

def execute_program(source, x):
    # A deliberately restricted grammar; no eval or arbitrary code execution.
    match = re.fullmatch(r"(-?\d+)\*x\*x\+(-?\d+)\*x\+(-?\d+)", source)
    if match is None:
        raise ValueError("not a polynomial in the declared grammar")
    a, b, c = map(int, match.groups())
    return a*x*x+b*x+c

class ProposalGenerator:
    def __init__(self):
        self.calls = 0
    def propose(self, task_id, visible_f0, candidate_index):
        # Same deterministic random stream for each task/policy comparison.
        # Hidden coefficients, test outputs and future candidates are unavailable.
        rng = random.Random(150_000 + task_id*10_000 + candidate_index)
        a, b = rng.randrange(-2, 3), rng.randrange(-2, 3)
        self.calls += 1
        return f"{a}*x*x+{b}*x+{visible_f0}"

class ProgramVerifier:
    def __init__(self):
        self.calls = 0
        self.test_cases = 0
    def score(self, source, examples):
        self.calls += 1
        results = []
        for x, target in examples:
            results.append(execute_program(source, x) == target)
            self.test_cases += 1
        return sum(results)/len(results), results

def search_program(task, *, budget, threshold=None):
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if threshold is not None and threshold not in (.5, 1.):
        raise ValueError("threshold must be .5, 1, or None")
    generator, verifier = ProposalGenerator(), ProgramVerifier()
    # The generator sees only f(0). The verifier receives two more examples.
    examples = [(x, task.target(x)) for x in (1, -1)]
    records = []
    for index in range(budget):
        source = generator.propose(task.task_id, task.target(0), index)
        score, checks = verifier.score(source, examples)
        records.append(dict(index=index, source=source, score=score, checks=checks))
        if threshold is not None and score >= threshold:
            break
    # Deterministic earliest candidate on a score tie. No extra generation or scoring.
    best = max(range(len(records)), key=lambda i: records[i]['score'])
    assert generator.calls == verifier.calls == len(records)
    assert verifier.test_cases == 2*len(records)
    if threshold is not None and len(records) < budget:
        assert records[-1]['score'] >= threshold
        assert all(r['score'] < threshold for r in records[:-1])
    # Hidden tests are evaluation only; they cannot influence selection or stopping.
    before = (generator.calls, verifier.calls, verifier.test_cases)
    hidden = [all(execute_program(r['source'], x) == task.target(x) for x in (2, 3))
              for r in records]
    assert before == (generator.calls, verifier.calls, verifier.test_cases)
    result = dict(task_id=task.task_id, records=records, selected_index=best,
                  correct=hidden[best], observed_coverage=any(hidden),
                  generated=generator.calls, verifier_calls=verifier.calls,
                  visible_test_cases=verifier.test_cases, hidden_tests_max=2*len(records))
    assert json.loads(json.dumps(result)) == result
    return result

def evaluate_program_policy(tasks, *, budget, threshold=None):
    runs = [search_program(t, budget=budget, threshold=threshold) for t in tasks]
    n = len(runs)
    assert n > 0
    return dict(accuracy=sum(r['correct'] for r in runs)/n,
                mean_candidates=sum(r['generated'] for r in runs)/n,
                observed_coverage=sum(r['observed_coverage'] for r in runs)/n,
                generator_calls=sum(r['generated'] for r in runs),
                verifier_calls=sum(r['verifier_calls'] for r in runs),
                visible_test_cases=sum(r['visible_test_cases'] for r in runs)), runs

def run_program_search():
    results = {}
    traces = {}
    for name, budget, threshold in [('one', 1, None), ('fixed16', 16, None),
                                    ('stop_partial', 16, .5), ('stop_complete', 16, 1.)]:
        metrics, records = evaluate_program_policy(HELD_OUT, budget=budget, threshold=threshold)
        results[name], traces[name] = metrics, records
    assert results['fixed16']['generator_calls'] == 1600
    assert results['one']['generator_calls'] == 100
    assert results['stop_complete']['accuracy'] == results['fixed16']['accuracy']
    assert results['stop_complete']['mean_candidates'] < 16
    assert results['stop_partial']['accuracy'] < results['stop_complete']['accuracy']
    # Passing all three x values (one visible to the generator, two to the verifier)
    # identifies a degree-at-most-two polynomial exactly. This guarantee is grammar-specific.
    for name in ('fixed16', 'stop_complete'):
        for record in traces[name]:
            assert (record['records'][record['selected_index']]['score'] == 1) == record['correct']
    for name, result in results.items(): print(name, result)
    return results, traces

if __name__ == '__main__':
    program_results, program_traces = run_program_search()

