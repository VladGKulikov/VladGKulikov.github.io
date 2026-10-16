"""Run every capstone check with one command: python3 -m minillm.tests_all"""
from __future__ import annotations

import importlib
import time

SUITES = [
    ("architecture", "minillm.tests"),
    ("training", "minillm.tests_train"),
    ("inference", "minillm.tests_generate"),
    ("data", "minillm.tests_data"),
    ("post-training", "minillm.tests_posttrain"),
]


def main() -> int:
    results = []
    for label, module_name in SUITES:
        print(f"\n{'=' * 72}\n{label.upper()}\n{'=' * 72}")
        module = importlib.import_module(module_name)
        module.PASSED.clear()
        module.FAILED.clear()
        start = time.time()
        module.main()
        results.append((label, len(module.PASSED), len(module.FAILED), time.time() - start))

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    total_ok = total_fail = 0
    for label, ok, fail, seconds in results:
        total_ok += ok
        total_fail += fail
        print(f"  {label:<14} {ok:>3} passed, {fail} failed   ({seconds:.1f} s)")
    print(f"  {'TOTAL':<14} {total_ok:>3} passed, {total_fail} failed")
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
