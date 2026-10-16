"""Все проверки капстоуна одной командой: python3 -m minillm.tests_all"""
from __future__ import annotations

import importlib
import time

SUITES = [
    ("архитектура", "minillm.tests"),
    ("обучение", "minillm.tests_train"),
    ("инференс", "minillm.tests_generate"),
    ("данные", "minillm.tests_data"),
    ("посттрейн", "minillm.tests_posttrain"),
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

    print(f"\n{'=' * 72}\nСВОДКА\n{'=' * 72}")
    total_ok = total_fail = 0
    for label, ok, fail, seconds in results:
        total_ok += ok
        total_fail += fail
        print(f"  {label:<14} {ok:>3} пройдено, {fail} провалено   ({seconds:.1f} c)")
    print(f"  {'ИТОГО':<14} {total_ok:>3} пройдено, {total_fail} провалено")
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
