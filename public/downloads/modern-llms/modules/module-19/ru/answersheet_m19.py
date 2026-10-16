#!/usr/bin/env python3
"""Лист проверки численных сценариев модуля 19.

По умолчанию скрипт заново вычисляет и проверяет замороженный контракт, не изменяя
файлы. Флаг ``--write`` следует использовать только после проверки изменений в
``reference_calculations.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))
from reference_calculations import (  # noqa: E402
    REFERENCE_PATH,
    build_reference_payload,
    write_reference,
)


def main() -> None:
    """Print the numerical contract and optionally rewrite its JSON artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="перезаписать data/reference_metrics.json")
    args = parser.parse_args()

    calculated = build_reference_payload()
    if REFERENCE_PATH.exists() and not args.write:
        frozen = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        if calculated != frozen:
            raise AssertionError(
                "замороженный reference_metrics.json расходится с расчётом; проверьте diff и используйте --write явно"
            )
    if args.write:
        write_reference()

    leaderboard = calculated["leaderboard"]
    aggregate = leaderboard["aggregate_approximation"]
    paired = leaderboard["paired_demo"]
    passk = calculated["pass_at_k"]
    judge = calculated["judge"]
    litm = calculated["lost_in_middle"]
    golden = calculated["golden"]

    print("== A. Агрегированная аппроксимация и парное сравнение ==")
    print(
        f"aggregate delta={aggregate['delta']*100:.1f} pp, "
        f"SE={aggregate['se']*100:.2f} pp, z={aggregate['z_score']:.2f}, "
        f"95% CI=[{aggregate['ci_low']*100:.1f}, {aggregate['ci_high']*100:.1f}] pp"
    )
    print(
        f"paired demo: {paired['first_accuracy']:.0%} vs {paired['second_accuracy']:.0%}; "
        f"discordant={paired['first_only']}:{paired['second_only']}; "
        f"exact McNemar p={paired['mcnemar_p']:.3f}"
    )
    print("число задач для полуширины интервала в независимой модели:", leaderboard["required_tasks"])

    print("\n== B. pass@k ==")
    for k in (1, 5, 10):
        unbiased = passk["unbiased"][str(k)]
        naive = passk["naive"][str(k)]
        print(f"pass@{k}: unbiased={unbiased:.4f}, подстановочная={naive:.4f}, delta={(naive-unbiased)*100:+.1f} pp")

    print("\n== C. Протокол модельного судьи ==")
    print(
        f"A first: {judge['forward_a_wins']}/{judge['pairs']}; A second: "
        f"{judge['swapped_a_wins']}/{judge['pairs']}; averaged={judge['averaged_a_rate']:.1%}; "
        f"calls/pair={judge['calls_per_pair']}"
    )

    print("\n== D. Синтетический рабочий контекст ==")
    print(
        f"mean={litm['mean']:.3f}; share={litm['working_share']:.3%}; "
        f"tokens of nominal 1M={litm['working_tokens_of_1m']:,}"
    )

    print("\n== E. Планирование золотого набора и условная арифметическая оценка ==")
    cost = golden["cost"]
    print(
        f"N(±3pp)={golden['n_for_3pp']}; N(±5pp)={golden['n_for_5pp']}; "
        f"canary chance={golden['canary_guess_probability']:.2e}"
    )
    print(
        f"tokens={cost['total_tokens']:,}; оценка FLOP={cost['dense_flops_floor']:.3e}; "
        f"условный эквивалент, секунд={cost['seconds_floor']:.2f}"
    )
    print("\nконтракт проверен; файлы не изменены" if not args.write else f"записан {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
