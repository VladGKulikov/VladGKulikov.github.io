#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Доступный только для чтения лист решений М14; --write обновляет замороженный эталон."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))
from reference_calculations import REFERENCE_PATH, build_reference, write_reference  # noqa: E402


def main() -> None:
    """Проверить сохранённый эталон и вывести основные числа лекции."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite data/reference_metrics.json")
    args = parser.parse_args()

    expected = build_reference()
    if args.write:
        path = write_reference()
        print(f"эталон обновлён: {path}")
    else:
        committed = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        if committed != expected:
            raise AssertionError("reference_metrics.json diverged; review the diff and rerun with --write")
        print(f"эталон проверен без перезаписи: {REFERENCE_PATH}")

    bt = expected["bt"]
    print("\n== A. Bradley–Terry ==")
    for delta, probability in zip(bt["deltas"], bt["probabilities"]):
        print(f"Δr={delta:.1f} -> P={probability:.3f}")
    print(f"75% побед → Δr=ln 3={bt['inverse_margin']:.3f}")

    print("\n== B. DPO ==")
    for name, case in expected["dpo"]["scalar_cases"].items():
        print(
            f"{name}: отступ={case['margin']:.3f} · функция потерь={case['loss']:.4f} · "
            f"локальный вес={case['weight']:.4f}"
        )
    batch = expected["dpo"]["batch"]
    print(f"средняя функция потерь пакета={batch['mean_loss']:.6f} · правильность={batch['accuracy']}")

    print("\n== C. GRPO и базовые линии ==")
    grpo = expected["grpo"]
    print(
        f"среднее={grpo['mean']:.3f} · стандартное отклонение={grpo['std']:.3f} · "
        f"advantages={grpo['advantages']}"
    )
    print(f"P(вырождение)={100*grpo['degenerate_probability']:.2f}%")
    baseline = expected["baseline_estimator"]
    print(
        f"точный градиент Бернулли={baseline['true_gradient']:.5f} · "
        f"ожидание при среднем всей группы={baseline['full_group_mean_expected']:.5f}"
    )

    print("\n== D. Память и сценарий генерации траекторий ==")
    memory = expected["memory"]
    print(
        f"обучаемая={memory['train_gib']:.1f} GiB · замороженная={memory['frozen_gib']:.2f} GiB · "
        f"PPO={memory['ppo_gib']:.1f} GiB · DPO/GRPO={memory['dpo_grpo_gib']:.1f} GiB"
    )
    rollout = expected["rollout"]
    print(
        f"токены={rollout['generated_tokens']/1e6:.2f} млн · генерация={rollout['generation_minutes']:.1f} мин · "
        f"обучение={rollout['training_minutes']:.1f} мин · отношение={rollout['ratio']:.1f}:1"
    )


if __name__ == "__main__":
    main()
