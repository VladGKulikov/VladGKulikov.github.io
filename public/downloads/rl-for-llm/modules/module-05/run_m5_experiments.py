#!/usr/bin/env python3
"""Reproduce the numerical artifacts used in Module 5 figures and notebooks."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from module_5_training import (
    KeyDoorGrid,
    collect_offline_dataset,
    dagger_experiment,
    evaluate_decision_transformer,
    offline_experiment,
    train_decision_transformer,
    train_online_dqn,
)
from module_5_language import language_offline_experiment, token_text, PROMPTS

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)


def main() -> None:
    dagger = dagger_experiment(
        seed=17,
        expert_episodes=4,
        rounds=3,
        learner_episodes_per_round=8,
        epochs=12,
        evaluation_episodes=24,
    )
    np.savez_compressed(
        DATA / "M5_dagger_curve.npz",
        rounds=np.asarray([row["round"] for row in dagger["records"]]),
        dataset_size=np.asarray([row["dataset_size"] for row in dagger["records"]]),
        mean_return=np.asarray([row["mean_return"] for row in dagger["records"]]),
        success_rate=np.asarray([row["success_rate"] for row in dagger["records"]]),
    )

    dqn = train_online_dqn(seed=5, episodes=130, double=False, evaluation_episodes=24)
    double = train_online_dqn(seed=5, episodes=130, double=True, evaluation_episodes=24)
    np.savez_compressed(
        DATA / "M5_online_dqn_curve.npz",
        dqn_returns=dqn["returns"],
        dqn_success=dqn["success"],
        double_returns=double["returns"],
        double_success=double["success"],
    )

    offline = offline_experiment(
        seed=41,
        dataset_episodes=80,
        train_epochs=16,
        evaluation_episodes=24,
    )
    method_names = list(offline["results"])
    np.savez_compressed(
        DATA / "M5_offline_method_results.npz",
        method_names=np.asarray(method_names),
        mean_return=np.asarray([offline["results"][name]["mean_return"] for name in method_names]),
        success_rate=np.asarray([offline["results"][name]["success_rate"] for name in method_names]),
        mean_length=np.asarray([offline["results"][name]["mean_length"] for name in method_names]),
        action_counts=np.asarray(offline["action_counts"]),
        fqe_names=np.asarray(list(offline["fqe"])),
        fqe_values=np.asarray(list(offline["fqe"].values())),
    )

    dt_episodes, _ = collect_offline_dataset(seed=41, episodes=120, profile="mixed")
    dt_model, dt_loss = train_decision_transformer(
        dt_episodes,
        seed=61,
        state_dim=KeyDoorGrid().observation_dim,
        action_dim=KeyDoorGrid().action_dim,
        context_steps=16,
        epochs=12,
        batch_size=96,
    )
    targets = np.asarray([-0.5, 0.2, 0.7, 1.2], dtype=np.float32)
    dt_eval = [
        evaluate_decision_transformer(
            KeyDoorGrid(seed=41),
            dt_model,
            target_return=float(target),
            episodes=6,
            seed=2000,
            context_steps=16,
        )
        for target in targets
    ]
    np.savez_compressed(
        DATA / "M5_decision_transformer_results.npz",
        train_loss=np.asarray(dt_loss),
        target_return=targets,
        mean_return=np.asarray([row["mean_return"] for row in dt_eval]),
        success_rate=np.asarray([row["success_rate"] for row in dt_eval]),
    )

    language = language_offline_experiment(
        seed=71,
        episodes_per_prompt=45,
        bc_epochs=12,
        iql_epochs=20,
    )
    language_names = list(language["evaluation"])
    np.savez_compressed(
        DATA / "M5_language_ilql_results.npz",
        method_names=np.asarray(language_names),
        mean_reward=np.asarray([language["evaluation"][name]["mean_reward"] for name in language_names]),
        exact_match=np.asarray([language["evaluation"][name]["exact_match"] for name in language_names]),
    )

    protocol = {
        "artifact_scope": "Reproduced synthetic course experiments; not external benchmarks.",
        "snapshot": "2026-08-05",
        "dagger": {
            "seed": 17,
            "expert_episodes": 4,
            "rounds": 3,
            "learner_episodes_per_round": 8,
            "training_epochs_per_round": 12,
            "evaluation_episodes": 24,
            "records": dagger["records"],
            "warning": "One controlled synthetic run illustrates state-distribution correction; it is not an external DAgger benchmark.",
        },
        "online_dqn": {
            "seed": 5,
            "episodes": 130,
            "evaluation_episodes": 24,
            "replay_warm_start": int(dqn["prefill_episodes"]),
            "update_every_environment_steps": int(dqn["update_every"]),
            "dqn": {
                "evaluation": dqn["evaluation"],
                "last_30_training_success": float(np.mean(dqn["success"][-30:])),
            },
            "double_dqn": {
                "evaluation": double["evaluation"],
                "last_30_training_success": float(np.mean(double["success"][-30:])),
            },
            "warning": "One controlled seed illustrates mechanics and variability; it is not evidence that Double DQN universally outperforms DQN.",
        },
        "offline_rl": {
            "seed": 41,
            "dataset_episodes": 80,
            "training_epochs": 16,
            "evaluation_episodes": 24,
            "dataset_return_mean": offline["dataset_return_mean"],
            "dataset_return_median": offline["dataset_return_median"],
            "cql_alpha": 0.15,
            "bcq_support_ratio": 0.35,
            "iql_expectile": 0.75,
            "action_counts": offline["action_counts"].tolist(),
            "results": offline["results"],
            "fqe": offline["fqe"],
            "warning": "The environment and hyperparameters are pedagogical. Method rankings are not claimed to generalize.",
        },
        "decision_transformer": {
            "seed_data": 41,
            "seed_model": 61,
            "dataset_episodes": 120,
            "context_steps": 16,
            "epochs": 12,
            "evaluation_episodes_per_target": 6,
            "targets": targets.tolist(),
            "mean_return": [row["mean_return"] for row in dt_eval],
            "success_rate": [row["success_rate"] for row in dt_eval],
            "warning": "Target return is a conditioning variable, not a guarantee; this run did not produce monotone extrapolation outside the data range.",
        },
        "language_ilql_like": {
            "seed": 71,
            "episodes_per_prompt": 45,
            "bc_epochs": 12,
            "iql_epochs": 20,
            "dataset": language["dataset"],
            "evaluation": {
                name: {
                    "mean_reward": row["mean_reward"],
                    "exact_match": row["exact_match"],
                    "generations": [
                        {"prompt": prompt, "tokens": token_text(tokens), "reward": reward}
                        for prompt, tokens, reward in zip(PROMPTS, row["sequences"], row["rewards"])
                    ],
                }
                for name, row in language["evaluation"].items()
            },
            "limitation": "Educational neural prefix/action implementation of IQL/ILQL mechanics; not a reproduction of the full ILQL paper system.",
        },
    }
    (DATA / "M5_experiment_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
