from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from plot_utils import bar_chart
from run_fedids_experiment import (
    N_CLIENTS,
    encode_features,
    family_performance,
    load_data,
    local_train,
    metrics,
    sigmoid,
)


RESULTS = HERE / "results" / "run_fedids_poisoning"
RESULTS.mkdir(parents=True, exist_ok=True)
REPEAT_SEEDS = [73, 193, 307, 421, 557]
N_ROUNDS = 18
DIRICHLET_ALPHA = 0.35
CLIP_NORM = 1.0
ATTACK_SCALE = 5.0


def seeded_partition(families: np.ndarray, seed: int) -> tuple[list[np.ndarray], int]:
    for attempt in range(100):
        actual_seed = seed + attempt * 10007
        rng = np.random.default_rng(actual_seed)
        clients: list[list[int]] = [[] for _ in range(N_CLIENTS)]
        for family in sorted(set(families)):
            indices = np.flatnonzero(families == family)
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.full(N_CLIENTS, DIRICHLET_ALPHA))
            split_points = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
            for client, part in enumerate(np.split(indices, split_points)):
                clients[client].extend(part.tolist())
        arrays = [np.asarray(sorted(client), dtype=int) for client in clients]
        if min(map(len, arrays)) >= 1000:
            return arrays, actual_seed
    raise RuntimeError(f"Could not produce a usable partition for seed {seed}")


def clip_delta(delta_w: np.ndarray, delta_b: float) -> tuple[np.ndarray, float]:
    norm = math.sqrt(float(delta_w @ delta_w) + delta_b**2)
    factor = min(1.0, CLIP_NORM / max(norm, 1e-12))
    return delta_w * factor, delta_b * factor


def run_method(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    clients: list[np.ndarray],
    *,
    method: str,
    seed: int,
) -> tuple[np.ndarray, float, list[dict]]:
    weights = np.zeros(train_x.shape[1], dtype=np.float64)
    bias = 0.0
    history = []
    malicious_client = int(np.argmax([len(indices) for indices in clients]))
    total = sum(map(len, clients))

    for round_number in range(1, N_ROUNDS + 1):
        updates = []
        learning_rate = 0.055 / math.sqrt(round_number)
        for client_id, indices in enumerate(clients):
            local_w, local_b = local_train(
                train_x[indices],
                train_y[indices],
                weights,
                bias,
                learning_rate,
                seed + round_number * 100 + client_id,
            )
            delta_w = local_w - weights
            delta_b = local_b - bias
            if "attacked" in method and client_id == malicious_client:
                delta_w = -ATTACK_SCALE * delta_w
                delta_b = -ATTACK_SCALE * delta_b
            if method == "Clipped FedAvg attacked":
                delta_w, delta_b = clip_delta(delta_w, delta_b)
            updates.append((len(indices), delta_w, delta_b))

        if method == "Coordinate median attacked":
            weights += np.median(np.stack([row[1] for row in updates]), axis=0)
            bias += float(np.median([row[2] for row in updates]))
        else:
            weights += sum(n * delta_w for n, delta_w, _ in updates) / total
            bias += sum(n * delta_b for n, _, delta_b in updates) / total

        test_scores = sigmoid(test_x @ weights + bias)
        round_metrics = metrics(test_y, test_scores)
        history.append(
            {
                "round": round_number,
                "f1": round_metrics["f1"],
                "roc_auc": round_metrics["roc_auc"],
                "false_positive_rate": round_metrics["false_positive_rate"],
            }
        )
    return weights, float(bias), history


def main() -> None:
    train, test = load_data()
    train_x, test_x, _ = encode_features(train, test)
    train_y = train["binary_label"].to_numpy()
    test_y = test["binary_label"].to_numpy()
    methods = [
        "FedAvg benign",
        "FedAvg attacked",
        "Clipped FedAvg attacked",
        "Coordinate median attacked",
    ]

    run_rows = []
    family_rows = []
    partition_rows = []
    histories: dict[str, dict[str, list[dict]]] = {}
    for repeat_index, seed in enumerate(REPEAT_SEEDS, start=1):
        clients, partition_seed = seeded_partition(train["attack_family"].to_numpy(), seed)
        malicious_client = int(np.argmax([len(indices) for indices in clients]))
        for client_id, indices in enumerate(clients):
            subset = train.iloc[indices]
            partition_rows.append(
                {
                    "repeat": repeat_index,
                    "requested_seed": seed,
                    "partition_seed": partition_seed,
                    "client": client_id + 1,
                    "n": int(len(indices)),
                    "attack_rate": float(subset["binary_label"].mean()),
                    "malicious_client": bool(client_id == malicious_client),
                }
            )
        histories[str(repeat_index)] = {}
        for method in methods:
            weights, bias, history = run_method(
                train_x,
                train_y,
                test_x,
                test_y,
                clients,
                method=method,
                seed=seed,
            )
            scores = sigmoid(test_x @ weights + bias)
            row = {
                "repeat": repeat_index,
                "requested_seed": seed,
                "partition_seed": partition_seed,
                "method": method,
                **metrics(test_y, scores),
            }
            run_rows.append(row)
            for family in family_performance(test, scores):
                family_rows.append(
                    {
                        "repeat": repeat_index,
                        "method": method,
                        **family,
                    }
                )
            histories[str(repeat_index)][method] = history
            print(
                f"repeat={repeat_index} method={method} "
                f"f1={row['f1']:.4f} roc_auc={row['roc_auc']:.4f} fpr={row['false_positive_rate']:.4f}",
                flush=True,
            )

    runs = pd.DataFrame(run_rows)
    families = pd.DataFrame(family_rows)
    partitions = pd.DataFrame(partition_rows)
    numeric_metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "false_positive_rate",
    ]
    summary = runs.groupby("method")[numeric_metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    family_summary = families.groupby(["method", "family"])[["recall"]].agg(["mean", "std"]).reset_index()
    family_summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in family_summary.columns
    ]

    runs.to_csv(RESULTS / "fedids_extension_runs.csv", index=False)
    summary.to_csv(RESULTS / "fedids_extension_summary.csv", index=False)
    families.to_csv(RESULTS / "fedids_extension_family_runs.csv", index=False)
    family_summary.to_csv(RESULTS / "fedids_extension_family_summary.csv", index=False)
    partitions.to_csv(RESULTS / "fedids_extension_partitions.csv", index=False)
    (RESULTS / "fedids_extension_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")

    ordered = summary.sort_values("f1_mean", ascending=False)
    bar_chart(
        RESULTS / "fedids_extension_robustness.png",
        "Five-repeat robustness under one reversed-update client",
        ordered["method"].tolist(),
        {
            "Mean F1": ordered["f1_mean"].tolist(),
            "Mean ROC AUC": ordered["roc_auc_mean"].tolist(),
            "Mean balanced accuracy": ordered["balanced_accuracy_mean"].tolist(),
        },
        minimum=0.35,
    )
    selected_families = ["dos", "probe", "r2l", "u2r", "normal"]
    selected = family_summary[family_summary["family"].isin(selected_families)]
    bar_chart(
        RESULTS / "fedids_extension_family_recall.png",
        "Mean family recall across five heterogeneous partitions",
        selected_families,
        {
            method: [
                float(
                    selected[
                        (selected["method"] == method) & (selected["family"] == family)
                    ]["recall_mean"].iloc[0]
                )
                for family in selected_families
            ]
            for method in methods
        },
        minimum=0.0,
    )

    payload = {
        "repeat_seeds": REPEAT_SEEDS,
        "rounds": N_ROUNDS,
        "clients": N_CLIENTS,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "attack": {
            "threat_model": "The largest simulated client reverses and scales its local model update by 5 in every round.",
            "scale": ATTACK_SCALE,
            "clipping_norm": CLIP_NORM,
        },
        "privacy_accountability": {
            "data_locality": "Raw training records remain within simulated client partitions during optimization.",
            "formal_differential_privacy": False,
            "claim_boundary": "No epsilon-delta differential-privacy guarantee is claimed; secure aggregation, authenticated clients, and formal accounting remain deployment requirements.",
        },
        "summary": summary.to_dict(orient="records"),
        "limitations": [
            "NSL-KDD is a historical benchmark and is not operational energy-network traffic.",
            "Clients are simulated partitions of a centralized public dataset rather than separate organizations or devices.",
            "The reversed-update experiment represents one bounded poisoning scenario and is not a complete adversarial evaluation.",
            "Coordinate-wise median robustness with five clients does not establish Byzantine tolerance for arbitrary attacks.",
            "Federated data locality is not equivalent to formal privacy; no differential-privacy guarantee is claimed.",
            "Binary aggregate metrics conceal weak recall for rare attack families.",
        ],
    }
    (RESULTS / "fedids_extension_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(family_summary.to_string(index=False))


if __name__ == "__main__":
    main()
