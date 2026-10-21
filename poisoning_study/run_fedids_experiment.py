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
from plot_utils import bar_chart, line_chart


DATA_ROOT = Path(os.environ.get("NSL_KDD_DIR", HERE.parent / "data" / "nsl_kdd_extracted" / "NSL-KDD-Dataset-master"))
RESULTS = HERE / "results" / "run_fedids_baseline"
RESULTS.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
N_CLIENTS = 5
N_ROUNDS = 24

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
FEATURES = [column for column in COLUMNS if column not in {"label", "difficulty"}]
NUMERIC = [column for column in FEATURES if column not in CATEGORICAL]

ATTACK_FAMILIES = {
    "dos": {"back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb", "apache2", "processtable", "udpstorm"},
    "probe": {"ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"},
    "r2l": {"ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster", "snmpgetattack", "snmpguess", "worm", "xlock", "xsnoop", "sendmail", "named"},
    "u2r": {"buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel", "ps", "sqlattack", "xterm"},
}


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35, 35)))


def attack_family(label: str) -> str:
    if label == "normal":
        return "normal"
    for family, labels in ATTACK_FAMILIES.items():
        if label in labels:
            return family
    return "other_attack"


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = labels == 1
    n_pos = positives.sum()
    n_neg = len(labels) - n_pos
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    precision = cumulative / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / max(sorted_labels.sum(), 1))


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    predicted = (scores >= 0.5).astype(np.int8)
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": average_precision(labels, scores),
        "false_positive_rate": fp / max(fp + tn, 1),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_ROOT / "KDDTrain+.txt", names=COLUMNS)
    test = pd.read_csv(DATA_ROOT / "KDDTest+.txt", names=COLUMNS)
    for frame in (train, test):
        frame["binary_label"] = (frame["label"] != "normal").astype(np.int8)
        frame["attack_family"] = frame["label"].map(attack_family)
    return train, test


def encode_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_numeric = train[NUMERIC].astype(np.float64)
    test_numeric = test[NUMERIC].astype(np.float64)
    mean = train_numeric.mean(axis=0)
    scale = train_numeric.std(axis=0).replace(0, 1)
    train_numeric_array = ((train_numeric - mean) / scale).to_numpy(dtype=np.float32)
    test_numeric_array = ((test_numeric - mean) / scale).to_numpy(dtype=np.float32)

    train_cat = pd.get_dummies(train[CATEGORICAL], dtype=np.float32)
    test_cat = pd.get_dummies(test[CATEGORICAL], dtype=np.float32)
    test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0)
    train_matrix = np.hstack([train_numeric_array, train_cat.to_numpy(dtype=np.float32)])
    test_matrix = np.hstack([test_numeric_array, test_cat.to_numpy(dtype=np.float32)])
    names = NUMERIC + train_cat.columns.tolist()
    return train_matrix, test_matrix, names


def dirichlet_partition(families: np.ndarray, alpha: float) -> list[np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    clients: list[list[int]] = [[] for _ in range(N_CLIENTS)]
    for family in sorted(set(families)):
        indices = np.flatnonzero(families == family)
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(N_CLIENTS, alpha))
        split_points = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
        for client, part in enumerate(np.split(indices, split_points)):
            clients[client].extend(part.tolist())
    arrays = [np.asarray(sorted(client), dtype=int) for client in clients]
    if min(map(len, arrays)) < 1000:
        raise RuntimeError("Undersized simulated client")
    return arrays


def iid_partition(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    return [part.astype(int) for part in np.array_split(rng.permutation(n), N_CLIENTS)]


def local_train(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    learning_rate: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    local_weights = weights.copy()
    local_bias = float(bias)
    positive_weight = len(y) / max(2 * y.sum(), 1)
    negative_weight = len(y) / max(2 * (1 - y).sum(), 1)
    order = rng.permutation(len(y))
    for start in range(0, len(y), 1024):
        indices = order[start : start + 1024]
        probability = sigmoid(x[indices] @ local_weights + local_bias)
        sample_weight = np.where(y[indices] == 1, positive_weight, negative_weight)
        residual = (probability - y[indices]) * sample_weight
        gradient = x[indices].T @ residual / len(indices) + 1e-5 * local_weights
        local_weights -= learning_rate * gradient
        local_bias -= learning_rate * float(residual.mean())
    return local_weights, local_bias


def run_federated(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    clients: list[np.ndarray],
    *,
    noise_std: float | None,
) -> tuple[np.ndarray, float, list[dict]]:
    rng = np.random.default_rng(RANDOM_SEED + (100 if noise_std else 0))
    weights = np.zeros(train_x.shape[1], dtype=np.float64)
    bias = 0.0
    history = []
    total = sum(map(len, clients))
    for round_number in range(1, N_ROUNDS + 1):
        local_models = []
        learning_rate = 0.055 / math.sqrt(round_number)
        for client_id, indices in enumerate(clients):
            local_w, local_b = local_train(
                train_x[indices],
                train_y[indices],
                weights,
                bias,
                learning_rate,
                RANDOM_SEED + round_number * 100 + client_id,
            )
            delta_w = local_w - weights
            delta_b = local_b - bias
            if noise_std is not None:
                norm = math.sqrt(float(delta_w @ delta_w) + delta_b**2)
                scale = min(1.0, 1.0 / max(norm, 1e-12))
                delta_w *= scale
                delta_b *= scale
            local_models.append((len(indices), weights + delta_w, bias + delta_b))
        weights = sum(n * w for n, w, _ in local_models) / total
        bias = sum(n * b for n, _, b in local_models) / total
        if noise_std is not None:
            weights += rng.normal(0, noise_std / N_CLIENTS, size=weights.shape)
            bias += float(rng.normal(0, noise_std / N_CLIENTS))
        test_scores = sigmoid(test_x @ weights + bias)
        history.append(
            {
                "round": round_number,
                "test_f1": metrics(test_y, test_scores)["f1"],
                "test_roc_auc": metrics(test_y, test_scores)["roc_auc"],
            }
        )
    return weights, float(bias), history


def run_centralized(train_x: np.ndarray, train_y: np.ndarray) -> tuple[np.ndarray, float]:
    weights = np.zeros(train_x.shape[1], dtype=np.float64)
    bias = 0.0
    for epoch in range(1, 25):
        weights, bias = local_train(
            train_x,
            train_y,
            weights,
            bias,
            0.045 / math.sqrt(epoch),
            RANDOM_SEED + epoch,
        )
    return weights, bias


def family_performance(test: pd.DataFrame, scores: np.ndarray) -> list[dict]:
    predicted = (scores >= 0.5).astype(np.int8)
    rows = []
    for family in ["dos", "probe", "r2l", "u2r", "other_attack"]:
        mask = test["attack_family"].to_numpy() == family
        if mask.any():
            rows.append({"family": family, "n": int(mask.sum()), "recall": float(np.mean(predicted[mask] == 1))})
    normal = test["attack_family"].to_numpy() == "normal"
    rows.append({"family": "normal", "n": int(normal.sum()), "recall": float(np.mean(predicted[normal] == 0))})
    return rows


def main() -> None:
    train, test = load_data()
    train_x, test_x, feature_names = encode_features(train, test)
    train_y = train["binary_label"].to_numpy()
    test_y = test["binary_label"].to_numpy()
    non_iid_clients = dirichlet_partition(train["attack_family"].to_numpy(), 0.35)
    iid_clients = iid_partition(len(train))

    central_w, central_b = run_centralized(train_x, train_y)
    central_scores = sigmoid(test_x @ central_w + central_b)
    experiments: dict[str, dict] = {
        "Centralized": {**metrics(test_y, central_scores), "attack_family_performance": family_performance(test, central_scores)}
    }
    histories = {}
    for name, clients, noise in (
        ("FedAvg IID", iid_clients, None),
        ("FedAvg non-IID", non_iid_clients, None),
        ("Clipped/noisy non-IID", non_iid_clients, 0.02),
    ):
        weights, bias, history = run_federated(train_x, train_y, test_x, test_y, clients, noise_std=noise)
        scores = sigmoid(test_x @ weights + bias)
        experiments[name] = {**metrics(test_y, scores), "attack_family_performance": family_performance(test, scores)}
        histories[name] = history

    metric_rows = []
    for method, values in experiments.items():
        metric_rows.append(
            {
                "method": method,
                **{key: values[key] for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate"]},
            }
        )
    metric_table = pd.DataFrame(metric_rows).sort_values("f1", ascending=False)
    metric_table.to_csv(RESULTS / "fedids_summary_results.csv", index=False)

    client_rows = []
    for client_id, indices in enumerate(non_iid_clients, start=1):
        subset = train.iloc[indices]
        counts = subset["attack_family"].value_counts()
        client_rows.append(
            {
                "client": client_id,
                "n": len(indices),
                "attack_rate": float(subset["binary_label"].mean()),
                **{family: int(counts.get(family, 0)) for family in sorted(train["attack_family"].unique())},
            }
        )
    pd.DataFrame(client_rows).to_csv(RESULTS / "fedids_non_iid_clients.csv", index=False)

    family_rows = []
    for method, values in experiments.items():
        for row in values["attack_family_performance"]:
            family_rows.append({"method": method, **row})
    family_table = pd.DataFrame(family_rows)
    family_table.to_csv(RESULTS / "fedids_attack_family_results.csv", index=False)

    methods = metric_table["method"].tolist()
    bar_chart(
        RESULTS / "fedids_model_comparison.png",
        "NSL-KDD test performance",
        methods,
        {
            "Balanced accuracy": metric_table["balanced_accuracy"].tolist(),
            "F1": metric_table["f1"].tolist(),
            "ROC AUC": metric_table["roc_auc"].tolist(),
        },
        minimum=0.45,
    )
    line_chart(
        RESULTS / "fedids_convergence.png",
        "Federated convergence under IID and non-IID partitions",
        list(range(1, N_ROUNDS + 1)),
        {name: [row["test_f1"] for row in history] for name, history in histories.items()},
        minimum=0.45,
    )
    selected_family = family_table[family_table["method"].isin(["Centralized", "FedAvg non-IID", "Clipped/noisy non-IID"])]
    families = ["dos", "probe", "r2l", "u2r", "normal"]
    bar_chart(
        RESULTS / "fedids_family_performance.png",
        "Performance varies materially by traffic family",
        families,
        {
            method: [
                float(selected_family[(selected_family["method"] == method) & (selected_family["family"] == family)]["recall"].iloc[0])
                for family in families
            ]
            for method in ["Centralized", "FedAvg non-IID", "Clipped/noisy non-IID"]
        },
        minimum=0.0,
    )

    train_attack_types = set(train.loc[train["label"] != "normal", "label"])
    test_attack_types = set(test.loc[test["label"] != "normal", "label"])
    parameter_count = train_x.shape[1] + 1
    communication_bytes = N_ROUNDS * N_CLIENTS * 2 * parameter_count * 8
    payload = {
        "random_seed": RANDOM_SEED,
        "dataset": {
            "name": "NSL-KDD",
            "train_records": int(len(train)),
            "test_records": int(len(test)),
            "train_attack_records": int(train_y.sum()),
            "test_attack_records": int(test_y.sum()),
            "train_attack_types": len(train_attack_types),
            "test_attack_types": len(test_attack_types),
            "test_only_attack_types": sorted(test_attack_types - train_attack_types),
            "encoded_features": int(train_x.shape[1]),
        },
        "federated_design": {
            "clients": N_CLIENTS,
            "rounds": N_ROUNDS,
            "non_iid_dirichlet_alpha": 0.35,
            "local_epochs_per_round": 1,
            "batch_size": 1024,
            "estimated_bidirectional_communication_bytes": int(communication_bytes),
            "estimated_bidirectional_communication_megabytes": communication_bytes / 1_000_000,
            "noise_stress_test": "Client updates clipped to L2 norm 1.0; Gaussian standard deviation 0.02 divided by five clients added after aggregation. No epsilon privacy guarantee is claimed.",
        },
        "experiments": experiments,
        "client_partitions": client_rows,
        "limitations": [
            "NSL-KDD is an older benchmark and is not a substitute for validation on current operational energy-network traffic.",
            "The study simulates clients by partitioning a centralized public dataset; it does not involve real organizations or deployed edge devices.",
            "Federated learning keeps raw records local in the simulation but does not by itself guarantee privacy.",
            "The clipped/noisy experiment is a utility stress test, not a formally accounted differential-privacy mechanism.",
            "Binary detection results conceal substantial variation across rare attack families.",
        ],
    }
    (RESULTS / "fedids_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RESULTS / "fedids_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    print(metric_table.to_string(index=False))
    print(pd.DataFrame(client_rows).to_string(index=False))
    print(json.dumps(payload["dataset"], indent=2))


if __name__ == "__main__":
    main()
