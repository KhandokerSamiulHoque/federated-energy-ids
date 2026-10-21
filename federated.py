"""Transparent parameter-averaged logistic federated learning utilities."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [column for column in NSL_COLUMNS[:-2] if column not in CATEGORICAL]


def load_nsl(root: Path):
    archive_path = root / "data" / "NSL-KDD-Dataset-master.zip"
    extract_dir = root / "data" / "nsl_kdd_extracted"
    train_path = extract_dir / "NSL-KDD-Dataset-master" / "KDDTrain+.txt"
    test_path = extract_dir / "NSL-KDD-Dataset-master" / "KDDTest+.txt"
    if not train_path.exists() or not test_path.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)
    train = pd.read_csv(train_path, names=NSL_COLUMNS)
    test = pd.read_csv(test_path, names=NSL_COLUMNS)
    y_train = (train["label"] != "normal").astype(np.int8).to_numpy()
    y_test = (test["label"] != "normal").astype(np.int8).to_numpy()
    return train, test, y_train, y_test, archive_path


def preprocess_nsl(train: pd.DataFrame, test: pd.DataFrame, fit_indices=None):
    features = NUMERIC + CATEGORICAL
    prep = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL),
        ],
        sparse_threshold=1.0,
    )
    fit_frame = train[features] if fit_indices is None else train.iloc[fit_indices][features]
    prep.fit(fit_frame)
    X_train = prep.transform(train[features]).astype(np.float32).tocsr()
    X_test = prep.transform(test[features]).astype(np.float32).tocsr()
    names = prep.get_feature_names_out().tolist()
    return X_train, X_test, prep, names


def stratified_iid_clients(y, clients: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    partitions = [[] for _ in range(clients)]
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        for client, chunk in enumerate(np.array_split(indices, clients)):
            partitions[client].extend(chunk.tolist())
    return [np.asarray(sorted(indices), dtype=int) for indices in partitions]


def dirichlet_clients(y, clients: int, alpha: float, seed: int = 42, min_size: int = 100):
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        partitions = [[] for _ in range(clients)]
        for label in np.unique(y):
            indices = np.flatnonzero(y == label)
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.repeat(alpha, clients))
            cut = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
            for client, chunk in enumerate(np.split(indices, cut)):
                partitions[client].extend(chunk.tolist())
        arrays = [np.asarray(indices, dtype=int) for indices in partitions]
        if min(map(len, arrays)) >= min_size:
            return arrays
    raise RuntimeError("Could not generate a Dirichlet partition meeting the minimum client size")


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def predict_probability(X, weights, intercept):
    return sigmoid(np.asarray(X @ weights).ravel() + intercept)


def local_update(X, y, weights, intercept, epochs, learning_rate, l2, rng, batch_size=2048):
    w = weights.copy()
    b = float(intercept)
    order = np.arange(len(y))
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            xb = X[batch]
            yb = y[batch].astype(np.float64)
            probability = predict_probability(xb, w, b)
            error = probability - yb
            grad_w = np.asarray(xb.T @ error).ravel() / len(batch) + l2 * w
            grad_b = float(error.mean())
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b
    return w, b


def fedavg_logistic(
    X,
    y,
    partitions,
    rounds=20,
    local_epochs=1,
    learning_rate=0.05,
    l2=1e-4,
    client_fraction=1.0,
    seed=42,
    validation=None,
):
    rng = np.random.default_rng(seed)
    weights = np.zeros(X.shape[1], dtype=np.float64)
    intercept = 0.0
    history = []
    total_communication = 0
    model_bytes = weights.nbytes + np.asarray(intercept, dtype=np.float64).nbytes
    clients_per_round = max(1, int(np.ceil(client_fraction * len(partitions))))
    for round_index in range(1, rounds + 1):
        selected = rng.choice(len(partitions), size=clients_per_round, replace=False)
        local_weights, local_intercepts, sample_counts = [], [], []
        for client in selected:
            indices = partitions[client]
            local_rng = np.random.default_rng(seed + 10000 * round_index + int(client))
            w, b = local_update(X[indices], y[indices], weights, intercept, local_epochs, learning_rate, l2, local_rng)
            local_weights.append(w)
            local_intercepts.append(b)
            sample_counts.append(len(indices))
        proportions = np.asarray(sample_counts, dtype=float) / np.sum(sample_counts)
        weights = np.sum(np.vstack(local_weights) * proportions[:, None], axis=0)
        intercept = float(np.sum(np.asarray(local_intercepts) * proportions))
        round_bytes = int(2 * clients_per_round * model_bytes)
        total_communication += round_bytes
        record = {"round": round_index, "selected_clients": clients_per_round, "round_bytes": round_bytes, "cumulative_bytes": total_communication}
        if validation is not None:
            Xv, yv = validation
            p = predict_probability(Xv, weights, intercept)
            pred = (p >= 0.5).astype(int)
            tn = np.sum((yv == 0) & (pred == 0))
            fp = np.sum((yv == 0) & (pred == 1))
            record.update({
                "validation_accuracy": float(np.mean(pred == yv)),
                "validation_f1": float(2 * np.sum((yv == 1) & (pred == 1)) / max(2 * np.sum((yv == 1) & (pred == 1)) + np.sum((yv == 0) & (pred == 1)) + np.sum((yv == 1) & (pred == 0)), 1)),
                "validation_fpr": float(fp / max(fp + tn, 1)),
            })
        history.append(record)
    return weights, intercept, pd.DataFrame(history), total_communication


def client_profile(partitions, y):
    rows = []
    for client, indices in enumerate(partitions):
        rows.append({"client": client + 1, "records": len(indices), "attack_fraction": float(np.mean(y[indices]))})
    return pd.DataFrame(rows)


def threshold_for_fpr(y_true, probabilities, target_fpr=0.03):
    thresholds = np.linspace(0.01, 0.99, 991)
    candidates = []
    for threshold in thresholds:
        pred = probabilities >= threshold
        tn = np.sum((y_true == 0) & (~pred))
        fp = np.sum((y_true == 0) & pred)
        tp = np.sum((y_true == 1) & pred)
        fn = np.sum((y_true == 1) & (~pred))
        fpr = fp / max(fp + tn, 1)
        recall = tp / max(tp + fn, 1)
        if fpr <= target_fpr:
            candidates.append((recall, -threshold, threshold, fpr))
    if not candidates:
        return 0.99
    return float(max(candidates)[2])
