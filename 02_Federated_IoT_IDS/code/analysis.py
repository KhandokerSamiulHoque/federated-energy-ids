"""Paper 2: true parameter-averaged federated logistic IDS on NSL-KDD."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import binary_metrics, ensure_dirs, expected_calibration_error, model_scores, save_json, savefig, seed_everything, set_plot_style, sha256
from federated import client_profile, fedavg_logistic, load_nsl, predict_probability, preprocess_nsl, stratified_iid_clients


PAPER = ROOT / "02_Federated_IoT_IDS"
RESULTS, FIGURES = ensure_dirs(PAPER)


def main() -> None:
    seed_everything()
    set_plot_style()
    train, test, y_all, y_test, archive = load_nsl(ROOT)
    fit_i, validation_i = train_test_split(np.arange(len(y_all)), test_size=0.10, stratify=y_all, random_state=42)
    X_all, X_test, prep, feature_names = preprocess_nsl(train, test, fit_indices=fit_i)
    X_fit, y_fit = X_all[fit_i], y_all[fit_i]
    X_validation, y_validation = X_all[validation_i], y_all[validation_i]
    partitions = stratified_iid_clients(y_fit, clients=10, seed=42)
    profile = client_profile(partitions, y_fit)
    profile.to_csv(RESULTS / "client_partition_profile.csv", index=False)

    weights, intercept, history, communication = fedavg_logistic(
        X_fit, y_fit, partitions, rounds=25, local_epochs=1, learning_rate=0.05, l2=1e-4,
        client_fraction=1.0, seed=42, validation=(X_validation, y_validation),
    )
    history.to_csv(RESULTS / "fedavg_convergence.csv", index=False)
    fed_score = predict_probability(X_test, weights, intercept)
    fed_pred = (fed_score >= 0.5).astype(int)
    rows = [{"model": "FedAvg logistic (10 clients)", **binary_metrics(y_test, fed_pred, fed_score), "ece": expected_calibration_error(y_test, fed_score)}]

    baselines = {
        "Central logistic regression": LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1, random_state=42),
        "Central SGD logistic": SGDClassifier(loss="log_loss", alpha=1e-4, class_weight="balanced", max_iter=2000, tol=1e-4, random_state=42),
        "Central random forest": RandomForestClassifier(n_estimators=250, max_depth=None, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
    }
    fitted = {}
    for name, model in baselines.items():
        model.fit(X_fit, y_fit)
        score = model_scores(model, X_test)
        pred = (score >= 0.5).astype(int)
        rows.append({"model": name, **binary_metrics(y_test, pred, score), "ece": expected_calibration_error(y_test, score)})
        fitted[name] = model
    metrics = pd.DataFrame(rows).sort_values("f1", ascending=False)
    metrics.to_csv(RESULTS / "official_test_metrics.csv", index=False)

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(history["round"], history["validation_f1"], marker="o", markersize=3, label="Validation F1")
    plt.plot(history["round"], history["validation_accuracy"], marker="s", markersize=3, label="Validation accuracy")
    plt.xlabel("Federated communication round")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("True coefficient-averaged FedAvg convergence")
    plt.legend()
    savefig(FIGURES / "figure_1_fedavg_convergence.png")

    plt.figure(figsize=(8.0, 4.8))
    plot = metrics.sort_values("f1")
    y_pos = np.arange(len(plot))
    plt.barh(y_pos - 0.16, plot["f1"], height=0.32, label="F1", color="#457b9d")
    plt.barh(y_pos + 0.16, plot["roc_auc"], height=0.32, label="ROC-AUC", color="#2a9d8f")
    plt.yticks(y_pos, plot["model"])
    plt.xlim(0, 1)
    plt.xlabel("Official NSL-KDD test score")
    plt.title("Centralized and federated intrusion-detection comparison")
    plt.legend(loc="lower right")
    savefig(FIGURES / "figure_2_model_comparison.png")

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    RocCurveDisplay.from_predictions(y_test, fed_score, name="FedAvg logistic", ax=ax)
    for name in ["Central logistic regression", "Central random forest"]:
        RocCurveDisplay.from_predictions(y_test, model_scores(fitted[name], X_test), name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
    ax.set_title("Official NSL-KDD test ROC curves")
    savefig(FIGURES / "figure_3_roc_curves.png")

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(y_test, fed_pred, display_labels=["Normal", "Attack"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("FedAvg logistic on the official test set")
    savefig(FIGURES / "figure_4_fedavg_confusion_matrix.png")

    plt.figure(figsize=(7.0, 4.4))
    plt.bar(profile["client"], profile["attack_fraction"], color="#f4a261")
    plt.axhline(np.mean(y_fit), color="#264653", linestyle="--", label="Overall attack fraction")
    plt.xlabel("Simulated client")
    plt.ylabel("Attack fraction")
    plt.ylim(0, 1)
    plt.title("Stratified IID client composition")
    plt.legend()
    savefig(FIGURES / "figure_5_client_composition.png")

    summary = {
        "train_records_total": len(train), "federated_fit_records": len(fit_i), "validation_records": len(validation_i), "official_test_records": len(test),
        "encoded_features": len(feature_names), "clients": 10, "rounds": 25, "local_epochs": 1,
        "communication_bytes_measured_from_parameter_dimensions": communication,
        "archive_sha256": sha256(archive), "fedavg_test_metrics": metrics.set_index("model").loc["FedAvg logistic (10 clients)"].to_dict(),
        "limitations": ["Clients are simulated partitions of NSL-KDD rather than independent organizations.", "NSL-KDD is a dated benchmark and does not establish performance on contemporary operational IoT traffic.", "No privacy guarantee follows from federated learning alone; secure aggregation and formal differential privacy were outside this experiment."],
    }
    save_json(RESULTS / "analysis_summary.json", summary)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
