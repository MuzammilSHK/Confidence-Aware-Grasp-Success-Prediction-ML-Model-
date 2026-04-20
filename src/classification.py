"""
Classification models for grasp success prediction:
K-Nearest Neighbors, Support Vector Machine, Decision Tree.

Each model is trained on the scaled per-experiment features and evaluated
with accuracy, precision, recall, F1, and a confusion matrix. Training
time is logged to support the computational-complexity discussion.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
METRIC_DIR = os.path.join(RESULTS_DIR, "metrics")


# =============================================================================
# Training Helpers
# =============================================================================

def train_and_time(model, X_train, y_train):
    start = time.time()
    model.fit(X_train, y_train)
    return model, time.time() - start


def evaluate(model, X_test, y_test, model_name):
    start = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start

    # Probability scores if supported — needed for ROC and calibration
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df_scores = model.decision_function(X_test)
        y_proba = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min())
    else:
        y_proba = None

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "prediction_time_sec": pred_time,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics, y_pred, y_proba


# =============================================================================
# K-Nearest Neighbors
# =============================================================================

def run_knn(X_train, X_test, y_train, y_test, k=7):
    print(f"\n[KNN] Training with k={k}...")
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model, train_time = train_and_time(model, X_train, y_train)
    metrics, y_pred, y_proba = evaluate(model, X_test, y_test, "KNN")
    metrics["train_time_sec"] = train_time
    metrics["hyperparameters"] = {"k": k}
    print(f"[KNN] Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
          f"train_time={train_time:.2f}s")
    return model, metrics, y_proba


# =============================================================================
# Support Vector Machine
# =============================================================================

def run_svm(X_train, X_test, y_train, y_test, kernel="rbf", max_samples=15000):
    # SVM with RBF kernel scales poorly with sample count. Subsample training
    # data if it's too large; the test set stays the full size.
    if len(X_train) > max_samples:
        idx = np.random.default_rng(42).choice(len(X_train), max_samples, replace=False)
        X_tr = X_train[idx]
        y_tr = y_train[idx]
        print(f"[SVM] Subsampled training set to {max_samples} for RBF kernel speed")
    else:
        X_tr, y_tr = X_train, y_train

    print(f"[SVM] Training with kernel={kernel}...")
    model = SVC(kernel=kernel, C=1.0, probability=True, random_state=42)
    model, train_time = train_and_time(model, X_tr, y_tr)
    metrics, y_pred, y_proba = evaluate(model, X_test, y_test, "SVM")
    metrics["train_time_sec"] = train_time
    metrics["hyperparameters"] = {"kernel": kernel, "C": 1.0,
                                  "train_samples_used": len(X_tr)}
    print(f"[SVM] Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
          f"train_time={train_time:.2f}s")
    return model, metrics, y_proba


# =============================================================================
# Decision Tree
# =============================================================================

def run_decision_tree(X_train, X_test, y_train, y_test, max_depth=15):
    print(f"\n[DecisionTree] Training with max_depth={max_depth}...")
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model, train_time = train_and_time(model, X_train, y_train)
    metrics, y_pred, y_proba = evaluate(model, X_test, y_test, "DecisionTree")
    metrics["train_time_sec"] = train_time
    metrics["hyperparameters"] = {"max_depth": max_depth}
    print(f"[DecisionTree] Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
          f"train_time={train_time:.2f}s")
    return model, metrics, y_proba


# =============================================================================
# Visualization
# =============================================================================

def plot_confusion_matrices(results, save_path):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, m) in zip(axes, results.items()):
        cm = np.array(m["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Unstable", "Stable"],
                    yticklabels=["Unstable", "Stable"], cbar=False)
        ax.set_title(f"{name}\nAcc={m['accuracy']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_metric_comparison(results, save_path):
    df = pd.DataFrame([
        {"Model": name,
         "Accuracy": m["accuracy"],
         "Precision": m["precision"],
         "Recall": m["recall"],
         "F1": m["f1"]}
        for name, m in results.items()
    ])
    df_melt = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Model")
    plt.ylim(0, 1.05)
    plt.title("Classification Performance Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_roc_curves(probas, y_test, save_path):
    plt.figure(figsize=(7, 6))
    for name, proba in probas.items():
        if proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Classification Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# =============================================================================
# Full Classification Pipeline
# =============================================================================

def run_all_classifiers(data):
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    results = {}
    probas = {}
    models = {}

    knn_model, knn_metrics, knn_proba = run_knn(X_train, X_test, y_train, y_test)
    results["KNN"] = knn_metrics
    probas["KNN"] = knn_proba
    models["KNN"] = knn_model

    svm_model, svm_metrics, svm_proba = run_svm(X_train, X_test, y_train, y_test)
    results["SVM"] = svm_metrics
    probas["SVM"] = svm_proba
    models["SVM"] = svm_model

    dt_model, dt_metrics, dt_proba = run_decision_tree(X_train, X_test, y_train, y_test)
    results["DecisionTree"] = dt_metrics
    probas["DecisionTree"] = dt_proba
    models["DecisionTree"] = dt_model

    # Save plots
    plot_confusion_matrices(results, os.path.join(FIG_DIR, "classification_confusion.png"))
    plot_metric_comparison(results, os.path.join(FIG_DIR, "classification_metrics.png"))
    plot_roc_curves(probas, y_test, os.path.join(FIG_DIR, "classification_roc.png"))

    # Save metrics
    with open(os.path.join(METRIC_DIR, "classification_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== Classification Summary ===")
    print(pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1",
                                    "train_time_sec"]])

    return models, results, probas


if __name__ == "__main__":
    from preprocessing import run_full_pipeline
    data = run_full_pipeline()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)
    run_all_classifiers(data)
