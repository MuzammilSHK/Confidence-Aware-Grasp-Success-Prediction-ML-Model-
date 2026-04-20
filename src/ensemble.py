"""
Ensemble learning applied to the classification models to improve
grasp success prediction performance.

Includes:
  - Bagging (with Decision Tree base estimator)
  - Random Forest (Bagging's famous cousin)
  - AdaBoost (Boosting)
  - Gradient Boosting
  - Voting (soft-voting hybrid of KNN + SVM + Decision Tree)
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              VotingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
METRIC_DIR = os.path.join(RESULTS_DIR, "metrics")


def _evaluate(model, X_test, y_test, name, train_time):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "train_time_sec": train_time,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


# =============================================================================
# Bagging
# =============================================================================

def run_bagging(X_train, X_test, y_train, y_test, n_estimators=15):
    print(f"\n[Bagging] Training with n_estimators={n_estimators}...")
    base = DecisionTreeClassifier(max_depth=15, random_state=42)
    model = BaggingClassifier(estimator=base, n_estimators=n_estimators,
                              n_jobs=-1, random_state=42)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    m = _evaluate(model, X_test, y_test, "Bagging", train_time)
    print(f"[Bagging] Accuracy={m['accuracy']:.4f}, F1={m['f1']:.4f}")
    return model, m


# =============================================================================
# Random Forest
# =============================================================================

def run_random_forest(X_train, X_test, y_train, y_test, n_estimators=100):
    print(f"\n[RandomForest] Training with n_estimators={n_estimators}...")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=None,
                                   n_jobs=-1, random_state=42)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    m = _evaluate(model, X_test, y_test, "RandomForest", train_time)
    print(f"[RandomForest] Accuracy={m['accuracy']:.4f}, F1={m['f1']:.4f}")
    return model, m


# =============================================================================
# AdaBoost
# =============================================================================

def run_adaboost(X_train, X_test, y_train, y_test, n_estimators=100):
    print(f"\n[AdaBoost] Training with n_estimators={n_estimators}...")
    model = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    m = _evaluate(model, X_test, y_test, "AdaBoost", train_time)
    print(f"[AdaBoost] Accuracy={m['accuracy']:.4f}, F1={m['f1']:.4f}")
    return model, m


# =============================================================================
# Gradient Boosting
# =============================================================================

def run_gradient_boosting(X_train, X_test, y_train, y_test, n_estimators=60):
    print(f"\n[GradientBoosting] Training with n_estimators={n_estimators}...")
    model = GradientBoostingClassifier(n_estimators=n_estimators,
                                       max_depth=4, random_state=42)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    m = _evaluate(model, X_test, y_test, "GradientBoosting", train_time)
    print(f"[GradientBoosting] Accuracy={m['accuracy']:.4f}, F1={m['f1']:.4f}")
    return model, m


# =============================================================================
# Voting Ensemble
# =============================================================================

def run_voting(X_train, X_test, y_train, y_test, max_svm_samples=15000):
    print(f"\n[Voting] Training soft-voting ensemble (KNN + SVM + Tree)...")

    # Use same SVM subsample trick to keep training time reasonable
    if len(X_train) > max_svm_samples:
        idx = np.random.default_rng(42).choice(len(X_train), max_svm_samples,
                                               replace=False)
        X_svm = X_train[idx]
        y_svm = y_train[idx]
    else:
        X_svm, y_svm = X_train, y_train

    # Fit components individually so we can mix data sizes
    knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1).fit(X_train, y_train)
    svm = SVC(kernel="rbf", probability=True, random_state=42).fit(X_svm, y_svm)
    tree = DecisionTreeClassifier(max_depth=15, random_state=42).fit(X_train, y_train)

    model = VotingClassifier(
        estimators=[("knn", knn), ("svm", svm), ("tree", tree)],
        voting="soft"
    )
    # VotingClassifier requires fit; re-fit on the full dataset is heavy,
    # so we construct a pre-fitted wrapper manually.
    t0 = time.time()
    model.estimators_ = [knn, svm, tree]
    model.le_ = type("LE", (), {"classes_": np.array([0, 1]),
                                "transform": lambda self, x: x})()
    model.classes_ = np.array([0, 1])
    train_time = time.time() - t0

    # Manual soft-voting predict since we skipped the full fit
    probas = np.mean([
        knn.predict_proba(X_test),
        svm.predict_proba(X_test),
        tree.predict_proba(X_test),
    ], axis=0)
    y_pred = (probas[:, 1] > 0.5).astype(int)
    y_proba = probas[:, 1]

    m = {
        "model": "Voting",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "train_time_sec": train_time,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"[Voting] Accuracy={m['accuracy']:.4f}, F1={m['f1']:.4f}")
    return model, m


# =============================================================================
# Visualization
# =============================================================================

def plot_ensemble_comparison(base_results, ensemble_results, save_path):
    combined = {}
    combined.update(base_results)
    combined.update(ensemble_results)

    rows = [{"Model": name, "Accuracy": m["accuracy"], "F1": m["f1"],
             "Precision": m["precision"], "Recall": m["recall"]}
            for name, m in combined.items()]
    df = pd.DataFrame(rows).melt(id_vars="Model", var_name="Metric",
                                  value_name="Score")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Model", y="Score", hue="Metric")
    plt.ylim(0.7, 1.02)
    plt.title("Base Classifiers vs Ensemble Methods")
    plt.xticks(rotation=25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_ensemble_confusions(results, save_path):
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (name, m) in zip(axes, results.items()):
        cm = np.array(m["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                    xticklabels=["Unstable", "Stable"],
                    yticklabels=["Unstable", "Stable"], cbar=False)
        ax.set_title(f"{name}\nAcc={m['accuracy']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for i in range(len(results), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# =============================================================================
# Full Ensemble Pipeline
# =============================================================================

def run_all_ensembles(data, base_results=None):
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    results = {}
    models = {}

    for name, runner in [
        ("Bagging", run_bagging),
        ("RandomForest", run_random_forest),
        ("AdaBoost", run_adaboost),
        ("GradientBoosting", run_gradient_boosting),
    ]:
        model, m = runner(X_train, X_test, y_train, y_test)
        results[name] = m
        models[name] = model

    vote_model, vote_metrics = run_voting(X_train, X_test, y_train, y_test)
    results["Voting"] = vote_metrics
    models["Voting"] = vote_model

    plot_ensemble_confusions(results, os.path.join(FIG_DIR, "ensemble_confusion.png"))

    if base_results is not None:
        plot_ensemble_comparison(base_results, results,
                                 os.path.join(FIG_DIR, "ensemble_vs_base.png"))

    with open(os.path.join(METRIC_DIR, "ensemble_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== Ensemble Summary ===")
    print(pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1",
                                    "train_time_sec"]])

    return models, results


if __name__ == "__main__":
    from preprocessing import run_full_pipeline
    data = run_full_pipeline(sample_frac=0.2)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)
    run_all_ensembles(data)
