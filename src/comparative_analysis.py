"""
Cross-module comparative analysis.

Pulls together the classification, regression, clustering, and
Q-Learning results to produce the summary plots and tables that
anchor the final report. Generates:

  - A master comparison table (CSV)
  - A computational complexity bar chart
  - A unified performance overview
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
METRIC_DIR = os.path.join(RESULTS_DIR, "metrics")


# =============================================================================
# Collection
# =============================================================================

def load_all_metrics():
    result = {}
    for name in ["classification", "ensemble", "regression",
                "clustering", "qlearning"]:
        path = os.path.join(METRIC_DIR, f"{name}_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                result[name] = json.load(f)
        else:
            result[name] = None
    return result


# =============================================================================
# Master Table
# =============================================================================

def build_master_table(metrics):
    rows = []

    if metrics["classification"]:
        for name, m in metrics["classification"].items():
            rows.append({
                "Task": "Classification",
                "Model": name,
                "Primary Metric": "F1",
                "Score": m["f1"],
                "Accuracy": m["accuracy"],
                "Train Time (s)": m["train_time_sec"],
            })

    if metrics["ensemble"]:
        for name, m in metrics["ensemble"].items():
            rows.append({
                "Task": "Ensemble",
                "Model": name,
                "Primary Metric": "F1",
                "Score": m["f1"],
                "Accuracy": m["accuracy"],
                "Train Time (s)": m["train_time_sec"],
            })

    if metrics["regression"]:
        for name, m in metrics["regression"].items():
            rows.append({
                "Task": "Regression",
                "Model": name,
                "Primary Metric": "R²",
                "Score": m["R2"],
                "Accuracy": np.nan,
                "Train Time (s)": m["train_time_sec"],
            })

    if metrics["clustering"]:
        rows.append({
            "Task": "Clustering",
            "Model": f"K-Means (k={metrics['clustering']['best_k']})",
            "Primary Metric": "Silhouette",
            "Score": metrics["clustering"]["silhouette_best_k"],
            "Accuracy": metrics["clustering"].get("k2_label_agreement", np.nan),
            "Train Time (s)": metrics["clustering"]["train_time_sec"],
        })

    if metrics["qlearning"]:
        conv = metrics["qlearning"]["convergence"] or {}
        rows.append({
            "Task": "Reinforcement Learning",
            "Model": "Q-Learning",
            "Primary Metric": "Final Success Rate",
            "Score": conv.get("final_success_rate", np.nan),
            "Accuracy": np.nan,
            "Train Time (s)": np.nan,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Visualization
# =============================================================================

def plot_classification_vs_ensemble(metrics, save_path):
    if not (metrics["classification"] and metrics["ensemble"]):
        return

    combined = {}
    for name, m in metrics["classification"].items():
        combined[name] = {"Accuracy": m["accuracy"], "F1": m["f1"],
                          "Type": "Base"}
    for name, m in metrics["ensemble"].items():
        combined[name] = {"Accuracy": m["accuracy"], "F1": m["f1"],
                          "Type": "Ensemble"}

    df = pd.DataFrame(combined).T.reset_index().rename(columns={"index": "Model"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["Accuracy", "F1"]):
        sns.barplot(data=df, x="Model", y=metric, hue="Type",
                    ax=ax, palette={"Base": "steelblue", "Ensemble": "darkorange"})
        ax.set_ylim(0.85, 1.02)
        ax.set_title(f"{metric} — Base vs Ensemble")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_computational_cost(metrics, save_path):
    """Grouped bar: training time for all classification/ensemble models."""
    rows = []
    for group, key in [("Base", "classification"), ("Ensemble", "ensemble")]:
        if metrics[key]:
            for name, m in metrics[key].items():
                rows.append({"Model": name, "Group": group,
                             "Train Time (s)": m["train_time_sec"]})

    if not rows:
        return

    df = pd.DataFrame(rows)
    plt.figure(figsize=(11, 5))
    sns.barplot(data=df, x="Model", y="Train Time (s)", hue="Group",
                palette={"Base": "steelblue", "Ensemble": "darkorange"})
    plt.xticks(rotation=25)
    plt.title("Training Time per Model (Computational Complexity)")
    plt.yscale("log")
    plt.ylabel("Training Time (seconds, log scale)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_overall_summary(metrics, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))

    rows = []
    if metrics["classification"]:
        for name, m in metrics["classification"].items():
            rows.append({"Model": name, "Task": "Classification",
                         "Score": m["f1"]})
    if metrics["ensemble"]:
        for name, m in metrics["ensemble"].items():
            rows.append({"Model": name, "Task": "Ensemble",
                         "Score": m["f1"]})
    if metrics["regression"]:
        for name, m in metrics["regression"].items():
            rows.append({"Model": name, "Task": "Regression",
                         "Score": max(0, m["R2"])})
    if metrics["clustering"]:
        rows.append({"Model": f"K-Means (k={metrics['clustering']['best_k']})",
                     "Task": "Clustering",
                     "Score": metrics["clustering"]["k2_label_agreement"]})
    if metrics["qlearning"]:
        conv = metrics["qlearning"]["convergence"] or {}
        rows.append({"Model": "Q-Learning", "Task": "Reinforcement Learning",
                     "Score": conv.get("final_success_rate", 0.0)})

    df = pd.DataFrame(rows)
    palette = {
        "Classification": "steelblue",
        "Ensemble": "darkorange",
        "Regression": "seagreen",
        "Clustering": "mediumorchid",
        "Reinforcement Learning": "crimson",
    }
    sns.barplot(data=df, x="Model", y="Score", hue="Task", palette=palette, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Overall Performance Across All Tasks")
    ax.set_ylabel("Primary Score (F1 / R² / Silhouette / Success Rate)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# =============================================================================
# Markdown Summary
# =============================================================================

def write_summary_markdown(metrics, master_df, save_path):
    lines = []
    lines.append("# Comparative Analysis Summary\n")
    lines.append("*Auto-generated by `comparative_analysis.py`.*\n\n")

    lines.append("## Master Results Table\n")
    lines.append(master_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n\n")

    if metrics["classification"]:
        lines.append("## Best Classifier\n")
        best = max(metrics["classification"].items(), key=lambda x: x[1]["f1"])
        lines.append(f"- **{best[0]}** — "
                     f"F1={best[1]['f1']:.4f}, "
                     f"Accuracy={best[1]['accuracy']:.4f}\n\n")

    if metrics["ensemble"]:
        lines.append("## Best Ensemble\n")
        best = max(metrics["ensemble"].items(), key=lambda x: x[1]["f1"])
        lines.append(f"- **{best[0]}** — "
                     f"F1={best[1]['f1']:.4f}, "
                     f"Accuracy={best[1]['accuracy']:.4f}\n\n")

    if metrics["regression"]:
        lines.append("## Best Regressor\n")
        best = max(metrics["regression"].items(), key=lambda x: x[1]["R2"])
        lines.append(f"- **{best[0]}** — "
                     f"R²={best[1]['R2']:.4f}, "
                     f"RMSE={best[1]['RMSE']:.3f}\n\n")

    if metrics["clustering"]:
        c = metrics["clustering"]
        lines.append("## Clustering\n")
        lines.append(f"- **K-Means** — best k = {c['best_k']}, "
                     f"silhouette = {c['silhouette_best_k']:.4f}, "
                     f"k=2 label agreement = {c['k2_label_agreement']:.2%}\n\n")

    if metrics["qlearning"]:
        q = metrics["qlearning"]
        conv = q["convergence"] or {}
        lines.append("## Q-Learning\n")
        lines.append(f"- Grid size: {q['grid_size']}x{q['grid_size']}, "
                     f"obstacles: {q['n_obstacles']}\n")
        lines.append(f"- Final success rate: "
                     f"{conv.get('final_success_rate', 0):.2%}\n")
        lines.append(f"- Convergence episode: "
                     f"{conv.get('convergence_episode', 'N/A')}\n")
        lines.append(f"- Greedy rollout: reached goal = "
                     f"{q['reached_goal_under_greedy_policy']}, "
                     f"path length = {q['greedy_path_length']}\n")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))


# =============================================================================
# Main Entrypoint
# =============================================================================

def run_comparative_analysis():
    metrics = load_all_metrics()
    master_df = build_master_table(metrics)

    master_df.to_csv(os.path.join(METRIC_DIR, "master_results.csv"), index=False)

    plot_classification_vs_ensemble(
        metrics, os.path.join(FIG_DIR, "comparison_base_vs_ensemble.png"))
    plot_computational_cost(
        metrics, os.path.join(FIG_DIR, "comparison_train_time.png"))
    plot_overall_summary(
        metrics, os.path.join(FIG_DIR, "comparison_overall.png"))

    write_summary_markdown(
        metrics, master_df,
        os.path.join(RESULTS_DIR, "comparative_summary.md"))

    print("\n=== Master Results Table ===")
    print(master_df.to_string(index=False))

    return master_df, metrics


if __name__ == "__main__":
    run_comparative_analysis()
