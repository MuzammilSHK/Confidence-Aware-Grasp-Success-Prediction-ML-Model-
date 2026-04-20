"""
K-Means clustering on engineered grasp features.

The goal: group grasp attempts into natural patterns based on joint
dynamics, then see whether those clusters correlate with grasp stability.
This is an unsupervised sanity check — if the features carry meaningful
information, the clusters should line up with the stable/unstable labels.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
METRIC_DIR = os.path.join(RESULTS_DIR, "metrics")


# =============================================================================
# K-Means Search Over k
# =============================================================================

def search_k(X, k_values=(2, 3, 4, 5, 6, 7, 8), sample_size=5000,
             random_state=42):
    """
    Try several values of k, compute silhouette and Davies-Bouldin scores,
    and return the best configuration plus all scores for plotting.
    """
    # Silhouette is expensive; subsample for speed without hurting the
    # qualitative comparison across k values.
    rng = np.random.default_rng(random_state)
    if len(X) > sample_size:
        idx = rng.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    scores = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X_sample)
        sil = silhouette_score(X_sample, labels)
        db = davies_bouldin_score(X_sample, labels)
        inertia = model.inertia_
        print(f"  k={k}: silhouette={sil:.4f}, davies-bouldin={db:.4f}, "
              f"inertia={inertia:.1f}")
        scores.append({"k": k, "silhouette": sil,
                       "davies_bouldin": db, "inertia": inertia})

    return pd.DataFrame(scores)


# =============================================================================
# Fit Best Model
# =============================================================================

def fit_kmeans(X, k, random_state=42):
    t0 = time.time()
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    train_time = time.time() - t0
    return model, labels, train_time


# =============================================================================
# Visualization
# =============================================================================

def plot_k_search(scores_df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(scores_df["k"], scores_df["inertia"], "o-", color="steelblue")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia (SSE)")
    axes[0].set_title("Elbow Method")

    axes[1].plot(scores_df["k"], scores_df["silhouette"], "o-", color="seagreen")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette (higher = better)")

    axes[2].plot(scores_df["k"], scores_df["davies_bouldin"], "o-", color="indianred")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("Davies-Bouldin Index")
    axes[2].set_title("Davies-Bouldin (lower = better)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_clusters_pca(X, labels, y_binary, save_path, sample_size=4000,
                     random_state=42):
    """
    Project features to 2D via PCA and color-code by (a) cluster and
    (b) true stability label to see how well the clustering lines up.
    """
    rng = np.random.default_rng(random_state)
    if len(X) > sample_size:
        idx = rng.choice(len(X), sample_size, replace=False)
    else:
        idx = np.arange(len(X))

    pca = PCA(n_components=2, random_state=random_state)
    X2d = pca.fit_transform(X[idx])
    labels_sample = labels[idx]
    y_sample = y_binary[idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    scat1 = axes[0].scatter(X2d[:, 0], X2d[:, 1], c=labels_sample,
                             cmap="tab10", s=10, alpha=0.6)
    axes[0].set_title(f"K-Means Clusters (k={len(np.unique(labels_sample))})")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.colorbar(scat1, ax=axes[0], label="Cluster")

    scat2 = axes[1].scatter(X2d[:, 0], X2d[:, 1], c=y_sample,
                             cmap="coolwarm", s=10, alpha=0.6)
    axes[1].set_title("True Stability Labels")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    cb = plt.colorbar(scat2, ax=axes[1], label="Stability")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Unstable", "Stable"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_cluster_vs_label_heatmap(labels, y_binary, save_path):
    """Cross-tab heatmap: how do clusters align with true labels?"""
    crosstab = pd.crosstab(labels, y_binary,
                           rownames=["Cluster"], colnames=["Stability"])
    crosstab.columns = ["Unstable", "Stable"]

    plt.figure(figsize=(6, max(3, len(crosstab) * 0.5)))
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlOrRd")
    plt.title("Cluster Composition by True Stability")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# =============================================================================
# Full Clustering Pipeline
# =============================================================================

def run_clustering(data):
    # Use the full feature matrix (train only) for unsupervised analysis
    X = data["X_train"]
    y_binary = data["y_train"]

    print("\n[Clustering] Searching over k values...")
    scores_df = search_k(X)
    plot_k_search(scores_df, os.path.join(FIG_DIR, "cluster_k_search.png"))

    # Pick k with highest silhouette
    best_row = scores_df.sort_values("silhouette", ascending=False).iloc[0]
    best_k = int(best_row["k"])
    print(f"[Clustering] Best k by silhouette: {best_k}")

    model, labels, train_time = fit_kmeans(X, best_k)
    sil = silhouette_score(X[:5000], labels[:5000])
    db = davies_bouldin_score(X[:5000], labels[:5000])

    plot_clusters_pca(X, labels, y_binary,
                      os.path.join(FIG_DIR, "cluster_pca_projection.png"))
    plot_cluster_vs_label_heatmap(labels, y_binary,
                                  os.path.join(FIG_DIR, "cluster_vs_label.png"))

    # Also try k=2 explicitly — it's the natural choice given we have a
    # binary stable/unstable label
    print("\n[Clustering] Forcing k=2 for comparison with stability label...")
    model_k2, labels_k2, _ = fit_kmeans(X, 2)
    sil_k2 = silhouette_score(X[:5000], labels_k2[:5000])
    db_k2 = davies_bouldin_score(X[:5000], labels_k2[:5000])

    # Alignment between k=2 clusters and binary label
    # (either cluster 0→unstable,1→stable or the reverse)
    agreement = max(
        (labels_k2 == y_binary).mean(),
        (labels_k2 != y_binary).mean()
    )
    print(f"[Clustering] k=2 agreement with stability label: {agreement:.4f}")

    results = {
        "best_k": best_k,
        "silhouette_best_k": float(sil),
        "davies_bouldin_best_k": float(db),
        "silhouette_k2": float(sil_k2),
        "davies_bouldin_k2": float(db_k2),
        "k2_label_agreement": float(agreement),
        "train_time_sec": train_time,
        "k_search_scores": scores_df.to_dict(orient="records"),
    }

    with open(os.path.join(METRIC_DIR, "clustering_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== Clustering Summary ===")
    print(f"Best k: {best_k} (Silhouette={sil:.4f}, DB={db:.4f})")
    print(f"k=2 agreement with stability: {agreement:.4f}")

    return model, labels, results


if __name__ == "__main__":
    from preprocessing import run_full_pipeline
    data = run_full_pipeline(sample_frac=0.2)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)
    run_clustering(data)
