"""
End-to-end pipeline orchestrator.

Runs preprocessing, all classification models, ensembles, regression,
clustering, Q-Learning, and the comparative analysis — in that order —
and saves every figure, metric, and summary table under ../results/.

Usage:
    python main.py                    # full dataset
    python main.py --sample 0.3       # use 30% of experiments (faster)
"""

import os
import argparse
import time

from preprocessing import run_full_pipeline
from classification import run_all_classifiers
from ensemble import run_all_ensembles
from regression import run_all_regressors
from clustering import run_clustering
from qlearning import run_qlearning
from comparative_analysis import run_comparative_analysis

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
METRIC_DIR = os.path.join(RESULTS_DIR, "metrics")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")


def ensure_dirs():
    for d in [RESULTS_DIR, FIG_DIR, METRIC_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)


def banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main(sample_frac=None):
    ensure_dirs()
    t_start = time.time()

    banner("STEP 1  —  PREPROCESSING")
    data = run_full_pipeline(sample_frac=sample_frac)

    banner("STEP 2  —  CLASSIFICATION  (KNN, SVM, Decision Tree)")
    _, cls_results, _ = run_all_classifiers(data)

    banner("STEP 3  —  ENSEMBLE LEARNING")
    _, ens_results = run_all_ensembles(data, base_results=cls_results)

    banner("STEP 4  —  REGRESSION  (Linear, Polynomial)")
    run_all_regressors(data)

    banner("STEP 5  —  CLUSTERING  (K-Means)")
    run_clustering(data)

    banner("STEP 6  —  REINFORCEMENT LEARNING  (Q-Learning)")
    run_qlearning()

    banner("STEP 7  —  COMPARATIVE ANALYSIS")
    master_df, _ = run_comparative_analysis()

    total = time.time() - t_start
    banner(f"DONE — total runtime: {total:.1f}s")
    print(f"Results written to: {RESULTS_DIR}")
    print(f"  - Figures:      {FIG_DIR}")
    print(f"  - Metrics:      {METRIC_DIR}")
    print(f"  - Summary:      {os.path.join(RESULTS_DIR, 'comparative_summary.md')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=None,
                        help="Fraction of experiments to use (0 < f <= 1). "
                             "Omit for full dataset.")
    args = parser.parse_args()
    main(sample_frac=args.sample)
