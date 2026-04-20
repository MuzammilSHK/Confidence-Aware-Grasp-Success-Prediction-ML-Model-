"""
Data loading, cleaning, feature engineering, and train/test splitting
for the Shadow Robot grasping dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                        "shadow_robot_dataset.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

JOINTS = [
    "H1_F1J1", "H1_F1J2", "H1_F1J3",
    "H1_F2J1", "H1_F2J2", "H1_F2J3",
    "H1_F3J1", "H1_F3J2", "H1_F3J3",
]
READINGS = ["pos", "vel", "eff"]


# =============================================================================
# Loading
# =============================================================================

def load_raw():
    # Pre-declare dtypes to keep memory use down on the ~1M-row CSV
    df = pd.read_csv(DATA_PATH, skipinitialspace=True, dtype={
        "experiment_number": str,
    })
    df.columns = df.columns.str.strip()
    # Downcast floats from 64→32 where possible
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def describe_dataset(df: pd.DataFrame) -> dict:
    """Collect descriptive stats for the report."""
    summary = {
        "total_rows": int(len(df)),
        "total_experiments": int(df["experiment_number"].nunique()),
        "total_columns": int(df.shape[1]),
        "robustness_min": float(df["robustness"].min()),
        "robustness_max": float(df["robustness"].max()),
        "robustness_mean": float(df["robustness"].mean()),
        "robustness_median": float(df["robustness"].median()),
        "missing_values": int(df.isnull().sum().sum()),
        "measurements_per_experiment_mean": float(
            df.groupby("experiment_number").size().mean()
        ),
    }
    return summary


# =============================================================================
# Cleaning
# =============================================================================

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop any fully duplicated rows (measurement noise artifacts)
    df = df.drop_duplicates()

    # Handle any missing values (the dataset is clean but this guards
    # against corruption if the CSV is ever re-exported)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Drop rows where robustness is non-finite (just in case)
    df = df[np.isfinite(df["robustness"])]

    return df


# =============================================================================
# Feature Engineering
# =============================================================================

def aggregate_per_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each grasp attempt has multiple time-step measurements.
    Collapse each experiment into a single row with statistical features
    summarizing the joint trajectories. This keeps the object-agnostic
    spirit of the original paper while giving classical models a rich
    feature vector per grasp.
    """
    sensor_cols = [f"{j}_{r}" for j in JOINTS for r in READINGS]

    grouped = df.groupby("experiment_number")

    agg_mean = grouped[sensor_cols].mean().add_suffix("_mean")
    agg_std = grouped[sensor_cols].std().fillna(0).add_suffix("_std")
    agg_last = grouped[sensor_cols].last().add_suffix("_last")
    robustness = grouped["robustness"].first()

    features = pd.concat([agg_mean, agg_std, agg_last], axis=1)
    features["robustness"] = robustness
    features = features.reset_index(drop=True)

    return features


def engineer_derived_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Add physically motivated derived features: overall torque magnitude,
    velocity magnitude, and per-finger aggregates. These often help
    tree-based models pick up patterns without needing deep trees.
    """
    df = features.copy()

    # Overall magnitude across all joints (mean of absolute values)
    eff_cols = [c for c in df.columns if "_eff_mean" in c]
    vel_cols = [c for c in df.columns if "_vel_mean" in c]
    pos_cols = [c for c in df.columns if "_pos_mean" in c]

    df["total_torque_magnitude"] = df[eff_cols].abs().mean(axis=1)
    df["total_velocity_magnitude"] = df[vel_cols].abs().mean(axis=1)
    df["total_position_magnitude"] = df[pos_cols].abs().mean(axis=1)

    # Per-finger aggregates (finger 1, 2, 3)
    for finger in ["F1", "F2", "F3"]:
        finger_eff = [c for c in eff_cols if finger in c]
        finger_vel = [c for c in vel_cols if finger in c]
        df[f"{finger}_torque"] = df[finger_eff].abs().mean(axis=1)
        df[f"{finger}_velocity"] = df[finger_vel].abs().mean(axis=1)

    # Inter-finger torque spread (unstable grasps often have uneven loading)
    df["finger_torque_spread"] = df[["F1_torque", "F2_torque", "F3_torque"]].std(axis=1)

    return df


# =============================================================================
# Label Creation
# =============================================================================

def make_binary_label(features: pd.DataFrame) -> pd.Series:
    """
    Convert continuous robustness into stable/unstable using the median split.
    This matches Ugo Cupcic's original setup and gives a balanced 50/50 dataset.
    """
    threshold = features["robustness"].median()
    return (features["robustness"] > threshold).astype(int)


# =============================================================================
# Scaling and Splitting
# =============================================================================

def split_and_scale(features: pd.DataFrame, target: pd.Series,
                   test_size=0.2, random_state=42):
    """
    Returns: X_train, X_test, y_train, y_test, scaler, feature_names
    Excludes robustness column from features (it's the source of the target).
    """
    X = features.drop(columns=["robustness"])
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=test_size, random_state=random_state,
        stratify=target if target.nunique() == 2 else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, X_test_scaled,
            y_train.values, y_test.values,
            scaler, feature_names)


def apply_pca(X_train, X_test, n_components=15, random_state=42):
    """Optional dimensionality reduction for visualization and comparison."""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


# =============================================================================
# Convenience Pipeline
# =============================================================================

def run_full_pipeline(sample_frac=None, random_state=42):
    """
    One-shot pipeline: load → clean → aggregate → engineer → split → scale.
    Returns a dict with everything downstream modules need.

    sample_frac: if set to e.g. 0.3, uses only that fraction of experiments
    (useful for faster iteration; leave None for full data).
    """
    print("[preprocessing] Loading raw CSV...")
    raw = load_raw()

    if sample_frac is not None and sample_frac < 1.0:
        all_exps = raw["experiment_number"].unique()
        rng = np.random.default_rng(random_state)
        keep = rng.choice(all_exps, size=int(len(all_exps) * sample_frac),
                         replace=False)
        raw = raw[raw["experiment_number"].isin(keep)]
        print(f"[preprocessing] Subsampled to {len(keep)} experiments")

    print("[preprocessing] Cleaning...")
    clean_df = clean(raw)

    print("[preprocessing] Aggregating per experiment...")
    features = aggregate_per_experiment(clean_df)

    print("[preprocessing] Engineering derived features...")
    features = engineer_derived_features(features)

    print(f"[preprocessing] Final feature matrix: {features.shape}")

    y_binary = make_binary_label(features)
    y_regression = features["robustness"].copy()

    print("[preprocessing] Splitting and scaling for classification...")
    X_train, X_test, y_train, y_test, scaler, feat_names = split_and_scale(
        features, y_binary, random_state=random_state
    )

    print("[preprocessing] Splitting and scaling for regression...")
    Xr_train, Xr_test, yr_train, yr_test, _, _ = split_and_scale(
        features, y_regression, random_state=random_state
    )

    summary = describe_dataset(raw)

    return {
        "raw_summary": summary,
        "features": features,
        "feature_names": feat_names,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "Xr_train": Xr_train, "Xr_test": Xr_test,
        "yr_train": yr_train, "yr_test": yr_test,
        "scaler": scaler,
        "robustness_median": float(features["robustness"].median()),
    }


if __name__ == "__main__":
    data = run_full_pipeline()
    print("\n=== Dataset Summary ===")
    for k, v in data["raw_summary"].items():
        print(f"  {k}: {v}")
    print(f"\nClassification shapes: "
          f"X_train={data['X_train'].shape}, X_test={data['X_test'].shape}")
    print(f"Class balance in train: {np.bincount(data['y_train'])}")
