"""
Regression: predict the continuous 'robustness' score of a grasp.

A robot doesn't just need to know whether a grasp succeeds; estimating
how stable a grasp is can be useful for ranking candidate grasps or
deciding whether to re-plan. This module implements Linear Regression
and Polynomial Regression on the same engineered features.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
METRIC_DIR = os.path.join(RESULTS_DIR, "metrics")


def _evaluate_regressor(model, X_test, y_test, name, train_time):
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    return {
        "model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "train_time_sec": train_time,
    }, y_pred


# =============================================================================
# Linear Regression
# =============================================================================

def run_linear_regression(X_train, X_test, y_train, y_test):
    print("\n[LinearRegression] Training...")
    model = LinearRegression()
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    m, y_pred = _evaluate_regressor(model, X_test, y_test, "LinearRegression",
                                    train_time)
    print(f"[LinearRegression] RMSE={m['RMSE']:.3f}, MAE={m['MAE']:.3f}, "
          f"R2={m['R2']:.4f}")
    return model, m, y_pred


# =============================================================================
# Polynomial Regression
# =============================================================================

def run_polynomial_regression(X_train, X_test, y_train, y_test, degree=2,
                              n_features=20):
    # Using degree-2 polynomial features on 91 features → ~4000+ features,
    # which overfits and slows things down. We first pick the top-variance
    # features to keep the expansion manageable.
    print(f"\n[PolynomialRegression] Training with degree={degree}...")

    variances = X_train.var(axis=0)
    top_idx = np.argsort(variances)[-n_features:]
    X_train_top = X_train[:, top_idx]
    X_test_top = X_test[:, top_idx]

    # Ridge inside a polynomial pipeline to regularize the expanded space
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("ridge", Ridge(alpha=1.0)),
    ])
    t0 = time.time()
    pipeline.fit(X_train_top, y_train)
    train_time = time.time() - t0
    m, y_pred = _evaluate_regressor(pipeline, X_test_top, y_test,
                                    "PolynomialRegression", train_time)
    print(f"[PolynomialRegression] RMSE={m['RMSE']:.3f}, MAE={m['MAE']:.3f}, "
          f"R2={m['R2']:.4f}")
    return pipeline, m, y_pred


# =============================================================================
# Visualization
# =============================================================================

def plot_predictions_vs_actual(results_dict, save_path):
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, payload) in zip(axes, results_dict.items()):
        y_true = payload["y_true"]
        y_pred = payload["y_pred"]
        ax.scatter(y_true, y_pred, alpha=0.25, s=8)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual robustness")
        ax.set_ylabel("Predicted robustness")
        ax.set_title(f"{name}\nR²={payload['metrics']['R2']:.3f}, "
                     f"RMSE={payload['metrics']['RMSE']:.2f}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_residuals(results_dict, save_path):
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, payload) in zip(axes, results_dict.items()):
        residuals = payload["y_true"] - payload["y_pred"]
        ax.hist(residuals, bins=60, edgecolor="black", alpha=0.75)
        ax.axvline(0, color="red", linestyle="--")
        ax.set_xlabel("Residual (actual − predicted)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{name} — Residuals")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# =============================================================================
# Full Regression Pipeline
# =============================================================================

def _eval_from_predictions(y_true, y_pred, name, train_time):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "train_time_sec": train_time,
    }


def run_all_regressors(data):
    Xr_train, Xr_test = data["Xr_train"], data["Xr_test"]
    yr_train_raw, yr_test_raw = data["yr_train"], data["yr_test"]

    # Robustness spans orders of magnitude (from ~0 to 2900+) with a heavy tail.
    # Log-transform gives a well-behaved target for linear methods; we report
    # metrics on the back-transformed (original-scale) predictions.
    yr_train = np.log1p(yr_train_raw)
    yr_test = np.log1p(yr_test_raw)

    results = {}
    pred_payload = {}
    models = {}

    # Linear Regression
    lin_model, _, lin_pred_log = run_linear_regression(
        Xr_train, Xr_test, yr_train, yr_test
    )
    # Clip to training log-space range before inverting — guards against
    # occasional extrapolated values exploding after expm1().
    lin_pred_log = np.clip(lin_pred_log, yr_train.min(), yr_train.max())
    lin_pred = np.expm1(lin_pred_log)
    lin_metrics = _eval_from_predictions(yr_test_raw, lin_pred,
                                         "LinearRegression", 0.0)
    results["LinearRegression"] = lin_metrics
    models["LinearRegression"] = lin_model
    pred_payload["LinearRegression"] = {
        "y_true": yr_test_raw, "y_pred": lin_pred, "metrics": lin_metrics
    }

    # Polynomial Regression
    poly_model, _, poly_pred_log = run_polynomial_regression(
        Xr_train, Xr_test, yr_train, yr_test
    )
    # Clip log-space predictions to observed training range before inverting;
    # polynomial expansions can extrapolate wildly on unseen feature regions.
    poly_pred_log = np.clip(poly_pred_log, yr_train.min(), yr_train.max())
    poly_pred = np.expm1(poly_pred_log)
    poly_metrics = _eval_from_predictions(yr_test_raw, poly_pred,
                                          "PolynomialRegression", 0.0)
    results["PolynomialRegression"] = poly_metrics
    models["PolynomialRegression"] = poly_model
    pred_payload["PolynomialRegression"] = {
        "y_true": yr_test_raw, "y_pred": poly_pred, "metrics": poly_metrics
    }

    plot_predictions_vs_actual(pred_payload,
                               os.path.join(FIG_DIR, "regression_predictions.png"))
    plot_residuals(pred_payload,
                   os.path.join(FIG_DIR, "regression_residuals.png"))

    with open(os.path.join(METRIC_DIR, "regression_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== Regression Summary ===")
    print(pd.DataFrame(results).T[["RMSE", "MAE", "R2", "train_time_sec"]])

    return models, results


if __name__ == "__main__":
    from preprocessing import run_full_pipeline
    data = run_full_pipeline(sample_frac=0.2)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)
    run_all_regressors(data)
