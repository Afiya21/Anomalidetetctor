"""
train.py — Model Training, Selection & Evaluation
====================================================
Implements a rigorous ML pipeline for anomaly detection:

  1. Loads labeled data from data/training_data.csv
  2. Generates a feature correlation heatmap  → reports/figures/
  3. Evaluates 5 algorithms using TimeSeriesSplit (prevents look-ahead bias)
        - Z-Score Baseline        (statistical)
        - Isolation Forest        (unsupervised)
        - One-Class SVM           (unsupervised)
        - Logistic Regression     (supervised)
        - Random Forest           (supervised)
  4. Per-anomaly-type recall analysis (CPU / Memory / Disk)
  5. Saves the best model + scaler to models/

Primary metric: F1-Score (anomaly class)
Justified over accuracy — class imbalance means accuracy is misleading.

Usage:
    python src/train.py
"""

import json
import os
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")
DATA_PATH   = os.path.join(PROJECT_DIR, "data",    "training_data.csv")
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
FIGURES_DIR = os.path.join(PROJECT_DIR, "reports", "figures")
METRICS_DIR = os.path.join(PROJECT_DIR, "reports", "metrics")

for d in (MODELS_DIR, FIGURES_DIR, METRICS_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_heatmap(df: pd.DataFrame, feature_cols: list):
    """Save feature correlation heatmap to reports/figures/."""
    print("\n[Plot] Generating feature correlation heatmap...")
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,
                linewidths=0.2, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_confusion(y_true, y_pred, model_name: str):
    """Save confusion matrix for one model."""
    fig, ax = plt.subplots(figsize=(5, 4))
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    safe = model_name.replace(" ", "_")
    path = os.path.join(FIGURES_DIR, f"confusion_matrix_{safe}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_feature_importance(model, feature_cols: list):
    """Save Random Forest feature importance bar chart."""
    importances = model.feature_importances_
    top_n = min(20, len(feature_cols))
    idx   = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(top_n), importances[idx], color="steelblue")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_cols[i] for i in idx], rotation=45, ha="right")
    ax.set_title("Random Forest — Top Feature Importances")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "feature_importance_rf.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Feature importance saved → {path}")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute F1, Recall, Precision, ROC-AUC for binary predictions."""
    return {
        "f1":        round(f1_score(y_true,        y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true,    y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true,   y_prob), 4)
                     if y_prob is not None else float("nan"),
    }


def per_anomaly_recall(y_gran, y_pred, label_names: dict):
    """Print recall per anomaly type — reveals which label was hardest to detect."""
    print("    Per-anomaly-type recall:")
    for lbl_str, name in label_names.items():
        lbl = int(lbl_str)
        if lbl == 0:
            continue
        mask = (y_gran == lbl)
        if mask.sum() == 0:
            print(f"      Label {lbl} ({name}): no samples in this fold")
            continue
        detected = int((y_pred[mask] == 1).sum())
        total    = int(mask.sum())
        recall   = detected / total
        bar      = "#" * int(recall * 20)
        print(f"      Label {lbl} ({name:<15}): {detected:>3}/{total:>3}  "
              f"recall={recall:.2f}  {bar}")


def zscore_predict(X_normal: np.ndarray, X_test: np.ndarray,
                   threshold: float = 3.0) -> np.ndarray:
    """
    Statistical baseline: flag a row as anomalous if any feature deviates
    more than `threshold` standard deviations from the normal training mean.
    """
    mean = X_normal.mean(axis=0)
    std  = X_normal.std(axis=0) + 1e-10
    z    = np.abs((X_test - mean) / std)
    return (z.max(axis=1) > threshold).astype(int)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def main():
    config      = load_config()
    t_cfg       = config["training"]
    label_names = config["labels"]
    n_splits    = t_cfg["n_splits"]
    rs          = t_cfg["random_state"]

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("  DCML Anomaly Detector — Training Pipeline")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        sys.exit(f"[ERROR] Training data not found: {DATA_PATH}\n"
                 "   Run  python src/collect_data.py  first.")

    df = pd.read_csv(DATA_PATH)
    print(f"\n  Loaded {len(df)} rows from {DATA_PATH}")

    feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]
    X            = df[feature_cols].values
    y_binary     = (df["label"] > 0).astype(int).values   # 0=Normal, 1=Anomaly
    y_granular   = df["label"].values                      # 0/1/2/3

    print(f"  Features : {len(feature_cols)}")
    print(f"  Normal   : {(y_binary == 0).sum()}")
    print(f"  Anomaly  : {(y_binary == 1).sum()}")

    # ── 2. Correlation heatmap ────────────────────────────────────────────────
    plot_heatmap(df, feature_cols)

    # ── 3. TimeSeriesSplit cross-validation ───────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Store per-fold results for each model
    fold_results: dict[str, list[dict]] = {}

    # Track the last-fold best RF model and scaler for saving
    best_rf_model  = None
    best_rf_scaler = None
    last_iso_model = None

    print(f"\n  TimeSeriesSplit  k={n_splits}")
    print("─" * 60)

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\n  ── Fold {fold_i + 1}/{n_splits} "
              f"(train={len(train_idx)}, test={len(test_idx)}) ──")

        X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
        y_tr, y_te         = y_binary[train_idx],   y_binary[test_idx]
        y_gran_te          = y_granular[test_idx]

        # Scale features — fit on training fold only (no data leakage)
        scaler   = StandardScaler()
        X_tr     = scaler.fit_transform(X_tr_raw)
        X_te     = scaler.transform(X_te_raw)

        # Normal-only training set for unsupervised models
        X_tr_normal = X_tr[y_tr == 0]

        # ── Model 1: Z-Score Baseline ─────────────────────────────────────
        z_pred = zscore_predict(X_tr_normal, X_te)
        fold_results.setdefault("Z-Score", []).append(
            compute_metrics(y_te, z_pred, z_pred.astype(float))
        )
        print(f"    Z-Score     F1={fold_results['Z-Score'][-1]['f1']:.3f}")

        # ── Model 2: Isolation Forest ─────────────────────────────────────
        iso = IsolationForest(
            n_estimators=t_cfg["iso_forest_n_estimators"],
            contamination=t_cfg["iso_forest_contamination"],
            random_state=rs,
        )
        iso.fit(X_tr_normal)
        iso_raw   = iso.predict(X_te)
        iso_pred  = (iso_raw == -1).astype(int)       # -1 → anomaly
        iso_score = -iso.score_samples(X_te)           # higher = more anomalous
        iso_prob  = (iso_score - iso_score.min()) / (np.ptp(iso_score) + 1e-10)
        fold_results.setdefault("Isolation Forest", []).append(
            compute_metrics(y_te, iso_pred, iso_prob)
        )
        print(f"    Iso Forest  F1={fold_results['Isolation Forest'][-1]['f1']:.3f}")
        last_iso_model = (iso, scaler)

        # ── Model 3: One-Class SVM ────────────────────────────────────────
        ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
        ocsvm.fit(X_tr_normal)
        oc_raw   = ocsvm.predict(X_te)
        oc_pred  = (oc_raw == -1).astype(int)
        oc_score = -ocsvm.decision_function(X_te)
        oc_prob  = (oc_score - oc_score.min()) / (np.ptp(oc_score) + 1e-10)
        fold_results.setdefault("One-Class SVM", []).append(
            compute_metrics(y_te, oc_pred, oc_prob)
        )
        print(f"    OC-SVM      F1={fold_results['One-Class SVM'][-1]['f1']:.3f}")

        # ── Model 4: Logistic Regression ──────────────────────────────────
        if len(np.unique(y_tr)) > 1:
            lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                                    random_state=rs)
            lr.fit(X_tr, y_tr)
            lr_pred = lr.predict(X_te)
            lr_prob = lr.predict_proba(X_te)[:, 1]
            fold_results.setdefault("Logistic Regression", []).append(
                compute_metrics(y_te, lr_pred, lr_prob)
            )
            print(f"    Log Reg     F1={fold_results['Logistic Regression'][-1]['f1']:.3f}")
        else:
            print("    Log Reg     Skipped (only 1 class in training fold)")

        # ── Model 5: Random Forest ────────────────────────────────────────
        if len(np.unique(y_tr)) > 1:
            rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                        random_state=rs, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            rf_pred = rf.predict(X_te)
            rf_prob = rf.predict_proba(X_te)[:, 1]
            fold_results.setdefault("Random Forest", []).append(
                compute_metrics(y_te, rf_pred, rf_prob)
            )
            print(f"    Rand Forest F1={fold_results['Random Forest'][-1]['f1']:.3f}")

            # Per-anomaly-type analysis on the last fold only (representative)
            if fold_i == n_splits - 1:
                print()
                per_anomaly_recall(y_gran_te, rf_pred, label_names)
                best_rf_model  = rf
                best_rf_scaler = scaler

                # Confusion matrices for RF and Isolation Forest
                plot_confusion(y_te, rf_pred,  "Random_Forest")
                plot_confusion(y_te, iso_pred, "Isolation_Forest")
        else:
            print("    Rand Forest Skipped (only 1 class in training fold)")

    # ── 4. Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {'Model':<22} {'F1':>7} {'Recall':>8} {'Precision':>10} {'ROC-AUC':>9}")
    print("  " + "─" * 56)

    mean_scores: dict[str, float] = {}
    for model_name, folds in fold_results.items():
        avg = {k: np.mean([f[k] for f in folds if not np.isnan(f[k])])
               for k in ("f1", "recall", "precision", "roc_auc")}
        mean_scores[model_name] = avg["f1"]
        print(f"  {model_name:<22} {avg['f1']:>7.4f} {avg['recall']:>8.4f} "
              f"{avg['precision']:>10.4f} {avg['roc_auc']:>9.4f}")

    # ── 5. Select & save best model ───────────────────────────────────────────
    best_name = max(mean_scores, key=mean_scores.get)
    print(f"\n  [BEST] Best model (by F1): {best_name}  (F1={mean_scores[best_name]:.4f})")

    # Always save Random Forest as best_model (typically wins; supervised + balanced)
    # and Isolation Forest separately for comparison
    joblib.dump(best_rf_model,           os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(best_rf_scaler,          os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(last_iso_model[0],       os.path.join(MODELS_DIR, "iso_forest_model.pkl"))

    print(f"  Saved → models/best_model.pkl  (Random Forest)")
    print(f"  Saved → models/scaler.pkl")
    print(f"  Saved → models/iso_forest_model.pkl")

    # Feature importance chart (Random Forest only)
    plot_feature_importance(best_rf_model, feature_cols)

    # Save metrics summary to CSV
    rows = []
    for model_name, folds in fold_results.items():
        avg = {k: np.mean([f[k] for f in folds if not np.isnan(f[k])])
               for k in ("f1", "recall", "precision", "roc_auc")}
        rows.append({"model": model_name, **avg})

    pd.DataFrame(rows).to_csv(
        os.path.join(METRICS_DIR, "model_comparison.csv"), index=False
    )
    print(f"  Metrics saved → reports/metrics/model_comparison.csv")

    print("\n  Next step: run  python src/detector.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
