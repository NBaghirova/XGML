#!/usr/bin/env python3
"""
XGML Prediction Pipeline
=========================
Explainable Graph-theoretical Machine Learning for cognitive score prediction.

This script runs two stages:

  A) Internal validation (cross-validation on the training dataset)
     - Repeated stratified 3-fold CV (10 repetitions = 30 outer folds)
     - Main model: Kernel SVR with Optuna inner tuning
     - Baselines: Linear Regression, Ridge, MultiTask ElasticNet,
                  Random Forest, MLP
     - Permutation feature importance on a balanced 150-subject subset

  B) External validation (train on dataset A, predict on dataset B)
     - Subject-wise z-score normalisation applied independently to
       each dataset (zero data leakage)
     - Kernel SVR tuned on dataset A via Optuna
     - Predicts on dataset B and reports Pearson r, RMSE, MAE, R²

Usage
-----
1. Fill in the CONFIGURATION section below.
2. Run:  python xgml_prediction_pipeline.py

Input CSV format (both ADNI-style and external datasets)
---------------------------------------------------------
Required columns:
    subject_id   : unique subject identifier
    pet_id       : unique PET scan identifier
    group        : diagnostic group label (CN / MCI / AD) — required for
                   internal validation; optional for external test set
    <target_cols>: one column per cognitive score to predict

Feature files are expected at:
    <GRAPH_ROOT>/<subject_id>/<subject_id>_<pet_id>_features.npy
Each file must be a 1-D float64 NumPy array of length
    N_REGIONS * (N_REGIONS - 1) / 2   (upper-triangle of the distance matrix)

Requirements
------------
See requirements.txt

Reference
---------
If you use this code, please cite:
     Explainable Graph-theoretical Machine Learning: with Application to Alzheimer's Disease Prediction
	Authors: Narmina Baghirova, Duy-Thanh Vũ, Duy-Cat Can, Christelle Schneuwly Diaz, Julien Bodlet, Guillaume Blanc, Georgi Hrusanov, Bernard Ries, Oliver Y. Chén


Data access
-----------
ADNI:  https://adni.loni.usc.edu  (registration required)
OASIS: https://www.oasis-brains.org  (registration required)
Data are NOT distributed with this code.
"""

# =============================================================================
# CONFIGURATION — fill in your paths and settings here
# =============================================================================

# ---- Stage switches ----
RUN_INTERNAL_VALIDATION  = True   # internal cross-validation on TRAIN_CSV
RUN_EXTERNAL_VALIDATION  = True   # external validation: train on TRAIN_CSV, test on EXTERNAL_CSV
RUN_PERMUTATION_IMPORTANCE = True # permutation importance (only when RUN_INTERNAL_VALIDATION=True)

# ---- Internal validation paths ----
# CSV with cohort used for internal cross-validation
TRAIN_CSV        = "*** INSERT PATH TO INTERNAL COHORT CSV ***"
# Example: TRAIN_CSV = "/data/adni/cohort_600.csv"

# Root directory containing per-subject feature .npy files
TRAIN_GRAPH_ROOT = "*** INSERT PATH TO INTERNAL GRAPH FEATURES DIRECTORY ***"
# Example: TRAIN_GRAPH_ROOT = "/data/adni/graphs"

# Directory where all results will be written
RESULTS_ROOT     = "*** INSERT PATH TO OUTPUT DIRECTORY ***"
# Example: RESULTS_ROOT = "/data/results/xgml_run1"

# ---- Internal targets ----
# List of column names in TRAIN_CSV to predict
INTERNAL_TARGETS = [
    "CDRSB",
    "ADAS11",
    "ADAS13",
    "ADASQ4",
    "MMSE",
    "RAVLT_immediate",
    "RAVLT_learning",
    "RAVLT_perc_forgetting",
]

# ---- External validation paths ----
# CSV for the external test cohort (must contain subject_id, pet_id, and
# the columns named in EXTERNAL_TARGET_MAP values)
EXTERNAL_CSV        = "*** INSERT PATH TO EXTERNAL COHORT CSV ***"
# Example: EXTERNAL_CSV = "/data/oasis3/cohort_101.csv"

EXTERNAL_GRAPH_ROOT = "*** INSERT PATH TO EXTERNAL GRAPH FEATURES DIRECTORY ***"
# Example: EXTERNAL_GRAPH_ROOT = "/data/oasis3/graphs"

# ---- External target mapping ----
# Maps each internal target name → the corresponding column name in
# EXTERNAL_CSV.  Only targets listed here are used for external validation.
# Adjust to match your external dataset's column names.
EXTERNAL_TARGET_MAP = {
    "CDRSB": "CDRSUM",   # ADNI CDRSB  <->  OASIS3 CDRSUM
    "MMSE":  "MMSE",     # identical in both datasets
}

# ---- CPU usage ----
# Fraction of available logical CPU cores to use for parallel jobs (0.0–1.0)
CPU_FRACTION = 0.50

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# CPU / thread control — must be set BEFORE importing numpy / sklearn
# -----------------------------------------------------------------------------
import os

TOTAL_CPUS = os.cpu_count() or 1
N_JOBS     = max(1, int(TOTAL_CPUS * CPU_FRACTION))

os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.stats import pearsonr

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, MultiTaskElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

optuna.logging.set_verbosity(optuna.logging.WARNING)

# -----------------------------------------------------------------------------
# Derived paths (do not edit)
# -----------------------------------------------------------------------------
_RESULTS_ROOT     = Path(RESULTS_ROOT)
_TRAIN_CSV        = Path(TRAIN_CSV)
_TRAIN_GRAPH_ROOT = Path(TRAIN_GRAPH_ROOT)
_EXTERNAL_CSV        = Path(EXTERNAL_CSV)
_EXTERNAL_GRAPH_ROOT = Path(EXTERNAL_GRAPH_ROOT)

_INTERNAL_OUT = _RESULTS_ROOT / "internal_validation"
_EXTERNAL_OUT = _RESULTS_ROOT / "external_validation"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
N_REGIONS        = 200
EXPECTED_FEATURES = N_REGIONS * (N_REGIONS - 1) // 2   # 19 900
VALID_GROUPS     = {"CN", "MCI", "AD"}

# Cross-validation design
OUTER_CV_N_SPLITS = 3
OUTER_CV_REPEATS  = 10
OUTER_CV_SEEDS    = list(range(1, OUTER_CV_REPEATS + 1))
INNER_CV_FOLDS    = 5

# Kernel SVR Optuna search (internal)
INTERNAL_OPTUNA_TRIALS = 20
INTERNAL_OPTUNA_SPACE  = {
    "C_low": 5.0, "C_high": 30.0,
    "gamma_low": 3e-4, "gamma_high": 3e-3,
    "epsilon_low": 0.10, "epsilon_high": 0.30,
}

# Kernel SVR Optuna search (external)
EXTERNAL_OPTUNA_TRIALS = 30
EXTERNAL_OPTUNA_SEED   = 42
EXTERNAL_INNER_CV_SEED = 3000

# Permutation importance
IMPORTANCE_PER_GROUP  = 50   # balanced subset: 50 CN + 50 MCI + 50 AD = 150
IMPORTANCE_SEED       = 42
PERMUTATION_REPEATS   = 5
TOP_IMPORTANT_EDGES   = 15

# Baselines whose hyperparameters are tuned inside each outer fold
TUNED_BASELINE_MODELS = {"MultiTaskElasticNet_Standard", "RandomForest"}


# =============================================================================
# Utility helpers
# =============================================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):   return bool(obj)
    if isinstance(obj, np.ndarray):    return obj.tolist()
    return str(obj)


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def append_df(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def feat_path(graph_root: Path, sid: str, pid: str) -> Path:
    return graph_root / sid / f"{sid}_{pid}_features.npy"


def score_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    try:
        pr    = pearsonr(y_true, y_pred)
        r_val = float(pr.statistic) if hasattr(pr, "statistic") else float(pr[0])
        p_val = float(pr.pvalue)    if hasattr(pr, "pvalue")    else float(pr[1])
    except Exception:
        r_val = p_val = float("nan")
    try:    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    except: rmse = float("nan")
    try:    mae  = float(mean_absolute_error(y_true, y_pred))
    except: mae  = float("nan")
    try:    r2   = float(r2_score(y_true, y_pred))
    except: r2   = float("nan")
    return {"pearson_r": r_val, "pearson_pvalue": p_val,
            "rmse": rmse, "mae": mae, "r2": r2}


def warn_summary(caught) -> str:
    msgs = list({f"{w.category.__name__}: {w.message}" for w in caught})
    return " | ".join(msgs)


def fit_logged(model, X, y):
    t0 = time.time()
    status = "ok"
    wmsg   = ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model.fit(X, y)
        except Exception as e:
            return None, "failed", str(e), time.time() - t0
    wmsg = warn_summary(caught)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        status = "warning_convergence"
    return model, status, wmsg, time.time() - t0


def predict_logged(model, X):
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            y_pred = model.predict(X)
        except Exception as e:
            return None, "failed", str(e), time.time() - t0
    return y_pred, "ok", warn_summary(caught), time.time() - t0


def norm_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "group" in df.columns:
        df["group"] = df["group"].astype(str).str.strip().str.upper()
        df.loc[df["group"].isin(["", "NAN", "NONE"]), "group"] = float("nan")
    return df


# =============================================================================
# Subject-wise normalisation
# =============================================================================
def subject_wise_normalize(X: np.ndarray) -> np.ndarray:
    """
    Normalise each subject's feature vector independently to mean=0, std=1.

    Applied separately to the training and external datasets so that no
    statistics from one dataset influence the other (zero data leakage).
    """
    X     = np.array(X, dtype=np.float64)
    means = X.mean(axis=1, keepdims=True)
    stds  = X.std(axis=1, keepdims=True)
    n_zero = int((stds == 0).sum())
    if n_zero > 0:
        print(f"[WARN] {n_zero} subject(s) with zero feature std — left as zeros.")
    stds[stds == 0] = 1.0
    return (X - means) / stds


# =============================================================================
# Dataset loaders
# =============================================================================
def load_dataset(
    csv_path: Path,
    graph_root: Path,
    target_cols: list[str],
    name: str,
    require_group: bool,
    log_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load feature vectors and target values for one cohort."""
    if not csv_path.exists():
        raise FileNotFoundError(f"{name} CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df = norm_group(df)

    required = ["subject_id", "pet_id"] + target_cols
    if require_group:
        required += ["group"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} CSV missing columns: {missing}")

    df = df.dropna(subset=target_cols + (["group"] if require_group else [])).copy()

    if require_group:
        bad = sorted(set(df["group"].dropna().unique()) - VALID_GROUPS)
        if bad:
            raise ValueError(f"{name}: unexpected group labels {bad}. "
                             f"Expected {sorted(VALID_GROUPS)}.")

    rows, feats, skipped = [], [], []
    for _, row in df.iterrows():
        sid = str(row["subject_id"]).strip()
        pid = str(row["pet_id"]).strip()
        fp  = feat_path(graph_root, sid, pid)
        if not fp.exists():
            skipped.append({"subject_id": sid, "pet_id": pid,
                            "reason": f"not found: {fp}"})
            continue
        try:
            f = np.load(fp).reshape(-1).astype(np.float64)
        except Exception as e:
            skipped.append({"subject_id": sid, "pet_id": pid,
                            "reason": str(e)})
            continue
        if f.size != EXPECTED_FEATURES:
            skipped.append({"subject_id": sid, "pet_id": pid,
                            "reason": f"size {f.size} != {EXPECTED_FEATURES}"})
            continue
        rows.append(row.to_dict())
        feats.append(f)

    if not rows:
        raise RuntimeError(f"No valid {name} rows found.")

    ensure_dir(log_dir)
    pd.DataFrame(skipped).to_csv(log_dir / f"{name.lower()}_skipped.csv", index=False)

    df_out = pd.DataFrame(rows).reset_index(drop=True)
    X      = np.vstack(feats)
    Y      = df_out[target_cols].to_numpy(dtype=np.float64)

    print(f"[{name}] {len(df_out)} subjects loaded  |  {len(skipped)} skipped")
    print(f"[{name}] X shape: {X.shape}   Y shape: {Y.shape}")
    if "group" in df_out.columns:
        print(f"[{name}] Groups:\n{df_out['group'].value_counts(dropna=False).to_string()}")
    return df_out, X, Y


# =============================================================================
# Model builders
# =============================================================================
class ScaledRegressor(BaseEstimator, RegressorMixin):
    """Wrapper that applies a scaler before fitting a true multi-output regressor."""
    def __init__(self, scaler=None, regressor=None):
        self.scaler    = scaler
        self.regressor = regressor

    def fit(self, X, y):
        self.scaler_    = clone(self.scaler)
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(self.scaler_.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.regressor_.predict(self.scaler_.transform(X))


def svr_pipeline(C, gamma, epsilon, scaler_name):
    scaler = MinMaxScaler(feature_range=(0, 1)) if scaler_name == "minmax" \
             else StandardScaler()
    return Pipeline([
        ("scaler", scaler),
        ("regressor", MultiOutputRegressor(
            SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon), n_jobs=1
        )),
    ])


def baselines(seed: int) -> dict:
    models = {
        "LinearRegression_Standard": Pipeline([
            ("scaler", StandardScaler()), ("regressor", LinearRegression())
        ]),
        "Ridge_Standard": Pipeline([
            ("scaler", StandardScaler()), ("regressor", Ridge(alpha=1.0))
        ]),
        "MultiTaskElasticNet_Standard": ScaledRegressor(
            scaler=StandardScaler(),
            regressor=MultiTaskElasticNet(alpha=0.01, l1_ratio=0.5,
                                          max_iter=30000, random_state=seed),
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_split=2,
            min_samples_leaf=2, max_features="sqrt",
            random_state=seed, n_jobs=N_JOBS,
        ),
        "MLP_Standard": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", MLPRegressor(
                hidden_layer_sizes=(256, 128), activation="relu",
                solver="adam", alpha=0.001, max_iter=500,
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=30, random_state=seed,
            )),
        ]),
    }
    return models


def tune_baseline(name: str, seed: int, X_tr, y_tr, groups):
    inner_cv = list(StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True,
                                    random_state=4000 + seed).split(X_tr, groups))
    if name == "RandomForest":
        return GridSearchCV(
            RandomForestRegressor(random_state=seed, n_jobs=1),
            {"n_estimators": [300, 500], "max_features": ["sqrt"],
             "min_samples_leaf": [1, 2, 4]},
            scoring="neg_mean_squared_error", cv=inner_cv,
            n_jobs=N_JOBS, refit=True,
        )
    if name == "MultiTaskElasticNet_Standard":
        return GridSearchCV(
            ScaledRegressor(scaler=StandardScaler(),
                            regressor=MultiTaskElasticNet(max_iter=30000,
                                                          random_state=seed)),
            {"regressor__alpha": [0.001, 0.01, 0.1],
             "regressor__l1_ratio": [0.2, 0.5, 0.8]},
            scoring="neg_mean_squared_error", cv=inner_cv,
            n_jobs=N_JOBS, refit=True,
        )
    return None


# =============================================================================
# Optuna objectives
# =============================================================================
def _internal_optuna_objective(trial, X_tr, Y_tr, groups, inner_seed):
    C       = trial.suggest_float("C", INTERNAL_OPTUNA_SPACE["C_low"],
                                  INTERNAL_OPTUNA_SPACE["C_high"], log=True)
    gamma   = trial.suggest_float("gamma", INTERNAL_OPTUNA_SPACE["gamma_low"],
                                  INTERNAL_OPTUNA_SPACE["gamma_high"], log=True)
    eps     = trial.suggest_float("epsilon", INTERNAL_OPTUNA_SPACE["epsilon_low"],
                                  INTERNAL_OPTUNA_SPACE["epsilon_high"])
    scaler  = trial.suggest_categorical("scaler", ["minmax", "standard"])

    splits  = list(StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True,
                                   random_state=inner_seed).split(X_tr, groups))
    losses  = []
    for tr, val in splits:
        m = svr_pipeline(C, gamma, eps, scaler)
        m.fit(X_tr[tr], Y_tr[tr])
        yp = m.predict(X_tr[val])
        losses.append(float(np.mean(mean_squared_error(Y_tr[val], yp,
                                                        multioutput="raw_values"))))
    return float(np.mean(losses))


def _external_optuna_objective(trial, X_tr, Y_tr, groups, inner_seed):
    C      = trial.suggest_float("C", 3.0, 30.0, log=True)
    gamma  = trial.suggest_float("gamma", 3e-4, 3e-3, log=True)
    eps    = trial.suggest_float("epsilon", 0.1, 0.3)
    scaler = trial.suggest_categorical("scaler", ["minmax", "standard"])

    splits = list(StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True,
                                  random_state=inner_seed).split(X_tr, groups))
    losses = []
    for tr, val in splits:
        m = svr_pipeline(C, gamma, eps, scaler)
        m.fit(X_tr[tr], Y_tr[tr])
        yp = m.predict(X_tr[val])
        losses.append(float(np.mean(mean_squared_error(Y_tr[val], yp,
                                                        multioutput="raw_values"))))
    return float(np.mean(losses))


# =============================================================================
# Summary tables
# =============================================================================
def build_summary_tables(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    if metrics_df.empty:
        return
    by_mt = (
        metrics_df
        .groupby(["model", "score_name"], dropna=False)
        .agg(pearson_r_mean=("pearson_r","mean"), pearson_r_std=("pearson_r","std"),
             rmse_mean=("rmse","mean"), rmse_std=("rmse","std"),
             mae_mean=("mae","mean"),  mae_std=("mae","std"),
             r2_mean=("r2","mean"),    r2_std=("r2","std"))
        .reset_index()
        .sort_values(["score_name","model"])
    )
    by_mt.to_csv(out_dir / "metrics_by_model_and_target.csv", index=False)

    mean_tgt = (
        by_mt
        .groupby("model", dropna=False)
        .agg(r_mean=("pearson_r_mean","mean"), r_std=("pearson_r_mean","std"),
             rmse_mean=("rmse_mean","mean"),   mae_mean=("mae_mean","mean"),
             r2_mean=("r2_mean","mean"))
        .reset_index()
        .sort_values("r_mean", ascending=False)
    )
    mean_tgt.to_csv(out_dir / "metrics_mean_over_targets.csv", index=False)

    pivot = by_mt.pivot(index="model", columns="score_name",
                        values="pearson_r_mean")
    pivot.to_csv(out_dir / "pearson_r_pivot.csv")


def export_per_score_csvs(pred_csv: Path, targets: list[str],
                          out_dir: Path, model_filter: str | None = None) -> None:
    if not pred_csv.exists():
        return
    df = pd.read_csv(pred_csv)
    if model_filter and "model" in df.columns:
        df = df[df["model"] == model_filter].copy()
    per = out_dir / "per_score"
    ensure_dir(per)
    base = [c for c in ["subject_id","pet_id","group","outer_repeat","outer_fold",
                        "global_outer_fold_id","model","best_params_json"]
            if c in df.columns]
    for sc in targets:
        ac, pc, rc = f"{sc}_actual", f"{sc}_predicted", f"{sc}_residual"
        if not all(c in df.columns for c in [ac, pc, rc]):
            continue
        out = df[base + [ac, pc, rc]].rename(columns={ac:"Actual",pc:"Predicted",rc:"Residual"})
        out.to_csv(per / f"{sc}_actual_predicted.csv", index=False)


# =============================================================================
# A. Internal validation
# =============================================================================
def run_internal_validation(
    train_df:    pd.DataFrame,
    X:           np.ndarray,
    Y:           np.ndarray,
    targets:     list[str],
    out_dir:     Path,
) -> None:
    print("\n" + "="*80, flush=True)
    print("INTERNAL VALIDATION", flush=True)
    print("="*80, flush=True)

    dirs = {k: out_dir/k for k in
            ["logs","splits","predictions","metrics","importance","tuning"]}
    for d in dirs.values():
        ensure_dir(d)

    pred_csv    = dirs["predictions"] / "all_predictions.csv"
    metrics_csv = dirs["metrics"]     / "metrics_per_outer_fold.csv"
    status_csv  = dirs["logs"]        / "model_status.csv"

    for p in [pred_csv, metrics_csv, status_csv]:
        if p.exists():
            p.unlink()

    global_fold = 0
    t0_global   = time.time()

    for rep_idx, seed in enumerate(OUTER_CV_SEEDS, 1):
        print(f"\n  Repetition {rep_idx}/{OUTER_CV_REPEATS}  seed={seed}", flush=True)
        outer_cv   = StratifiedKFold(n_splits=OUTER_CV_N_SPLITS,
                                     shuffle=True, random_state=seed)
        base_models = baselines(seed)

        for fold_idx, (tr_idx, te_idx) in enumerate(
                outer_cv.split(X, train_df["group"].values), 1):
            global_fold += 1
            print(f"    Fold {global_fold:02d}  (rep {rep_idx}, fold {fold_idx})",
                  flush=True)

            X_tr, X_te = X[tr_idx], X[te_idx]
            Y_tr, Y_te = Y[tr_idx], Y[te_idx]
            df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
            df_te = train_df.iloc[te_idx].reset_index(drop=True)

            # ---- KernelSVR with Optuna ----
            optuna_seed   = 1000 + global_fold
            inner_cv_seed = 2000 + global_fold
            sampler = optuna.samplers.TPESampler(seed=optuna_seed)
            study   = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(
                lambda t: _internal_optuna_objective(
                    t, X_tr, Y_tr, df_tr["group"].values, inner_cv_seed),
                n_trials=INTERNAL_OPTUNA_TRIALS, n_jobs=1,
                show_progress_bar=False,
            )
            bp = study.best_params
            study.trials_dataframe().to_csv(
                dirs["tuning"] / f"fold_{global_fold:02d}_optuna_trials.csv",
                index=False)

            model, fit_st, fit_wn, fit_s = fit_logged(
                svr_pipeline(float(bp["C"]), float(bp["gamma"]),
                              float(bp["epsilon"]), bp["scaler"]),
                X_tr, Y_tr)

            if model is not None:
                yp, _, _, _ = predict_logged(model, X_te)
                if yp is not None:
                    yp = np.asarray(yp, dtype=float)
                    pred_row = df_te[["subject_id","pet_id","group"]].copy()
                    pred_row["global_outer_fold_id"] = global_fold
                    pred_row["outer_repeat"]         = rep_idx
                    pred_row["outer_fold"]           = fold_idx
                    pred_row["model"]                = "KernelSVR"
                    pred_row["best_params_json"]     = json.dumps(bp, default=_json_default)
                    for i, sc in enumerate(targets):
                        pred_row[f"{sc}_actual"]    = Y_te[:, i]
                        pred_row[f"{sc}_predicted"] = yp[:, i]
                        pred_row[f"{sc}_residual"]  = Y_te[:, i] - yp[:, i]
                    append_df(pred_row, pred_csv)

                    for i, sc in enumerate(targets):
                        m = score_metrics(Y_te[:, i], yp[:, i])
                        append_df(pd.DataFrame([{
                            "global_outer_fold_id": global_fold,
                            "outer_repeat": rep_idx, "outer_fold": fold_idx,
                            "model": "KernelSVR", "score_name": sc,
                            **m, "n_test": len(Y_te),
                        }]), metrics_csv)

            # ---- Baseline models ----
            for mname, mtemplate in base_models.items():
                mdl = clone(mtemplate)
                tgrid = tune_baseline(mname, seed, X_tr, Y_tr,
                                      df_tr["group"].values) \
                        if mname in TUNED_BASELINE_MODELS else None

                if tgrid is not None:
                    tgrid, _, _, _ = fit_logged(tgrid, X_tr, Y_tr)
                    if tgrid is None:
                        continue
                    mdl = tgrid.best_estimator_
                    bp_json = json.dumps(tgrid.best_params_, default=_json_default)
                else:
                    mdl, _, _, _ = fit_logged(mdl, X_tr, Y_tr)
                    if mdl is None:
                        continue
                    bp_json = ""

                yp, _, _, _ = predict_logged(mdl, X_te)
                if yp is None:
                    continue
                yp = np.asarray(yp, dtype=float)

                pred_row = df_te[["subject_id","pet_id","group"]].copy()
                pred_row["global_outer_fold_id"] = global_fold
                pred_row["outer_repeat"]         = rep_idx
                pred_row["outer_fold"]           = fold_idx
                pred_row["model"]                = mname
                pred_row["best_params_json"]     = bp_json
                for i, sc in enumerate(targets):
                    pred_row[f"{sc}_actual"]    = Y_te[:, i]
                    pred_row[f"{sc}_predicted"] = yp[:, i]
                    pred_row[f"{sc}_residual"]  = Y_te[:, i] - yp[:, i]
                append_df(pred_row, pred_csv)

                for i, sc in enumerate(targets):
                    m = score_metrics(Y_te[:, i], yp[:, i])
                    append_df(pd.DataFrame([{
                        "global_outer_fold_id": global_fold,
                        "outer_repeat": rep_idx, "outer_fold": fold_idx,
                        "model": mname, "score_name": sc,
                        **m, "n_test": len(Y_te),
                    }]), metrics_csv)

    # ---- Summary tables ----
    if metrics_csv.exists():
        build_summary_tables(pd.read_csv(metrics_csv), dirs["metrics"])
    export_per_score_csvs(pred_csv, targets, dirs["predictions"])

    # ---- Permutation importance ----
    if RUN_PERMUTATION_IMPORTANCE:
        _run_permutation_importance(train_df, X, Y, targets, dirs["importance"])

    print(f"\nInternal validation done in "
          f"{(time.time()-t0_global)/60:.1f} min", flush=True)
    print(f"Results: {out_dir}", flush=True)


# =============================================================================
# Permutation importance
# =============================================================================
def _run_permutation_importance(
    df: pd.DataFrame, X: np.ndarray, Y: np.ndarray,
    targets: list[str], imp_dir: Path,
) -> None:
    print("\n  [IMPORTANCE] Building balanced 150-subject subset ...", flush=True)
    rng  = np.random.RandomState(IMPORTANCE_SEED)
    idxs = []
    for grp in ["CN", "MCI", "AD"]:
        gi = np.where(df["group"].astype(str).values == grp)[0]
        if len(gi) < IMPORTANCE_PER_GROUP:
            raise ValueError(f"Not enough {grp} subjects for importance subset "
                             f"(need {IMPORTANCE_PER_GROUP}, have {len(gi)}).")
        idxs.extend(rng.choice(gi, IMPORTANCE_PER_GROUP, replace=False).tolist())
    idxs  = np.array(sorted(idxs))
    sub_X = X[idxs]
    sub_Y = Y[idxs]
    sub_g = df["group"].values[idxs]

    iu = np.triu_indices(N_REGIONS, k=1)

    param_grid = [
        {"scaler": [MinMaxScaler(feature_range=(0,1))],
         "svr__C": [5,10,20,30], "svr__gamma": [3e-4,5e-4,0.001,2e-3,3e-3],
         "svr__epsilon": [0.10,0.15,0.20,0.25,0.30]},
        {"scaler": [StandardScaler()],
         "svr__C": [5,10,20,30], "svr__gamma": [3e-4,5e-4,0.001,2e-3,3e-3],
         "svr__epsilon": [0.10,0.15,0.20,0.25,0.30]},
    ]
    inner_cv = list(StratifiedKFold(n_splits=5, shuffle=True,
                                    random_state=IMPORTANCE_SEED).split(sub_X, sub_g))

    for sc_idx, sc in enumerate(targets):
        print(f"  [IMPORTANCE] {sc}", flush=True)
        y_sub = sub_Y[:, sc_idx]
        sc_dir = imp_dir / sc
        ensure_dir(sc_dir)

        grid = GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))]),
            param_grid, scoring="neg_mean_squared_error",
            cv=inner_cv, n_jobs=N_JOBS, refit=True, verbose=0,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                grid.fit(sub_X, y_sub)
                best_model = grid.best_estimator_
            except Exception as e:
                print(f"    tuning failed: {e}", flush=True)
                continue

        perm = permutation_importance(
            best_model, sub_X, y_sub,
            n_repeats=PERMUTATION_REPEATS,
            random_state=IMPORTANCE_SEED,
            n_jobs=N_JOBS,
        )

        rows = []
        for rank, fi in enumerate(np.argsort(perm.importances_mean)[::-1], 1):
            rows.append({
                "rank": rank, "feature_index": int(fi),
                "region_1": int(iu[0][fi]) + 1,
                "region_2": int(iu[1][fi]) + 1,
                "importance_mean": float(perm.importances_mean[fi]),
                "importance_std":  float(perm.importances_std[fi]),
            })
        rank_df = pd.DataFrame(rows)
        rank_df.to_csv(sc_dir / "full_importance_ranking.csv", index=False)
        rank_df.head(TOP_IMPORTANT_EDGES).to_csv(
            sc_dir / f"top_{TOP_IMPORTANT_EDGES}_edges.csv", index=False)
        rank_df.head(TOP_IMPORTANT_EDGES)[
            ["region_1","region_2","importance_mean","importance_std"]
        ].to_csv(sc_dir / f"top_{TOP_IMPORTANT_EDGES}_edges_edge_list.csv",
                 index=False)


# =============================================================================
# B. External validation (with subject-wise normalisation)
# =============================================================================
def run_external_validation(
    train_df:    pd.DataFrame,
    X_train_raw: np.ndarray,
    Y_train:     np.ndarray,
    ext_df:      pd.DataFrame,
    X_ext_raw:   np.ndarray,
    Y_ext:       np.ndarray,
    train_targets: list[str],
    ext_target_map: dict[str, str],
    out_dir:     Path,
) -> None:
    print("\n" + "="*80, flush=True)
    print("EXTERNAL VALIDATION  (subject-wise normalisation)", flush=True)
    print("="*80, flush=True)

    dirs = {k: out_dir/k for k in ["logs","tuning","predictions","metrics"]}
    for d in dirs.values():
        ensure_dir(d)

    # ---- Subject-wise normalisation ----
    print("\n  Applying subject-wise z-score normalisation ...", flush=True)
    X_train = subject_wise_normalize(X_train_raw)
    X_ext   = subject_wise_normalize(X_ext_raw)

    print(f"  Train  — global mean: {X_train.mean():.4f}  std: {X_train.std():.4f}")
    print(f"  Ext    — global mean: {X_ext.mean():.4f}  std: {X_ext.std():.4f}")
    print(f"  Train per-subject mean max |z|: {np.abs(X_train.mean(axis=1)).max():.2e}")
    print(f"  Ext   per-subject mean max |z|: {np.abs(X_ext.mean(axis=1)).max():.2e}")

    save_json({
        "before": {
            "train": {"mean": float(X_train_raw.mean()), "std": float(X_train_raw.std())},
            "ext":   {"mean": float(X_ext_raw.mean()),   "std": float(X_ext_raw.std())},
        },
        "after": {
            "train": {"mean": float(X_train.mean()), "std": float(X_train.std())},
            "ext":   {"mean": float(X_ext.mean()),   "std": float(X_ext.std())},
        },
    }, dirs["logs"] / "normalisation_stats.json")

    # ---- Optuna tuning on normalised training data ----
    logical_targets = list(ext_target_map.keys())
    ti = [train_targets.index(t) for t in logical_targets]
    Y_train_2 = Y_train[:, ti]

    sampler = optuna.samplers.TPESampler(seed=EXTERNAL_OPTUNA_SEED)
    study   = optuna.create_study(direction="minimize", sampler=sampler,
                                   study_name="external_svr_subjectnorm")
    print(f"\n  Optuna tuning ({EXTERNAL_OPTUNA_TRIALS} trials) ...", flush=True)
    t0 = time.time()
    study.optimize(
        lambda t: _external_optuna_objective(
            t, X_train, Y_train_2,
            train_df["group"].values, EXTERNAL_INNER_CV_SEED),
        n_trials=EXTERNAL_OPTUNA_TRIALS, n_jobs=1, show_progress_bar=False,
    )
    tune_s = time.time() - t0
    bp = study.best_params
    study.trials_dataframe().to_csv(dirs["tuning"] / "optuna_trials.csv", index=False)
    save_json({"best_params": bp, "best_value": study.best_value,
               "tuning_seconds": tune_s}, dirs["tuning"] / "optuna_result.json")
    print(f"  Done in {tune_s:.1f}s  |  best MSE={study.best_value:.6f}")
    print(f"  Best params: {bp}")

    # ---- Refit on all training data ----
    best_model, fit_st, fit_wn, fit_s = fit_logged(
        svr_pipeline(float(bp["C"]), float(bp["gamma"]),
                     float(bp["epsilon"]), bp["scaler"]),
        X_train, Y_train_2,
    )
    if best_model is None:
        raise RuntimeError(f"Refit on training data failed: {fit_wn}")
    print(f"  Refit: {fit_s:.1f}s  status={fit_st}")

    # ---- Predict on external data ----
    yp, _, _, pred_s = predict_logged(best_model, X_ext)
    if yp is None:
        raise RuntimeError("Prediction on external data failed.")
    yp = np.asarray(yp, dtype=float)
    print(f"  Prediction done in {pred_s:.1f}s")

    # ---- Metrics ----
    print("\n  Metrics:")
    metrics_rows = []
    for i, lt in enumerate(logical_targets):
        ext_col = ext_target_map[lt]
        m = score_metrics(Y_ext[:, i], yp[:, i])
        metrics_rows.append({
            "score_name": lt, "external_column": ext_col,
            **m, "n_test": len(Y_ext),
            "best_C": float(bp["C"]), "best_gamma": float(bp["gamma"]),
            "best_epsilon": float(bp["epsilon"]), "best_scaler": bp["scaler"],
            "fit_status": fit_st, "tuning_seconds": tune_s,
        })
        print(f"    {lt:<12}  r={m['pearson_r']:+.4f}  "
              f"p={m['pearson_pvalue']:.4f}  "
              f"RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}")

    pd.DataFrame(metrics_rows).to_csv(
        dirs["metrics"] / "external_metrics.csv", index=False)

    # ---- Per-subject predictions CSV ----
    keep = ["subject_id","pet_id"] + [
        c for c in ["group","age","sex","estimated_delta_days","dx_chosen_label"]
        if c in ext_df.columns
    ]
    pred_df = ext_df[keep].copy()
    pred_df["model"]         = "KernelSVR_subjectwise_norm"
    pred_df["normalization"] = "subject_wise_zscore"
    for i, lt in enumerate(logical_targets):
        pred_df[f"{lt}_actual"]    = Y_ext[:, i]
        pred_df[f"{lt}_predicted"] = yp[:, i]
        pred_df[f"{lt}_residual"]  = Y_ext[:, i] - yp[:, i]
    pred_df.to_csv(
        dirs["predictions"] / "external_predictions.csv", index=False)

    print(f"\nExternal validation results: {out_dir}", flush=True)


# =============================================================================
# Main
# =============================================================================
def _validate_config() -> None:
    # Always required
    for name, val in [
        ("TRAIN_CSV",        TRAIN_CSV),
        ("TRAIN_GRAPH_ROOT", TRAIN_GRAPH_ROOT),
        ("RESULTS_ROOT",     RESULTS_ROOT),
    ]:
        if "*** INSERT" in str(val):
            raise ValueError(
                f"Please set {name} in the CONFIGURATION section before running."
            )
    # Only required when external validation is enabled
    if RUN_EXTERNAL_VALIDATION:
        for name, val in [
            ("EXTERNAL_CSV",        EXTERNAL_CSV),
            ("EXTERNAL_GRAPH_ROOT", EXTERNAL_GRAPH_ROOT),
        ]:
            if "*** INSERT" in str(val):
                raise ValueError(
                    f"RUN_EXTERNAL_VALIDATION is True but {name} has not been set. "
                    f"Please fill in the CONFIGURATION section, or set "
                    f"RUN_EXTERNAL_VALIDATION = False to skip external validation."
                )


def main() -> None:
    _validate_config()
    ensure_dir(_RESULTS_ROOT)

    print("="*80, flush=True)
    print("XGML PREDICTION PIPELINE", flush=True)
    print("="*80, flush=True)
    print(f"CPUs detected : {TOTAL_CPUS}   N_JOBS={N_JOBS}", flush=True)
    print(f"Results root  : {_RESULTS_ROOT}", flush=True)

    # ---- Load training data ----
    ensure_dir(_INTERNAL_OUT / "logs")
    train_df, X_train, Y_train = load_dataset(
        _TRAIN_CSV, _TRAIN_GRAPH_ROOT, INTERNAL_TARGETS,
        "TRAIN", require_group=True,
        log_dir=_INTERNAL_OUT / "logs",
    )
    train_df.to_csv(_RESULTS_ROOT / "train_cohort_used.csv", index=False)

    # ---- A. Internal validation ----
    if RUN_INTERNAL_VALIDATION:
        run_internal_validation(
            train_df, X_train, Y_train,
            INTERNAL_TARGETS, _INTERNAL_OUT,
        )

    # ---- B. External validation ----
    if RUN_EXTERNAL_VALIDATION:
        ext_targets = list(EXTERNAL_TARGET_MAP.values())   # columns in EXTERNAL_CSV

        # Check that all internal targets used externally are in INTERNAL_TARGETS
        missing_int = [t for t in EXTERNAL_TARGET_MAP
                       if t not in INTERNAL_TARGETS]
        if missing_int:
            raise ValueError(
                f"EXTERNAL_TARGET_MAP keys not in INTERNAL_TARGETS: {missing_int}"
            )

        ensure_dir(_EXTERNAL_OUT / "logs")
        ext_df, X_ext, Y_ext = load_dataset(
            _EXTERNAL_CSV, _EXTERNAL_GRAPH_ROOT, ext_targets,
            "EXTERNAL", require_group=False,
            log_dir=_EXTERNAL_OUT / "logs",
        )
        ext_df.to_csv(_RESULTS_ROOT / "external_cohort_used.csv", index=False)

        run_external_validation(
            train_df, X_train, Y_train,
            ext_df, X_ext, Y_ext,
            INTERNAL_TARGETS, EXTERNAL_TARGET_MAP,
            _EXTERNAL_OUT,
        )

    print("\n" + "="*80, flush=True)
    print("ALL DONE", flush=True)
    print(f"Results: {_RESULTS_ROOT}", flush=True)
    print("="*80, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise