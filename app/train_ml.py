"""ML helpers for the production forecasting web workflow."""

from __future__ import annotations

import pickle
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

WELL_COL = "NPD_WELL_BORE_NAME"
DATE_COL = "DATEPRD"
CHOKE_COL = "AVG Choke size"

PREPROCESS_RULES = {
    "oil_positive": ("BORE_OIL_VOL", lambda s: s > 0),
    "flow_production": ("FLOW_KIND", lambda s: s == "production"),
    "well_op": ("WELL_TYPE", lambda s: s == "OP"),
}

ALGORITHMS = {
    "decisiontree",
    "lightgbm",
    "sgd",
    "svm",
    "xgboost",
    "gradientboosting",
    "ridgeregressor",
    "lassoregressor",
    "extratrees",
    "randomforest",
}

OPT_METHODS = ("SLSQP", "Powell", "COBYLA", "Nelder-Mead", "BFGS", "L-BFGS-B", "TNC", "trust-constr")

BOUNDED_MINIMIZE_METHODS = frozenset(
    {"SLSQP", "L-BFGS-B", "TNC", "Powell", "trust-constr"}
)


def well_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or WELL_COL not in df.columns:
        return []
    vc = df[WELL_COL].value_counts()
    return [{"well": str(k), "count": int(v)} for k, v in vc.items()]


def apply_preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply notebook filters; return (filtered_df, list of skipped rule names)."""
    out = df.copy()
    skipped: list[str] = []
    for name, (col, pred) in PREPROCESS_RULES.items():
        if col not in out.columns:
            skipped.append(name)
            continue
        before = len(out)
        if col == "FLOW_KIND":
            out = out[out[col].astype(str).str.lower() == "production"]
        elif col == "WELL_TYPE":
            out = out[out[col].astype(str) == "OP"]
        else:
            out = out[pred(out[col])]
        if len(out) == before and before > 0:
            pass
    out = out.reset_index(drop=True)
    return out, skipped


def build_regressor(name: str, random_state: int = 42):
    n = name.lower().strip()
    if n not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {name}")
    if n == "decisiontree":
        return DecisionTreeRegressor(random_state=random_state, max_depth=12)
    if n == "randomforest":
        return RandomForestRegressor(
            n_estimators=200, random_state=random_state, n_jobs=-1, max_depth=12
        )
    if n == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=200, random_state=random_state, n_jobs=-1, max_depth=12
        )
    if n == "gradientboosting":
        return GradientBoostingRegressor(random_state=random_state, max_depth=5)
    if n == "ridgeregressor":
        return Ridge(alpha=1.0)
    if n == "lassoregressor":
        return Lasso(random_state=random_state, max_iter=10000)
    if n == "sgd":
        return SGDRegressor(random_state=random_state, max_iter=10000, tol=1e-3)
    if n == "svm":
        return SVR(kernel="rbf", C=1.0, epsilon=0.1)
    if n == "xgboost":
        if xgb is None:
            raise ImportError("xgboost is not installed")
        return xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
        )
    if n == "lightgbm":
        if lgb is None:
            raise ImportError("lightgbm is not installed")
        return lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Unhandled algorithm: {name}")


def _minimize_with_method(fun, x0, bounds, method: str):
    m = method.strip()
    if m == "BFGS":
        m = "L-BFGS-B"
    kwargs = {"fun": fun, "x0": x0, "method": m}
    if m in BOUNDED_MINIMIZE_METHODS:
        kwargs["bounds"] = bounds
    res = minimize(**kwargs)
    x = np.asarray(res.x, dtype=float).ravel()
    if m not in BOUNDED_MINIMIZE_METHODS:
        lo, hi = bounds[0]
        x[0] = float(np.clip(x[0], lo, hi))
    return res, x


def run_choke_optimization(
    model_oil,
    model_water,
    X_subset: pd.DataFrame,
    meta_subset: pd.DataFrame,
    feature_cols: list[str],
    target_oil: str,
    target_water: str,
    method: str,
    choke_col: str = CHOKE_COL,
    bounds: tuple[float, float] = (0.1, 1.0),
) -> list[dict[str, Any]]:
    if choke_col not in feature_cols:
        raise ValueError(
            f"Kolom choke '{choke_col}' harus ada di fitur agar optimasi choke berjalan."
        )
    b = [(bounds[0], bounds[1])]
    rows_out: list[dict[str, Any]] = []

    for i in range(len(X_subset)):
        row = X_subset.iloc[i]
        tanggal = meta_subset.iloc[i].get(DATE_COL, pd.NaT)
        choke_aktual = float(row[choke_col])
        x0 = np.array([choke_aktual], dtype=float)

        def objective_oil(choke_arr):
            sim = row.copy()
            sim[choke_col] = float(choke_arr[0])
            arr = sim[feature_cols].values.astype(np.float64).reshape(1, -1)
            pred = float(model_oil.predict(arr)[0])
            return -pred

        res, x_opt = _minimize_with_method(objective_oil, x0, b, method)
        choke_rekom = float(x_opt[0])

        produksi_sebelum = float(
            model_oil.predict(
                row[feature_cols].values.astype(np.float64).reshape(1, -1)
            )[0]
        )
        produksi_maksimal = float(-res.fun)

        sim_rekom = row.copy()
        sim_rekom[choke_col] = choke_rekom
        arr_rekom = sim_rekom[feature_cols].values.astype(np.float64).reshape(1, -1)
        water_pred_rekom = float(model_water.predict(arr_rekom)[0])
        water_pred_actual_choke = float(
            model_water.predict(
                row[feature_cols].values.astype(np.float64).reshape(1, -1)
            )[0]
        )

        oil_actual = float(meta_subset.iloc[i][target_oil])
        water_actual = float(meta_subset.iloc[i][target_water])

        rows_out.append(
            {
                "DATEPRD": pd.Timestamp(tanggal).isoformat()
                if pd.notna(tanggal)
                else None,
                "Choke_Aktual": choke_aktual,
                "Choke_Rekomendasi": choke_rekom,
                "Oil_Pred_ActualChoke": produksi_sebelum,
                "Oil_Pred_OptimalChoke": produksi_maksimal,
                "Oil_Actual": oil_actual,
                "Water_Pred_ActualChoke": water_pred_actual_choke,
                "Water_Pred_OptimalChoke": water_pred_rekom,
                "Water_Actual": water_actual,
            }
        )
    return rows_out


@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    df_raw: pd.DataFrame | None = None
    df_filtered: pd.DataFrame | None = None
    df_well: pd.DataFrame | None = None
    well_name: str | None = None
    feature_cols: list[str] = field(default_factory=list)
    target_oil: str = "BORE_OIL_VOL"
    target_water: str = "BORE_WAT_VOL"
    test_size: float = 0.2
    random_state: int = 42
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_oil_train: pd.Series | None = None
    y_oil_test: pd.Series | None = None
    y_water_train: pd.Series | None = None
    y_water_test: pd.Series | None = None
    test_meta: pd.DataFrame | None = None
    algorithm: str | None = None
    model_oil: Any = None
    model_water: Any = None
    cv_folds: int = 5
    cv_result_oil: dict | None = None
    cv_result_water: dict | None = None
    cv_r2_oil: dict | None = None
    cv_r2_water: dict | None = None
    test_rmse_oil: float | None = None
    test_rmse_water: float | None = None
    test_r2_oil: float | None = None
    test_r2_water: float | None = None
    optimization_rows: list[dict] | None = None
    preprocess_skipped: list[str] = field(default_factory=list)
    upload_filename: str | None = None


class SessionStore:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path:
        return self.base_dir / f"{sid}.pkl"

    def load(self, sid: str) -> SessionState | None:
        p = self._path(sid)
        if not p.is_file():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def save(self, state: SessionState) -> None:
        with open(self._path(state.session_id), "wb") as f:
            pickle.dump(state, f)


def split_aligned(
    df_well: pd.DataFrame,
    feature_cols: list[str],
    target_oil: str,
    target_water: str,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    meta_cols = [c for c in [DATE_COL, WELL_COL] if c in df_well.columns]
    need = feature_cols + [target_oil, target_water] + meta_cols
    missing = [c for c in need if c not in df_well.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    X = df_well[feature_cols].copy()
    y_oil = df_well[target_oil].copy()
    y_water = df_well[target_water].copy()
    meta = df_well[meta_cols + [target_oil, target_water]].copy()
    if CHOKE_COL in df_well.columns and CHOKE_COL not in meta.columns:
        meta[CHOKE_COL] = df_well[CHOKE_COL].values

    idx = np.arange(len(X))
    train_i, test_i = train_test_split(
        idx, test_size=test_size, random_state=random_state
    )
    return {
        "X_train": X.iloc[train_i].reset_index(drop=True),
        "X_test": X.iloc[test_i].reset_index(drop=True),
        "y_oil_train": y_oil.iloc[train_i].reset_index(drop=True),
        "y_oil_test": y_oil.iloc[test_i].reset_index(drop=True),
        "y_water_train": y_water.iloc[train_i].reset_index(drop=True),
        "y_water_test": y_water.iloc[test_i].reset_index(drop=True),
        "test_meta": meta.iloc[test_i].reset_index(drop=True),
    }


def test_date_bounds(
    test_meta: pd.DataFrame | None, date_col: str = DATE_COL
) -> tuple[str | None, str | None]:
    if test_meta is None or date_col not in test_meta.columns:
        return None, None
    s = pd.to_datetime(test_meta[date_col], errors="coerce").dropna()
    if s.empty:
        return None, None
    return s.min().date().isoformat(), s.max().date().isoformat()


def select_optimization_subset(
    X_test: pd.DataFrame,
    test_meta: pd.DataFrame,
    n_days: int,
    start_date: str | None,
    end_date: str | None,
    date_col: str = DATE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Pilih baris test untuk optimasi.

    - Tanpa rentang tanggal: ambil ``n_days`` baris pertama (perilaku lama / notebook).
    - Dengan rentang: filter ``date_col`` di [start, end], urut naik tanggal, lalu ambil
      paling banyak ``n_days`` baris.
    """
    if len(X_test) != len(test_meta):
        raise ValueError("X_test dan test_meta tidak sejajar.")
    n_days = int(n_days)
    if n_days < 1:
        raise ValueError("Jumlah hari minimal 1.")

    s_raw = (start_date or "").strip() if start_date is not None else ""
    e_raw = (end_date or "").strip() if end_date is not None else ""
    has_s = bool(s_raw)
    has_e = bool(e_raw)

    if has_s ^ has_e:
        raise ValueError(
            "Isi tanggal mulai dan tanggal akhir keduanya, atau kosongkan keduanya."
        )

    if not has_s:
        n = min(n_days, len(X_test))
        Xs = X_test.iloc[:n].reset_index(drop=True)
        ms = test_meta.iloc[:n].reset_index(drop=True)
        return Xs, ms, {
            "mode": "head",
            "n_rows": n,
            "date_filter": None,
        }

    if date_col not in test_meta.columns:
        raise ValueError(
            f"Kolom '{date_col}' tidak ada di metadata test; tidak bisa filter tanggal."
        )

    d_start = pd.Timestamp(s_raw).normalize()
    d_end = pd.Timestamp(e_raw).normalize()
    if d_start > d_end:
        raise ValueError("Tanggal mulai tidak boleh setelah tanggal akhir.")

    dates = pd.to_datetime(test_meta[date_col], errors="coerce")
    norm = dates.dt.normalize()
    mask = dates.notna() & (norm >= d_start) & (norm <= d_end)
    idx = np.flatnonzero(mask.to_numpy())
    if idx.size == 0:
        raise ValueError("Tidak ada baris test dalam rentang tanggal yang dipilih.")

    sub_X = X_test.iloc[idx].reset_index(drop=True)
    sub_m = test_meta.iloc[idx].reset_index(drop=True)
    sub_d = dates.iloc[idx].reset_index(drop=True)
    order = np.argsort(sub_d.to_numpy(), kind="mergesort")
    sub_X = sub_X.iloc[order].reset_index(drop=True)
    sub_m = sub_m.iloc[order].reset_index(drop=True)

    n_take = min(n_days, len(sub_X))
    sub_X = sub_X.iloc[:n_take].reset_index(drop=True)
    sub_m = sub_m.iloc[:n_take].reset_index(drop=True)

    return sub_X, sub_m, {
        "mode": "date_range",
        "n_rows": n_take,
        "date_filter": {
            "start": d_start.date().isoformat(),
            "end": d_end.date().isoformat(),
        },
        "matched_in_range": int(idx.size),
    }


def filter_well(df: pd.DataFrame, well_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Data kosong.")
    if WELL_COL not in df.columns:
        raise ValueError(f"Kolom '{WELL_COL}' tidak ada di dataset.")
    mask = df[WELL_COL].astype(str) == str(well_name)
    out = df.loc[mask].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"Tidak ada baris untuk sumur '{well_name}'.")
    return out


def coerce_numeric_well(df_well: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Coerce feature columns to numeric; rows with NaN in features/targets are dropped later."""
    out = df_well.copy()
    for c in feature_cols:
        if c not in out.columns:
            raise ValueError(f"Kolom fitur '{c}' tidak ada.")
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def apply_train_split(state: SessionState) -> dict[str, Any]:
    """Run split_aligned after df_well and feature_cols are set; mutates state."""
    if state.df_well is None:
        raise ValueError("Pilih sumur dulu.")
    if not state.feature_cols:
        raise ValueError("Pilih minimal satu kolom fitur.")
    df = coerce_numeric_well(state.df_well, state.feature_cols)
    for t in (state.target_oil, state.target_water):
        if t not in df.columns:
            raise ValueError(f"Kolom target '{t}' tidak ada.")
        df[t] = pd.to_numeric(df[t], errors="coerce")
    need_cols = state.feature_cols + [state.target_oil, state.target_water]
    df = df.dropna(subset=need_cols).reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Baris valid setelah cleaning terlalu sedikit (min ~10).")

    state.df_well = df
    sp = split_aligned(
        df,
        state.feature_cols,
        state.target_oil,
        state.target_water,
        state.test_size,
        state.random_state,
    )
    state.X_train = sp["X_train"]
    state.X_test = sp["X_test"]
    state.y_oil_train = sp["y_oil_train"]
    state.y_oil_test = sp["y_oil_test"]
    state.y_water_train = sp["y_water_train"]
    state.y_water_test = sp["y_water_test"]
    state.test_meta = sp["test_meta"]
    state.model_oil = None
    state.model_water = None
    state.cv_result_oil = None
    state.cv_result_water = None
    state.cv_r2_oil = None
    state.cv_r2_water = None
    state.test_rmse_oil = None
    state.test_rmse_water = None
    state.test_r2_oil = None
    state.test_r2_water = None
    state.optimization_rows = None
    dmin, dmax = test_date_bounds(state.test_meta)
    return {
        "train_rows": len(state.X_train),
        "test_rows": len(state.X_test),
        "test_date_min": dmin,
        "test_date_max": dmax,
    }


def _cv_scores_payload(scores: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=float)
    return {
        "scores": [float(x) for x in scores],
        "mean": float(scores.mean()),
        "std": float(scores.std()) if len(scores) > 1 else 0.0,
    }


def run_cross_validate_r2(state: SessionState, folds: int) -> dict[str, Any]:
    """Re-run R² cross-validation on fitted models."""
    if state.model_oil is None or state.model_water is None or state.X_train is None:
        raise ValueError("Latih model terlebih dahulu.")
    folds = max(2, min(20, int(folds)))
    state.cv_folds = folds
    Xtr = state.X_train.astype(float)
    s_oil = cross_val_score(
        state.model_oil,
        Xtr,
        state.y_oil_train,
        cv=folds,
        scoring="r2",
        n_jobs=-1,
    )
    s_water = cross_val_score(
        state.model_water,
        Xtr,
        state.y_water_train,
        cv=folds,
        scoring="r2",
        n_jobs=-1,
    )
    state.cv_r2_oil = _cv_scores_payload(s_oil)
    state.cv_r2_water = _cv_scores_payload(s_water)
    return {
        "cross_validation_oil": state.cv_r2_oil,
        "cross_validation_water": state.cv_r2_water,
    }


def run_training(state: SessionState) -> dict[str, Any]:
    """Fit oil/water regressors, CV on train, test metrics."""
    if state.X_train is None or state.X_test is None:
        raise ValueError("Lakukan split data terlebih dahulu.")
    if not state.algorithm:
        raise ValueError("Pilih algoritma.")

    oil = build_regressor(state.algorithm, state.random_state)
    water = build_regressor(state.algorithm, state.random_state)

    cv_oil = cross_val_score(
        oil,
        state.X_train,
        state.y_oil_train,
        cv=state.cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    cv_water = cross_val_score(
        water,
        state.X_train,
        state.y_water_train,
        cv=state.cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    oil.fit(state.X_train, state.y_oil_train)
    water.fit(state.X_train, state.y_water_train)
    state.model_oil = oil
    state.model_water = water

    s_oil_r2 = cross_val_score(
        oil,
        state.X_train,
        state.y_oil_train,
        cv=state.cv_folds,
        scoring="r2",
        n_jobs=-1,
    )
    s_water_r2 = cross_val_score(
        water,
        state.X_train,
        state.y_water_train,
        cv=state.cv_folds,
        scoring="r2",
        n_jobs=-1,
    )
    state.cv_r2_oil = _cv_scores_payload(s_oil_r2)
    state.cv_r2_water = _cv_scores_payload(s_water_r2)

    state.cv_result_oil = {
        "mean_neg_mse": float(np.mean(cv_oil)),
        "std_neg_mse": float(np.std(cv_oil)),
        "rmse_cv": float(np.sqrt(-float(np.mean(cv_oil)))),
    }
    state.cv_result_water = {
        "mean_neg_mse": float(np.mean(cv_water)),
        "std_neg_mse": float(np.std(cv_water)),
        "rmse_cv": float(np.sqrt(-float(np.mean(cv_water)))),
    }

    po = oil.predict(state.X_test)
    pw = water.predict(state.X_test)
    yo = state.y_oil_test.to_numpy(dtype=float)
    yw = state.y_water_test.to_numpy(dtype=float)

    max_viz = 500
    n_test = len(po)
    if n_test <= max_viz:
        idx = np.arange(n_test)
    else:
        idx = np.unique(
            np.linspace(0, n_test - 1, num=max_viz, dtype=int)
        )

    dates_out: list[str | None] | None = None
    if state.test_meta is not None and DATE_COL in state.test_meta.columns:
        dt = pd.to_datetime(state.test_meta[DATE_COL], errors="coerce")
        dates_out = []
        for i in idx:
            t = dt.iloc[int(i)]
            if pd.notna(t):
                dates_out.append(pd.Timestamp(t).isoformat())
            else:
                dates_out.append(None)

    visualization = {
        "oil": {
            "actual": [float(yo[i]) for i in idx],
            "predicted": [float(po[i]) for i in idx],
        },
        "water": {
            "actual": [float(yw[i]) for i in idx],
            "predicted": [float(pw[i]) for i in idx],
        },
        "dates": dates_out,
        "target_oil": state.target_oil,
        "target_water": state.target_water,
        "sampled_points": int(len(idx)),
        "total_test_points": int(n_test),
    }

    tr_oil = float(np.sqrt(mean_squared_error(yo, po)))
    tr_wa = float(np.sqrt(mean_squared_error(yw, pw)))
    r2_oil = float(r2_score(yo, po))
    r2_wa = float(r2_score(yw, pw))
    state.test_rmse_oil = tr_oil
    state.test_rmse_water = tr_wa
    state.test_r2_oil = r2_oil
    state.test_r2_water = r2_wa

    return {
        "test_rmse_oil": tr_oil,
        "test_rmse_water": tr_wa,
        "test_r2_oil": r2_oil,
        "test_r2_water": r2_wa,
        "cross_validation_oil": state.cv_r2_oil,
        "cross_validation_water": state.cv_r2_water,
        "visualization": visualization,
    }


def run_optimization_step(
    state: SessionState,
    opt_method: str,
    n_days: int,
    start_date: str | None,
    end_date: str | None,
    choke_bounds: tuple[float, float] = (0.1, 1.0),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if state.model_oil is None or state.model_water is None:
        raise ValueError("Latih model dulu.")
    if state.X_test is None or state.test_meta is None:
        raise ValueError("Data test tidak ada.")
    Xs, ms, meta = select_optimization_subset(
        state.X_test,
        state.test_meta,
        n_days,
        start_date,
        end_date,
    )
    rows = run_choke_optimization(
        state.model_oil,
        state.model_water,
        Xs,
        ms,
        state.feature_cols,
        state.target_oil,
        state.target_water,
        opt_method,
        CHOKE_COL,
        choke_bounds,
    )
    state.optimization_rows = rows
    return rows, meta
