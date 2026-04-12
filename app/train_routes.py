"""Flask routes for the Train Model workflow (session → upload → … → optimize)."""

import json
import re
from pathlib import Path

import joblib
import pandas as pd
from flask import Blueprint, current_app, jsonify, render_template, request
from werkzeug.utils import secure_filename

from app.config import CHOKE_MAX, CHOKE_MIN, FEATURE_NAMES
from app.utils import clear_model_cache
from app.train_ml import (
    ALGORITHMS,
    CHOKE_COL,
    OPT_METHODS,
    SessionState,
    SessionStore,
    apply_preprocess,
    apply_train_split,
    filter_well,
    run_cross_validate_r2,
    run_optimization_step,
    run_training,
    test_date_bounds,
    well_summary,
)

train_bp = Blueprint("train", __name__)


def _store() -> SessionStore:
    return SessionStore(Path(current_app.config["TRAIN_SESSION_DIR"]))


def _load_state(sid: str) -> SessionState | None:
    return _store().load(sid)


def _save_state(state: SessionState) -> None:
    _store().save(state)


def session_id_from_request() -> str | None:
    return (
        request.headers.get("X-Session-Id")
        or request.args.get("session_id")
        or request.form.get("session_id")
        or (request.get_json(silent=True) or {}).get("session_id")
    )


def _json_err(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


def _safe_model_stem(name: str) -> str:
    """User-facing model name → safe filename stem (no path segments)."""
    raw = (name or "").strip()
    if not raw:
        raise ValueError("Nama model wajib diisi.")
    stem = secure_filename(raw.replace(" ", "_"))
    if not stem:
        stem = re.sub(r"[^\w\-.]", "_", raw, flags=re.UNICODE).strip("._")
    if not stem or stem in (".", ".."):
        raise ValueError("Nama model tidak valid.")
    return stem[:120]


# --- Pages ---


@train_bp.route("/train-model")
def train_model_page():
    return render_template(
        "train_model.html",
        algorithms=sorted(ALGORITHMS),
        opt_methods=OPT_METHODS,
        default_features=FEATURE_NAMES,
    )


# --- API: session ---


@train_bp.route("/api/train/session", methods=["POST"])
def api_create_session():
    st = SessionState()
    _save_state(st)
    return jsonify({"ok": True, "session_id": st.session_id})


# --- API: upload (wajib sesi) ---


@train_bp.route("/api/train/upload", methods=["POST"])
def api_train_upload():
    sid = session_id_from_request()
    if not sid:
        return _json_err("Header X-Session-Id, form session_id, atau JSON session_id wajib.")
    state = _load_state(sid)
    if not state:
        return _json_err("Sesi tidak ditemukan. Buat sesi baru dulu.", 404)

    if "file" not in request.files:
        return _json_err("Tidak ada file")
    f = request.files["file"]
    if not f or not f.filename:
        return _json_err("File kosong")
    name = f.filename.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(f)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(f)
        else:
            return _json_err("Gunakan CSV atau Excel (.xlsx, .xls)")
    except Exception as e:
        return _json_err(f"Gagal membaca file: {e}")

    state.df_raw = df
    state.upload_filename = f.filename
    state.df_filtered = None
    state.df_well = None
    state.model_oil = None
    state.model_water = None
    state.optimization_rows = None
    _save_state(state)

    summary = well_summary(df)
    return jsonify(
        {
            "ok": True,
            "session_id": state.session_id,
            "filename": f.filename,
            "n_rows": len(df),
            "columns": list(df.columns.astype(str)),
            "n_wells": len(summary),
            "wells": summary,
        }
    )


@train_bp.route("/api/train/preprocess", methods=["POST"])
def api_train_preprocess():
    sid = session_id_from_request()
    if not sid:
        return _json_err("session_id wajib")
    state = _load_state(sid)
    if not state or state.df_raw is None:
        return _json_err("Unggah data terlebih dahulu atau sesi tidak valid.", 404)

    df_f, skipped = apply_preprocess(state.df_raw)
    state.df_filtered = df_f
    state.preprocess_skipped = skipped
    state.df_well = None
    _save_state(state)
    summary = well_summary(df_f)
    return jsonify(
        {
            "ok": True,
            "n_rows": len(df_f),
            "n_wells": len(summary),
            "wells": summary,
            "skipped_rules_missing_columns": skipped,
        }
    )


@train_bp.route("/api/train/select-well", methods=["POST"])
def api_train_select_well():
    data = request.get_json(silent=True) or {}
    sid = session_id_from_request() or data.get("session_id")
    well_name = data.get("well_name")
    if not sid or not well_name:
        return _json_err("session_id dan well_name wajib")
    state = _load_state(sid)
    if not state:
        return _json_err("Sesi tidak ditemukan", 404)
    base = state.df_filtered if state.df_filtered is not None else state.df_raw
    if base is None:
        return _json_err("Unggah dan praproses data terlebih dahulu.")
    try:
        df_w = filter_well(base, well_name)
    except ValueError as e:
        return _json_err(str(e))
    state.df_well = df_w
    state.well_name = str(well_name)
    state.model_oil = None
    state.model_water = None
    state.optimization_rows = None
    _save_state(state)
    return jsonify(
        {
            "ok": True,
            "well_name": state.well_name,
            "n_rows": len(df_w),
            "columns": list(df_w.columns.astype(str)),
        }
    )


@train_bp.route("/api/train/features", methods=["POST"])
def api_train_set_features():
    data = request.get_json(silent=True) or {}
    sid = session_id_from_request() or data.get("session_id")
    features = data.get("feature_columns")
    t_oil = data.get("target_oil", "BORE_OIL_VOL")
    t_water = data.get("target_water", "BORE_WAT_VOL")
    if not sid:
        return _json_err("session_id wajib")
    if not features or not isinstance(features, list):
        return _json_err("feature_columns (array) wajib.")
    state = _load_state(sid)
    if not state or state.df_well is None:
        return _json_err("Pilih sumur terlebih dahulu.")
    cols = list(state.df_well.columns.astype(str))
    for c in list(features) + [t_oil, t_water]:
        if c not in cols:
            return _json_err(f"Kolom tidak ada di data sumur: {c}")
    state.feature_cols = [str(x) for x in features]
    state.target_oil = str(t_oil)
    state.target_water = str(t_water)
    _save_state(state)
    return jsonify(
        {
            "ok": True,
            "feature_columns": state.feature_cols,
            "target_oil": state.target_oil,
            "target_water": state.target_water,
        }
    )


@train_bp.route("/api/train/split", methods=["POST"])
def api_train_split():
    data = request.get_json(silent=True) or {}
    sid = session_id_from_request() or data.get("session_id")
    test_size = float(data.get("test_size", 0.2))
    random_state = int(data.get("random_state", 42))
    cv_folds = int(data.get("cv_folds", 5))
    if not sid:
        return _json_err("session_id wajib")
    if not 0 < test_size < 1:
        return _json_err("test_size harus antara 0 dan 1.")
    state = _load_state(sid)
    if not state or state.df_well is None:
        return _json_err("Pilih sumur dulu")
    if isinstance(data.get("feature_cols"), list) and data["feature_cols"]:
        state.feature_cols = [str(x) for x in data["feature_cols"]]
    if data.get("target_oil"):
        state.target_oil = str(data["target_oil"])
    if data.get("target_water"):
        state.target_water = str(data["target_water"])
    if not state.feature_cols:
        return _json_err("Atur fitur dulu (tombol Pilih kolom) atau kirim feature_cols.")
    state.test_size = test_size
    state.random_state = random_state
    state.cv_folds = max(2, min(10, cv_folds))
    try:
        summary = apply_train_split(state)
    except ValueError as e:
        return _json_err(str(e))
    _save_state(state)
    return jsonify({"ok": True, **summary, "test_size": test_size, "random_state": random_state})


@train_bp.route("/api/train/model", methods=["POST"])
def api_train_set_model():
    data = request.get_json(silent=True) or {}
    sid = session_id_from_request() or data.get("session_id")
    algo = (data.get("algorithm") or "").lower().strip()
    if not sid:
        return _json_err("session_id wajib")
    if algo not in ALGORITHMS:
        return _json_err(f"algorithm tidak valid: {algo}")
    state = _load_state(sid)
    if not state:
        return _json_err("Sesi tidak ditemukan", 404)
    state.algorithm = algo
    _save_state(state)
    return jsonify({"ok": True, "algorithm": state.algorithm})


def _api_train_fit_impl():
    data = request.get_json(silent=True) or {}
    sid = session_id_from_request() or data.get("session_id")
    algorithm = (data.get("algorithm") or "").lower().strip()
    if not sid:
        return _json_err("session_id wajib")
    state = _load_state(sid)
    if not state:
        return _json_err("Sesi tidak ditemukan", 404)
    if algorithm:
        if algorithm not in ALGORITHMS:
            return _json_err(f"Algoritma tidak dikenal: {algorithm}")
        state.algorithm = algorithm
    elif not state.algorithm:
        return _json_err("Pilih algoritma terlebih dahulu.")
    try:
        metrics = run_training(state)
    except ImportError as e:
        return _json_err(str(e))
    except Exception as e:
        return _json_err(str(e), 500)
    _save_state(state)
    out = {"ok": True, "session_id": sid, "algorithm": state.algorithm, **metrics}
    return jsonify(out)


@train_bp.route("/api/train/fit", methods=["POST"])
@train_bp.route("/api/train/train", methods=["POST"])
def api_train_fit():
    return _api_train_fit_impl()


@train_bp.route("/api/train/cross-validate", methods=["POST"])
def api_train_cross_validate():
    data = request.get_json(silent=True) or {}
    sid = session_id_from_request() or data.get("session_id")
    folds = int(data.get("cv", 5))
    if not sid:
        return _json_err("session_id wajib")
    state = _load_state(sid)
    if not state:
        return _json_err("Sesi tidak ditemukan", 404)
    try:
        payload = run_cross_validate_r2(state, folds)
    except ValueError as e:
        return _json_err(str(e))
    except Exception as e:
        return _json_err(str(e), 500)
    _save_state(state)
    return jsonify({"ok": True, **payload})


@train_bp.route("/api/train/optimize", methods=["POST"])
def api_train_optimize():
    data = request.get_json(silent=True) or {}
    sid = session_id_from_request() or data.get("session_id")
    opt_method = (data.get("method") or "Powell").strip()
    n_days = int(data.get("n_days", 30))
    start_date = data.get("start_date") or None
    end_date = data.get("end_date") or None

    if not sid:
        return _json_err("session_id wajib")
    if opt_method not in OPT_METHODS:
        return _json_err(f"Metode optimasi tidak didukung: {opt_method}")
    state = _load_state(sid)
    if not state:
        return _json_err("Sesi tidak ditemukan", 404)
    if CHOKE_COL not in (state.feature_cols or []):
        return _json_err(
            f"Untuk optimasi choke, kolom '{CHOKE_COL}' harus dipilih sebagai fitur."
        )
    try:
        rows, meta = run_optimization_step(
            state,
            opt_method,
            n_days,
            start_date,
            end_date,
            (float(CHOKE_MIN), float(CHOKE_MAX)),
        )
    except ValueError as e:
        return _json_err(str(e))
    except Exception as e:
        return _json_err(str(e), 500)
    _save_state(state)
    return jsonify(
        {
            "ok": True,
            "n_rows": len(rows),
            "method": opt_method,
            "subset": meta,
            "rows": rows,
        }
    )


@train_bp.route("/api/train/visualization", methods=["GET"])
def api_train_visualization():
    sid = session_id_from_request()
    if not sid:
        return _json_err("session_id wajib (query atau header)")
    state = _load_state(sid)
    if not state:
        return _json_err("Sesi tidak ditemukan", 404)
    if not state.optimization_rows:
        return _json_err("Jalankan optimasi terlebih dahulu.", 400)
    return jsonify({"ok": True, "series": state.optimization_rows})


@train_bp.route("/api/train/state", methods=["GET"])
def api_train_state():
    sid = session_id_from_request()
    if not sid:
        return _json_err("session_id wajib")
    state = _load_state(sid)
    if not state:
        return _json_err("Sesi tidak ditemukan", 404)
    td_min, td_max = (None, None)
    if state.test_meta is not None:
        td_min, td_max = test_date_bounds(state.test_meta)
    metrics = None
    if state.model_oil is not None:
        cv_o = state.cv_r2_oil
        cv_w = state.cv_r2_water
        metrics = {
            "algorithm": state.algorithm,
            "test_r2_oil": getattr(state, "test_r2_oil", None),
            "test_r2_water": getattr(state, "test_r2_water", None),
            "test_rmse_oil": getattr(state, "test_rmse_oil", None),
            "test_rmse_water": getattr(state, "test_rmse_water", None),
            "cv_r2_mean_oil": (cv_o or {}).get("mean"),
            "cv_r2_std_oil": (cv_o or {}).get("std"),
            "cv_r2_mean_water": (cv_w or {}).get("mean"),
            "cv_r2_std_water": (cv_w or {}).get("std"),
            "cv_folds": state.cv_folds,
        }

    return jsonify(
        {
            "ok": True,
            "upload_filename": state.upload_filename,
            "well_name": state.well_name,
            "has_raw": state.df_raw is not None,
            "has_filtered": state.df_filtered is not None,
            "has_well": state.df_well is not None,
            "feature_columns": state.feature_cols,
            "target_oil": state.target_oil,
            "target_water": state.target_water,
            "algorithm": state.algorithm,
            "test_size": state.test_size,
            "cv_folds": state.cv_folds,
            "has_split": state.X_train is not None,
            "has_models": state.model_oil is not None,
            "has_optimization": bool(state.optimization_rows),
            "test_date_min": td_min,
            "test_date_max": td_max,
            "metrics": metrics,
        }
    )


@train_bp.route("/api/train/save-models", methods=["POST"])
def api_train_save_models():
    data = request.get_json(silent=True) or {}
    sid = session_id_from_request() or data.get("session_id")
    if not sid:
        return _json_err("session_id wajib")
    state = _load_state(sid)
    if not state or state.model_oil is None or state.model_water is None:
        return _json_err("Belum ada model terlatih")
    if not state.optimization_rows:
        return _json_err("Jalankan optimasi dan tampilkan visualisasi terlebih dahulu.")

    model_name = data.get("model_name") or data.get("suffix")
    try:
        stem = _safe_model_stem(str(model_name) if model_name is not None else "")
    except ValueError as e:
        return _json_err(str(e))

    root = Path(current_app.root_path).parent
    ml_dir = root / "ml_models"
    ml_dir.mkdir(parents=True, exist_ok=True)

    oil_path = ml_dir / f"model_oil_{stem}.pkl"
    water_path = ml_dir / f"model_water_{stem}.pkl"

    joblib.dump(state.model_oil, oil_path)
    joblib.dump(state.model_water, water_path)

    meta_path = ml_dir / f"training_meta_{stem}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": list(state.feature_cols or []),
                "target_oil": state.target_oil,
                "target_water": state.target_water,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    clear_model_cache(stem)

    return jsonify(
        {
            "ok": True,
            "session_id": sid,
            "model_name": stem,
            "oil_path": str(oil_path.relative_to(root)),
            "water_path": str(water_path.relative_to(root)),
            "meta_path": str(meta_path.relative_to(root)),
        }
    )
