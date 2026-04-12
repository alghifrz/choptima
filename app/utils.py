import json
import numpy as np
import pandas as pd
import joblib
import re
from scipy.optimize import minimize
from app.config import (
    CHOKE_MIN, CHOKE_MAX, PENALTY_WEIGHT,
    OPTIMIZATION_METHOD, MODEL_PATHS, FEATURE_NAMES
)
import os

# ============================================
# PROJECT ROOT & ML MODEL DISCOVERY
# ============================================
def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


_MODEL_PAIR_CACHE: dict[str, tuple] = {}


def clear_model_cache(stem: str | None = None) -> None:
    """Drop cached joblib models (e.g. after overwriting files on disk)."""
    global _MODEL_PAIR_CACHE
    if stem:
        _MODEL_PAIR_CACHE.pop(stem, None)
    else:
        _MODEL_PAIR_CACHE.clear()


def training_meta_path(stem: str) -> str:
    return os.path.join(_project_root(), "ml_models", f"training_meta_{stem}.json")


def load_training_meta(stem: str) -> dict | None:
    """Load feature column order saved with the model (training pipeline)."""
    p = training_meta_path(stem)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def resolve_feature_order(stem: str | None, model_oil) -> list[str]:
    """
    Sklearn models use positional columns; order must match training.
    Prefer training_meta_{stem}.json; else fall back to FEATURE_NAMES (legacy bundles).
    """
    meta = load_training_meta(stem) if stem else None
    cols = (meta or {}).get("feature_columns")
    if isinstance(cols, list) and cols:
        ni = getattr(model_oil, "n_features_in_", None)
        if ni is not None and len(cols) != ni:
            return list(FEATURE_NAMES)
        return [str(c) for c in cols]
    return list(FEATURE_NAMES)


def _features_array(input_features: dict, feature_order: list[str]) -> np.ndarray:
    try:
        row = [float(input_features[f]) for f in feature_order]
    except KeyError as e:
        raise ValueError(f"Missing feature key: {e}") from e
    return np.array([row], dtype=np.float64)


def default_model_stem_from_config() -> str | None:
    """Stem from MODEL_PATHS['oil'], e.g. model_oil_xgb_14.pkl -> xgb_14."""
    base = os.path.basename(MODEL_PATHS.get("oil", ""))
    if base.startswith("model_oil_") and base.endswith(".pkl"):
        return base[len("model_oil_") : -len(".pkl")]
    return None


def list_model_pairs() -> list[dict[str, str]]:
    """
    Scan ml_models/ for matching model_oil_<stem>.pkl & model_water_<stem>.pkl.
    Returns sorted list of { stem, label }.
    """
    ml_dir = os.path.join(_project_root(), "ml_models")
    if not os.path.isdir(ml_dir):
        return []

    stems: set[str] = set()
    prefix_oil = "model_oil_"
    suffix = ".pkl"
    try:
        for name in os.listdir(ml_dir):
            if not name.startswith(prefix_oil) or not name.endswith(suffix):
                continue
            stem = name[len(prefix_oil) : -len(suffix)]
            if not stem or not re.match(r"^[a-zA-Z0-9_.-]+$", stem):
                continue
            water_name = f"model_water_{stem}{suffix}"
            oil_path = os.path.join(ml_dir, name)
            water_path = os.path.join(ml_dir, water_name)
            if os.path.isfile(oil_path) and os.path.isfile(water_path):
                stems.add(stem)
    except OSError:
        return []

    out = [{"stem": s, "label": s.replace("_", " ")} for s in sorted(stems, key=str.lower)]
    return out


def get_models_for_stem(stem: str):
    """
    Load (oil, water) sklearn models for stem; use cache.
    Returns (model_oil, model_water, error_message). error_message is None on success.
    """
    if not stem or not re.match(r"^[a-zA-Z0-9_.-]{1,200}$", stem):
        return None, None, "Nama model tidak valid."

    pairs = {p["stem"] for p in list_model_pairs()}
    if stem not in pairs:
        return None, None, "Model tidak ditemukan di folder ml_models."

    if stem in _MODEL_PAIR_CACHE:
        mo, mw = _MODEL_PAIR_CACHE[stem]
        return mo, mw, None

    ml_dir = os.path.join(_project_root(), "ml_models")
    oil_path = os.path.join(ml_dir, f"model_oil_{stem}.pkl")
    water_path = os.path.join(ml_dir, f"model_water_{stem}.pkl")
    try:
        mo = joblib.load(oil_path)
        mw = joblib.load(water_path)
        _MODEL_PAIR_CACHE[stem] = (mo, mw)
        return mo, mw, None
    except Exception as e:
        return None, None, f"Gagal memuat model: {e}"


# ============================================
# LOAD MODELS (Global - loaded once)
# ============================================
def load_models():
    """Load XGBoost models for oil and water prediction"""
    try:
        # Get the root directory to construct absolute paths
        root_dir = _project_root()

        # Build absolute paths
        model_oil_path = os.path.join(root_dir, MODEL_PATHS['oil'])
        model_water_path = os.path.join(root_dir, MODEL_PATHS['water'])

        print("Loading models from:")
        print(f"  Oil: {model_oil_path}")
        print(f"  Water: {model_water_path}")

        model_oil = joblib.load(model_oil_path)
        model_water = joblib.load(model_water_path)
        print("[ok] Models loaded successfully")
        return model_oil, model_water
    except Exception as e:
        print(f"[err] Error loading models: {e}")
        return None, None

# Initialize models
MODEL_OIL = None
MODEL_WATER = None

def initialize_models():
    """Initialize global model variables"""
    global MODEL_OIL, MODEL_WATER
    MODEL_OIL, MODEL_WATER = load_models()

# ============================================
# PREDICTION FUNCTIONS
# ============================================
def validate_input(input_dict):
    """
    Validate user input against bounds
    Returns: (is_valid, error_message)
    """
    from app.config import FEATURE_BOUNDS
    
    errors = []
    
    for feature in FEATURE_NAMES:
        if feature not in input_dict:
            errors.append(f"Missing feature: {feature}")
            continue
        
        try:
            value = float(input_dict[feature])
        except (ValueError, TypeError):
            errors.append(f"Invalid value for {feature}: must be numeric")
            continue
        
        min_val, max_val = FEATURE_BOUNDS[feature]
        if not (min_val <= value <= max_val):
            errors.append(
                f"{feature} out of bounds. Expected {min_val}-{max_val}, got {value}"
            )
    
    if errors:
        return False, errors
    
    return True, None


def predict_production(
    input_features,
    model_oil=None,
    model_water=None,
    feature_order: list[str] | None = None,
):
    """
    Predict oil and water production given input features

    Args:
        input_features: dict or list of features (dict keys must match training columns)
        model_oil, model_water: optional; default to global MODEL_OIL / MODEL_WATER
        feature_order: column order used at training time (default FEATURE_NAMES)

    Returns:
        dict with 'oil' and 'water' predictions (guaranteed ≥ 0)
    """
    mo = model_oil if model_oil is not None else MODEL_OIL
    mw = model_water if model_water is not None else MODEL_WATER
    if mo is None or mw is None:
        return None, "Models not loaded"

    order = feature_order if feature_order is not None else list(FEATURE_NAMES)

    try:
        if isinstance(input_features, dict):
            input_array = _features_array(input_features, order)
        else:
            input_array = np.array(input_features).reshape(1, -1)

        oil_pred = mo.predict(input_array)[0]
        water_pred = mw.predict(input_array)[0]
        
        # Ensure non-negative predictions (models might predict negative)
        oil_pred = max(0, float(oil_pred))
        water_pred = max(0, float(water_pred))
        
        return {
            'oil': oil_pred,
            'water': water_pred
        }, None
    
    except Exception as e:
        return None, f"Prediction error: {str(e)}"


def optimize_choke(
    input_features,
    model_oil=None,
    model_water=None,
    feature_order: list[str] | None = None,
):
    """
    Find optimal choke size using multi-objective optimization

    Objective: Maximize (Oil - Penalty * Water)
    With constraint: Oil >= 0, Water >= 0

    Args:
        input_features: dict with features keyed by name (see feature_order)
        model_oil, model_water: optional; default to global MODEL_OIL / MODEL_WATER
        feature_order: column order used at training (default FEATURE_NAMES)

    Returns:
        dict with optimization results
    """
    mo = model_oil if model_oil is not None else MODEL_OIL
    mw = model_water if model_water is not None else MODEL_WATER
    if mo is None or mw is None:
        return None, "Models not loaded"

    order = feature_order if feature_order is not None else list(FEATURE_NAMES)
    choke_key = "AVG Choke size"
    if choke_key not in order:
        return None, "Model tidak punya kolom choke (AVG Choke size) dalam urutan fitur."

    try:
        # Current choke value as initial guess
        choke_actual = input_features.get(choke_key, 0.45)
        x0 = [max(CHOKE_MIN, min(choke_actual, CHOKE_MAX))]

        # Define objective function for optimization
        def objective_function(choke_array):
            choke_value = choke_array[0]

            condition = input_features.copy()
            condition[choke_key] = choke_value

            input_array = _features_array(condition, order)
            pred_oil = mo.predict(input_array)[0]
            pred_water = mw.predict(input_array)[0]
            
            # Ensure non-negative values (models might predict negative)
            pred_oil = max(0, pred_oil)
            pred_water = max(0, pred_water)
            
            # Multi-objective score: maximize oil, minimize water (with penalty)
            score = pred_oil - (PENALTY_WEIGHT * pred_water)
            
            # Return negative because scipy minimizes
            return -score
        
        # Optimize
        bounds = [(CHOKE_MIN, CHOKE_MAX)]
        result = minimize(
            fun=objective_function,
            x0=x0,
            method=OPTIMIZATION_METHOD,
            bounds=bounds
        )
        
        if not result.success:
            return None, f"Optimization failed: {result.message}"
        
        # Get optimal choke value
        choke_optimal = float(result.x[0])
        
        # Predict with optimal choke
        condition_opt = input_features.copy()
        condition_opt[choke_key] = choke_optimal
        input_array = _features_array(condition_opt, order)
        oil_opt = max(0, float(mo.predict(input_array)[0]))
        water_opt = max(0, float(mw.predict(input_array)[0]))

        # Predict with actual choke
        condition_actual = input_features.copy()
        condition_actual[choke_key] = choke_actual
        input_array = _features_array(condition_actual, order)
        oil_actual = max(0, float(mo.predict(input_array)[0]))
        water_actual = max(0, float(mw.predict(input_array)[0]))
        
        return {
            'choke_optimal': choke_optimal,
            'oil_actual': oil_actual,
            'oil_optimal': oil_opt,
            'oil_gain': oil_opt - oil_actual,
            'water_actual': water_actual,
            'water_optimal': water_opt,
            'water_reduction': water_actual - water_opt
        }, None
    
    except Exception as e:
        return None, f"Optimization error: {str(e)}"


def process_excel_file(filepath):
    """
    Process uploaded Excel file
    
    Expected columns (MUST match exactly):
    - AVG Choke size
    - DP_CHOKE_SIZE
    - ON_STREAM_HRS
    - AVG_DP_TUBING
    - AVG_DOWNHOLE_PRESSURE
    - AVG_DOWNHOLE_TEMPERATURE
    - AVG_WHP_P
    - AVG_WHT_P
    
    Returns: list of (success/fail, result_dict)
    """
    try:
        df = pd.read_excel(filepath)
        results = []
        
        # Check if required columns exist
        missing_cols = [col for col in FEATURE_NAMES if col not in df.columns]
        if missing_cols:
            return {
                'error': f'Missing required columns: {", ".join(missing_cols)}. Expected columns: {", ".join(FEATURE_NAMES)}'
            }
        
        # Process each row
        for idx, row in df.iterrows():
            # Convert row to dict
            input_dict = row.to_dict()
            
            # Filter only required features & check for NaN values
            filtered_dict = {}
            has_nan = False
            for feature in FEATURE_NAMES:
                val = input_dict[feature]
                # Check for NaN or None
                if pd.isna(val) or val is None:
                    has_nan = True
                    break
                try:
                    filtered_dict[feature] = float(val)
                except (ValueError, TypeError):
                    has_nan = True
                    break
            
            if has_nan:
                results.append({
                    'row': idx + 2,  # +2 because row 1 is header, idx is 0-based
                    'success': False,
                    'error': 'Row has missing or invalid values (NaN, empty, non-numeric)'
                })
                continue
            
            # Validate input bounds
            is_valid, errors = validate_input(filtered_dict)
            
            if not is_valid:
                results.append({
                    'row': idx + 2,
                    'success': False,
                    'error': ', '.join(errors)
                })
                continue
            
            # Predict
            pred, pred_error = predict_production(filtered_dict)
            if pred_error:
                results.append({
                    'row': idx + 2,
                    'success': False,
                    'error': pred_error
                })
                continue
            
            # Optimize
            opt, opt_error = optimize_choke(filtered_dict)
            if opt_error:
                results.append({
                    'row': idx + 2,
                    'success': False,
                    'error': opt_error
                })
                continue
            
            # Success
            results.append({
                'row': idx + 2,
                'success': True,
                'data': {
                    **filtered_dict,
                    'predicted_oil': pred['oil'],
                    'predicted_water': pred['water'],
                    'choke_recommended': opt['choke_optimal'],
                    'oil_gain': opt['oil_gain'],
                    'water_reduction': opt['water_reduction']
                }
            })
        
        return results
    
    except Exception as e:
        return {'error': f'Excel parsing error: {str(e)}'}
