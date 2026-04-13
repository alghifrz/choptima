from flask import Blueprint, render_template, request, jsonify, send_file
from app import db
from app.models import Prediction, TrainingRun
from app.utils import (
    validate_input,
    predict_production,
    optimize_choke,
    FEATURE_NAMES,
    list_model_pairs,
    get_models_for_stem,
    default_model_stem_from_config,
    resolve_feature_order,
    forecast_tomorrow_from_optimization,
)
from app.config import SELECTED_WELL, FEATURE_DESCRIPTIONS
from datetime import datetime, date
import pandas as pd
from io import BytesIO

bp = Blueprint('main', __name__)

# ============================================
# PAGE ROUTES
# ============================================

@bp.route('/')
def index():
    """Home page - daily optimization"""
    pairs = list_model_pairs()
    stems = [p['stem'] for p in pairs]
    default_stem = default_model_stem_from_config()
    if default_stem not in stems and stems:
        default_stem = stems[0]
    elif not stems:
        default_stem = None
    return render_template(
        'index.html',
        well_name=SELECTED_WELL,
        features=FEATURE_NAMES,
        feature_descriptions=FEATURE_DESCRIPTIONS,
        default_model_stem=default_stem,
    )

@bp.route('/history')
def history():
    """View prediction history"""
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
    return render_template('history.html', predictions=predictions)

# ============================================
# API ROUTES - PREDICTION
# ============================================


@bp.route('/api/ml-models', methods=['GET'])
def api_ml_models():
    """List oil/water model pairs available under ml_models/."""
    return jsonify({'ok': True, 'models': list_model_pairs()}), 200


@bp.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Single prediction endpoint
    
    Expected JSON:
    {
        "choke_size": 0.45,
        "features": { all 8 features },
        "prediction_date": "2024-01-15"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features
        input_features = data.get('features', {})
        prediction_date_str = data.get('prediction_date', date.today().isoformat())
        model_stem = (data.get('model_stem') or data.get('model_id') or '').strip()

        if not model_stem:
            return jsonify({'error': 'Pilih model (model_stem) terlebih dahulu.'}), 400

        mo, mw, model_err = get_models_for_stem(model_stem)
        if model_err:
            return jsonify({'error': model_err}), 400

        feature_order = resolve_feature_order(model_stem, mo)

        # Validate input
        is_valid, errors = validate_input(input_features)
        if not is_valid:
            detail_str = "; ".join(errors) if isinstance(errors, list) else str(errors)
            return jsonify(
                {
                    "error": "Validasi gagal: " + detail_str,
                    "details": errors,
                }
            ), 400

        # Predict production (array columns must match training order)
        pred_result, pred_error = predict_production(
            input_features, mo, mw, feature_order
        )
        if pred_error:
            return jsonify({'error': pred_error}), 500

        # Optimize choke
        opt_result, opt_error = optimize_choke(
            input_features, mo, mw, feature_order
        )
        if opt_error:
            return jsonify({'error': opt_error}), 500

        # Forecast next day (tomorrow) using forecast bundle if available.
        forecast = None
        try:
            forecast = forecast_tomorrow_from_optimization(
                prediction_date_str,
                float(opt_result.get("oil_optimal", pred_result["oil"])),
                float(opt_result.get("water_optimal", pred_result["water"])),
                input_features,
            )
        except Exception:
            forecast = None
        
        # Save to database
        try:
            prediction_date = datetime.strptime(prediction_date_str, '%Y-%m-%d').date()
        except:
            prediction_date = date.today()
        
        prediction_obj = Prediction(
            input_data=input_features,
            predicted_oil=pred_result['oil'],
            predicted_water=pred_result['water'],
            choke_actual=input_features['AVG Choke size'],
            choke_recommended=opt_result['choke_optimal'],
            potential_oil_gain=opt_result['oil_gain'],
            potential_water_reduction=opt_result['water_reduction'],
            prediction_date=prediction_date
        )
        db.session.add(prediction_obj)
        db.session.commit()
        
        # Return results
        return jsonify({
            'success': True,
            'prediction_id': prediction_obj.id,
            'prediction': {
                'oil_actual': round(opt_result['oil_actual'], 2),
                'oil_optimal': round(opt_result['oil_optimal'], 2),
                'oil_gain': round(opt_result['oil_gain'], 2),
                'water_actual': round(opt_result['water_actual'], 2),
                'water_optimal': round(opt_result['water_optimal'], 2),
                'water_reduction': round(opt_result['water_reduction'], 2),
                'choke_actual': round(input_features['AVG Choke size'], 3),
                'choke_recommended': round(opt_result['choke_optimal'], 3)
            },
            'forecast_tomorrow': forecast,
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


# ============================================
# API ROUTES - HISTORY
# ============================================

@bp.route('/api/history', methods=['GET'])
def api_get_history():
    """Get prediction history"""
    try:
        limit = request.args.get('limit', 100, type=int)
        predictions = Prediction.query.order_by(
            Prediction.created_at.desc()
        ).limit(limit).all()
        
        return jsonify({
            'success': True,
            'count': len(predictions),
            'data': [p.to_dict() for p in predictions]
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/train-history', methods=['GET'])
def api_get_train_history():
    """Get train-model history (saved model runs)."""
    try:
        limit = request.args.get('limit', 200, type=int)
        rows = TrainingRun.query.order_by(TrainingRun.created_at.desc()).limit(limit).all()
        return jsonify(
            {
                "success": True,
                "count": len(rows),
                "data": [r.to_dict() for r in rows],
            }
        ), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/train-history/export', methods=['GET'])
def api_export_train_history():
    """Export train-model history as CSV."""
    try:
        rows = TrainingRun.query.order_by(TrainingRun.created_at.desc()).all()
        data = [r.to_dict() for r in rows]
        df = pd.DataFrame(data)

        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='train_model_history.csv',
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/history/<int:prediction_id>', methods=['GET'])
def api_get_prediction(prediction_id):
    """Get single prediction detail"""
    try:
        prediction = Prediction.query.get(prediction_id)
        
        if not prediction:
            return jsonify({'error': 'Prediction not found'}), 404
        
        return jsonify({
            'success': True,
            'data': prediction.to_dict()
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/history/export', methods=['GET'])
def api_export_history():
    """Export prediction history as CSV"""
    try:
        predictions = Prediction.query.all()
        
        # Create DataFrame
        data = [p.to_dict() for p in predictions]
        df = pd.DataFrame(data)
        
        # Create CSV in memory
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='oil_optimization_history.csv'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================
# Health Check
# ============================================

@bp.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'well': SELECTED_WELL,
        'timestamp': datetime.utcnow().isoformat()
    }), 200
