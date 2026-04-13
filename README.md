# Oil Optimization Webapp

A Flask-based web application for optimizing oil production by predicting output and recommending optimal choke settings using XGBoost machine learning models.

## Features

-  **Single Prediction**: Input daily parameters and get production forecasts
-  **Batch Processing**: Upload Excel files with multiple days of data
-  **Visualization**: D3.js charts comparing actual vs. optimal production
-  **PostgreSQL Database**: Store all predictions for historical analysis
-  **Easy Deployment**: Ready for Railway deployment

## Tech Stack

- **Backend**: Flask (Python)
- **Database**: PostgreSQL + SQLAlchemy ORM
- **Frontend**: Jinja2 Templates + Tailwind CSS + Vanilla JavaScript
- **Visualization**: D3.js
- **ML Models**: XGBoost
- **Deployment**: Railway

## Installation

### Prerequisites
- Python 3.11+
- PostgreSQL
- pip

### Setup

1. **Clone the repository**
```bash
git clone <your-repo>
cd oil-optimization-webapp
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings
```

5. **Initialize database**
```bash
flask db upgrade  # If using Flask-Migrate
# Or SQLAlchemy will auto-create tables on first run
```

6. **Run development server**
```bash
python wsgi.py
```

Visit `http://localhost:5000`

## Database Models

### Prediction
Stores individual prediction results with inputs, outputs, and optimization gains.

```python
- id: Primary Key
- input_data: JSON (8 input features)
- predicted_oil: Float
- predicted_water: Float
- choke_actual: Float
- choke_recommended: Float
- potential_oil_gain: Float
- potential_water_reduction: Float
- prediction_date: Date
- created_at: DateTime
```

### BatchUpload
Tracks batch file processing metadata.

```python
- id: Primary Key
- filename: String
- total_rows: Integer
- processed_rows: Integer
- failed_rows: Integer
- status: String
- uploaded_at: DateTime
- completed_at: DateTime
```

## API Endpoints

### Prediction
- `POST /api/predict`: Single prediction
- `GET /api/health`: Health check

### Batch Upload
Fitur Batch Upload sudah dihapus dari aplikasi.

### History
- `GET /api/history`: Get prediction history
- `GET /api/history/<id>`: Get specific prediction
- `GET /api/history/export`: Export as CSV

## Input Features (8 parameters)

1. **AVG Choke size** (0.1 - 1.0)
2. **DP_CHOKE_SIZE** (psi)
3. **ON_STREAM_HRS** (0 - 24)
4. **AVG_DP_TUBING** (psi)
5. **AVG_DOWNHOLE_PRESSURE** (psi)
6. **AVG_DOWNHOLE_TEMPERATURE** (°F)
7. **AVG_WHP_P** (psi)
8. **AVG_WHT_P** (°F)

## ML Models

Two XGBoost models trained on Volve production data:
- `model_oil_xgb_14.pkl`: Predicts oil production (BORE_OIL_VOL)
- `model_water_xgb_14.pkl`: Predicts water production (BORE_WAT_VOL)

Place models in `ml_models/` directory.

## Optimization Algorithm

Uses `scipy.optimize.minimize` with multi-objective scoring:

```
Objective = Oil Output - (Penalty Weight × Water Output)
Method: Powell optimization
Bounds: Choke size 0.1 - 1.0
```

## Deployment (Railway)

1. Push to GitHub
2. Connect Railway to your repo
3. Set environment variables in Railway:
   - `DATABASE_URL`: PostgreSQL connection string
   - `SECRET_KEY`: Flask secret key
   - `FLASK_ENV`: production

4. Railway auto-detects Flask and deploys!

## Configuration

Edit `app/config.py` to customize:
- Feature bounds and descriptions
- Optimization parameters
- Model paths
- Database URI

## Troubleshooting

### Models not loading
- Ensure `.pkl` files are in `ml_models/` directory
- Check file paths in `config.py`

### Database errors
- Verify PostgreSQL is running
- Check `DATABASE_URL` format
- Ensure database exists

### Import errors
- Run `pip install -r requirements.txt` again
- Check Python version is 3.11+

## Future Enhancements

- [ ] Multiple well support
- [ ] User authentication
- [ ] Real-time monitoring dashboard
- [ ] Model retraining pipeline
- [ ] Advanced analytics (correlation, forecasting)
- [ ] Mobile app

## License

MIT

## Support

Contact your admin for support.
