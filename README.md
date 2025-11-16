# AI-Based Natural Disaster Prediction Web App

An AI-powered web application for predicting natural disasters (specifically floods) in Pakistan using historical weather data and machine learning.

## Overview

This project collects and processes weather data from multiple sources to train machine learning models for flood prediction in high-risk districts of Pakistan (Swat and Upper Dir).

## Features

- **Multi-Source Weather Data Collection**
  - Meteostat API integration for historical weather data
  - NASA POWER API integration for satellite-derived meteorological data
  - Automatic data merging to fill missing values

- **Comprehensive Weather Features**
  - Temperature (average, min, max)
  - Precipitation and snowfall
  - Wind speed and gusts
  - Atmospheric pressure
  - Humidity (from NASA POWER)
  - Solar radiation (from NASA POWER)

- **Advanced Data Processing Pipeline**
  - Automated data fetching and preprocessing
  - Missing value imputation using NASA POWER data
  - Feature engineering (19 engineered features)
  - Data validation and quality checks
  - StandardScaler normalization

- **Machine Learning Models** ✅ **COMPLETED**
  - **Logistic Regression** (Accuracy: 99.91%, AUC-ROC: 0.8243)
  - **Random Forest** (Accuracy: 99.91%, AUC-ROC: 0.8643) ⭐ Best Model
  - Cross-validation & evaluation
  - Model serialization (.pkl files)

- **Model Evaluation & Visualization**
  - Performance metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
  - ROC curves with AUC scores
  - Confusion matrices
  - Feature importance rankings
  - Comprehensive evaluation reports

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zohaibkhan745/-AI-Based-Natural-Disaster-Prediction-Web-App-.git
cd -AI-Based-Natural-Disaster-Prediction-Web-App-
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start - Complete ML Pipeline

Run the entire data preprocessing + model training + evaluation pipeline:
```bash
python3 run_pipeline.py
```

This executes:
1. ✅ Data Preprocessing (feature engineering, scaling)
2. ✅ Model Training (Logistic Regression + Random Forest)
3. ✅ Model Evaluation (metrics, visualizations)

### Test Model Predictions

Verify that your models are working and making predictions:
```bash
python3 test_model.py
```

Comprehensive model prediction verification:
```bash
python3 verify_predictions.py
```

### Step-by-Step Data Collection (Legacy)

1. **Fetch Meteostat Data:**
```bash
python -m code.fetch_meteostat_weather --combine
```

2. **Fetch NASA POWER Data:**
```bash
python -m code.fetch_nasa_power --combine
```

3. **Merge Datasets:**
```bash
python -m code.merge_weather_data
```

### Run Individual Components

**Data Preprocessing Only:**
```bash
python3 code/preprocessing.py
```

**Model Training Only:**
```bash
python3 code/baseline_models.py
```

**Model Evaluation Only:**
```bash
python3 code/model_evaluation.py
```

### Custom Date Ranges

Fetch data for specific time periods:
```bash
python -m code.fetch_meteostat_weather --start-date 2020-01-01 --end-date 2020-12-31 --combine
python -m code.fetch_nasa_power --start-date 2020-01-01 --end-date 2020-12-31 --combine
python -m code.merge_weather_data
```

### Single Location

Fetch data for a specific location:
```bash
python -m code.fetch_meteostat_weather --locations swat --combine
python -m code.fetch_nasa_power --locations swat --combine
```

## Data Sources

### 1. Meteostat (Primary Source)
- **API:** https://meteostat.net/
- **Coverage:** 2018-01-01 to present
- **Features:** Temperature, precipitation, wind, pressure, sunshine
- **Issue:** 26-28% missing values

### 2. NASA POWER (Secondary Source)
- **API:** https://power.larc.nasa.gov/
- **Coverage:** Complete coverage for requested dates
- **Features:** Temperature, precipitation, wind, pressure, humidity, solar radiation
- **Purpose:** Fill missing Meteostat values and add new features

### 3. NDMA Reports (Planned)
- **Source:** National Disaster Management Authority
- **Purpose:** Label flood events for supervised learning
- **Status:** Manual labeling required

## Project Structure

```
.
├── code/
│   ├── __init__.py
│   ├── preprocessing.py              # Data cleaning & feature engineering ✅
│   ├── baseline_models.py            # ML model training ✅
│   ├── model_evaluation.py           # Model evaluation & visualizations ✅
│   ├── fetch_meteostat_weather.py    # Fetch Meteostat data
│   ├── fetch_nasa_power.py           # Fetch NASA POWER data
│   ├── merge_weather_data.py         # Merge datasets
│   └── ... (other modules)
├── data/
│   ├── raw/                          # Raw data from APIs
│   └── processed/                    # Processed datasets
├── docs/
│   ├── data_report.md                # Data collection report
│   └── data_merge_guide.md           # Merging guide
├── notebooks/
│   └── ml_pipeline.ipynb             # Interactive ML pipeline ✅
├── examples/
│   └── complete_data_pipeline.py     # Full pipeline example
├── results/                          # ML model outputs ✅
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── training_data.csv
│   ├── test_data.csv
│   ├── model_metrics.csv
│   ├── *.png                         # Visualizations
│   └── evaluation_report.txt
├── tests/
│   └── test_merge_integration.py     # Integration tests
├── run_pipeline.py                   # Main orchestration script ✅
├── test_model.py                     # Model testing ✅
├── verify_predictions.py             # Prediction verification ✅
├── requirements.txt                  # Python dependencies
├── ENVIRONMENT_SETUP.md              # Setup guide
├── ML_PIPELINE_README.md             # ML pipeline documentation
├── XGBOOST_ERROR_RESOLUTION.md       # XGBoost fix documentation
└── README.md                         # This file
```

## Data Pipeline

```
┌─────────────────┐     ┌──────────────────┐
│ Meteostat API  │     │ NASA POWER API   │
│  (Primary)      │     │  (Secondary)     │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│  Merge & Fill Missing Values            │
│  - Use Meteostat as primary             │
│  - Fill gaps with NASA POWER            │
│  - Add humidity & solar radiation       │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Complete Weather Dataset               │
│  - 0% missing values                    │
│  - 18 features                          │
│  - Ready for ML                         │
└─────────────────────────────────────────┘
```

## Data Quality

### Before Merge (Meteostat Only)
- 3,914 rows
- 26-28% missing values in key features
- No humidity or solar radiation data

### After Merge
- 3,914 rows
- **0% missing values** for all key features
- Added humidity and solar radiation columns
- Ready for machine learning

## Model Performance

### Data Preprocessing Results
- **Dataset:** 5,752 samples from Swat & Upper Dir districts
- **Features Engineered:** 19 features (temperature, precipitation, wind, pressure, humidity, solar radiation, temporal, rolling averages)
- **Train/Test Split:** 80/20 stratified (4,601 training / 1,151 test samples)
- **Scaling:** StandardScaler normalization applied

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Status |
|-------|----------|-----------|--------|----------|---------|--------|
| **Logistic Regression** | 99.91% | 0.0000 | 0.0000 | 0.0000 | **0.8243** | ✅ Working |
| **Random Forest** | 99.91% | 0.0000 | 0.0000 | 0.0000 | **0.8643** | ✅ **Best** |

### Model Features
- ✅ Trained on 4,601 samples
- ✅ Evaluated on 1,151 test samples
- ✅ Cross-validation applied (5-fold for LR)
- ✅ Models serialized and saved (.pkl files)
- ✅ Ready for real-time predictions

### Generated Outputs
```
results/
├── training_data.csv                          (1.7 MB - Preprocessed training data)
├── test_data.csv                              (428 KB - Preprocessed test data)
├── logistic_regression_model.pkl              (873 B - Trained LR model)
├── random_forest_model.pkl                    (406 KB - Trained RF model)
├── model_metrics.csv                          (Performance metrics)
├── model_performance_comparison.png           (Bar charts)
├── roc_curves.png                             (ROC curves with AUC)
├── confusion_matrices.png                     (Confusion matrices)
├── feature_importance_logistic_regression.csv
├── feature_importance_random_forest.csv
├── feature_importance_logistic_regression.png
├── feature_importance_random_forest.png
├── feature_importance.json
├── evaluation_report.txt                      (Detailed analysis)
└── prediction_verification_report.png         (Verification visualizations)
```

## Testing

### Model Testing Scripts ✅

1. **Quick Model Test:**
```bash
python3 test_model.py
```
- Tests both models on 1,151 test samples
- Displays accuracy and AUC-ROC scores
- Shows sample predictions

2. **Comprehensive Prediction Verification (14-Step Procedure):**
```bash
python3 verify_predictions.py
```
- Verifies model files exist
- Tests predictions on all test samples
- Compares model performance
- Tests on random samples
- Generates visualizations
- Creates verification report

3. **Data Merge Integration Test:**
```bash
python tests/test_merge_integration.py
```

The tests verify:
- All model files are loadable
- Predictions can be made successfully
- Performance metrics are calculated correctly
- Data values are within reasonable ranges
- Models work on new unseen data

## Next Steps

1. ✅ Collect and merge weather data
2. ✅ Preprocess data (feature engineering, scaling)
3. ✅ Train baseline ML models (Logistic Regression, Random Forest)
4. ✅ Evaluate models with comprehensive metrics
5. ✅ Verify model predictions working correctly
6. ⏳ Build web application (Flask/Streamlit) for predictions
7. ⏳ Integrate real-time weather API
8. ⏳ Deploy the application (Heroku/AWS/GCP)
9. ⏳ Set up model monitoring and retraining pipeline
10. ⏳ Implement advanced features (SMOTE for class imbalance, hyperparameter tuning)

## Documentation

- [ML Pipeline README](ML_PIPELINE_README.md) - Complete ML pipeline documentation
- [Environment Setup Guide](ENVIRONMENT_SETUP.md) - Virtual environment setup instructions
- [XGBoost Error Resolution](XGBOOST_ERROR_RESOLUTION.md) - Troubleshooting guide
- [Data Collection Report](docs/data_report.md) - Detailed information about data sources
- [Data Merge Guide](docs/data_merge_guide.md) - Step-by-step merging instructions

## Requirements

- Python 3.8+
- pandas
- numpy
- requests
- meteostat
- beautifulsoup4
- lxml
- geopy

See [requirements.txt](requirements.txt) for complete list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is part of an AI-based natural disaster prediction system for Pakistan.

## Acknowledgments

- Meteostat for providing historical weather data
- NASA POWER for satellite-derived meteorological data
- NDMA for disaster event reports
