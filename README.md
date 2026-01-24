<div align="center">

# 🌊 AI-Based Natural Disaster Prediction System

### Intelligent Flood Prediction for Khyber Pakhtunkhwa, Pakistan

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Click_Here-success?style=for-the-badge)](https://ai-based-natural-disaster-prediction.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<p align="center">
  <strong>An AI-powered early warning system leveraging machine learning to predict flood events and save lives.</strong>
</p>

[**🌐 Try Live Demo**](https://ai-based-natural-disaster-prediction.streamlit.app/) • [**📖 Documentation**](#-documentation) • [**🚀 Quick Start**](#-quick-start) • [**🤝 Contributing**](#-contributing)

</div>

---

## 🎯 Overview

This project is a comprehensive **AI-based flood prediction system** designed for high-risk districts in Pakistan (Swat and Upper Dir). It combines cutting-edge machine learning with real-time weather data to provide accurate flood risk assessments.

### 🔑 Key Capabilities

| Feature | Description |
|---------|-------------|
| **🌐 Real-time Prediction** | Live weather data integration via OpenWeatherMap API |
| **📊 Historical Analysis** | 25 years of weather data (2000-2025) from NASA POWER & Meteostat |
| **🤖 ML-Powered** | Trained on 18,902+ weather observations with 24 engineered features |
| **🧠 Multi-AI Approach** | Search Algorithms, CSP, Neural Networks, Clustering, and RL |

### 💡 Why This Project?

Pakistan faces devastating floods every year, especially during monsoon season (June-September). This system aims to:

- ⚡ **Predict** flood risk based on weather conditions with 60% recall rate
- 🏛️ **Assist** authorities in making informed evacuation decisions
- 🚨 **Provide** early warnings to save lives and minimize damage

---

## 📋 Table of Contents

<details>
<summary>Click to expand</summary>

- [Overview](#-overview)
- [Features](#-features)
- [AI Techniques](#-ai-techniques-implemented)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#️-how-to-run)
- [Project Structure](#-project-structure)
- [How It Works](#️-how-it-works)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [API Configuration](#-api-keys)
- [Docker Deployment](#-docker-deployment)
- [Technologies](#️-technologies-used)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

</details>

---

## ✨ Features

### 🖥️ Main Application

| Feature | Description |
|---------|-------------|
| 🏠 **Interactive Dashboard** | Real-time flood risk prediction with live weather data |
| 🔮 **Custom Prediction** | Manual weather parameter input for custom scenarios |
| 📊 **Historical Explorer** | 25 years of weather and flood data visualization |
| 🤖 **Model Insights** | Performance metrics, feature importance & explainability |
| 📍 **Location Support** | Swat & Upper Dir district coverage |

### 🧠 AI Technique Demonstrations

| Technique | Application | Status |
|-----------|-------------|--------|
| 🔍 **Search Algorithms** | A*, BFS, DFS for evacuation route planning | ✅ Interactive |
| 🧩 **CSP Solver** | Resource allocation for emergency response | ✅ Interactive |
| 🧬 **LSTM Neural Network** | Time-series flood prediction | ✅ Interactive |
| 📈 **K-Means Clustering** | Weather pattern analysis & classification | ✅ Interactive |
| 🎮 **Q-Learning** | Reinforcement learning for evacuation decisions | ✅ Interactive |
| 🔬 **SHAP/LIME** | Model explainability & interpretability | ✅ Interactive |

---

## 🚀 Quick Start

**Try the live demo instantly — no installation required!**

<div align="center">

### [🌐 Launch Live Demo](https://ai-based-natural-disaster-prediction.streamlit.app/)

</div>

Or run locally in 3 steps:

```bash
# 1. Clone repository
git clone https://github.com/zohaibkhan745/-AI-Based-Natural-Disaster-Prediction-Web-App-.git
cd -AI-Based-Natural-Disaster-Prediction-Web-App-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch application
streamlit run app.py
```

---

## 🧠 AI Techniques Implemented

<details>
<summary><b>🔍 1. Search Algorithms (Evacuation Route Planning)</b></summary>

**File:** `code/search_algorithms.py`

Finds optimal evacuation routes from flooded areas to safe zones.

```python
# Algorithms implemented:
- A* Search (informed, optimal)
- Breadth-First Search (optimal for unweighted)
- Depth-First Search (memory efficient)
```

**How it works:** Creates a grid-based flood scenario where some cells are flooded (obstacles). The algorithms find the shortest path from a start position to the nearest safe zone.

</details>

<details>
<summary><b>🧩 2. Constraint Satisfaction Problem (Resource Allocation)</b></summary>

**File:** `code/csp_resource_allocation.py`

Allocates emergency resources (medical teams, rescue boats, supplies) to evacuation shelters.

```python
# Techniques used:
- AC-3 Arc Consistency (preprocessing)
- Backtracking Search
- MRV Heuristic (Minimum Remaining Values)
- LCV Heuristic (Least Constraining Value)
```

**How it works:** Given shelters with different populations and resource requirements, and limited resources, finds an optimal allocation that satisfies all constraints.

</details>

<details>
<summary><b>🧬 3. LSTM Neural Network (Time-Series Prediction)</b></summary>

**File:** `code/neural_network.py`

Time-series prediction using Long Short-Term Memory networks.

```
Architecture:
Input (7 days × 5 features) → LSTM (64 units) → Dense (1, sigmoid)
```

**How it works:** Looks at the past 7 days of weather data to predict if a flood will occur. The LSTM can capture patterns like gradual rainfall buildup.

</details>

<details>
<summary><b>📈 4. K-Means Clustering (Weather Pattern Analysis)</b></summary>

**File:** `code/clustering.py`

Groups weather conditions into risk categories.

```
Clusters identified:
- Monsoon Pattern (HIGH RISK)
- Flash Flood Conditions (HIGH RISK)
- Moderate Rain (MODERATE RISK)
- Dry Conditions (LOW RISK)
```

**How it works:** Uses K-Means++ initialization to group similar weather patterns. Automatically labels clusters based on their characteristics.

</details>

<details>
<summary><b>🎮 5. Q-Learning / Reinforcement Learning (Evacuation Decisions)</b></summary>

**File:** `code/reinforcement_learning.py`

Learns optimal evacuation decisions through trial and error.

```
Environment:
- States: (flood_level, population_at_risk, resources, time)
- Actions: Wait, Warn, Voluntary Evac, Mandatory Evac, Deploy Resources
- Rewards: +100/person saved, -500/casualty
```

**How it works:** Simulates thousands of flood scenarios. The agent learns when to issue warnings, start evacuations, and deploy resources to maximize lives saved.

</details>

<details>
<summary><b>🔬 6. SHAP & LIME Explainability (Model Interpretation)</b></summary>

**File:** `code/explainability.py`

Explains why the model made a specific prediction.

```
Example output:
"Flood risk is 85% because:
 - Heavy rainfall (+40%)
 - High humidity (+25%)
 - Monsoon season (+15%)"
```

</details>

---

## � Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.9+ |
| pip | Latest |
| Git | Latest |

### Step 1: Clone the Repository

```bash
git clone https://github.com/zohaibkhan745/-AI-Based-Natural-Disaster-Prediction-Web-App-.git
cd -AI-Based-Natural-Disaster-Prediction-Web-App-
```

### Step 2: Create Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up API Key (Optional but Recommended)

Create `.streamlit/secrets.toml`:

```toml
OPENWEATHER_API_KEY = "your_api_key_here"
```

Get a free API key from [OpenWeatherMap](https://openweathermap.org/api).

---

## ▶️ How to Run

### Option 1: Run the Web App (Recommended)

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Option 2: Run with Docker

```bash
docker-compose up --build
```

### Option 3: Run Individual Components

| Command                        | Description                   |
| ------------------------------ | ----------------------------- |
| `streamlit run app.py`         | Start web application         |
| `python run_pipeline.py`       | Run full ML training pipeline |
| `python test_model.py`         | Test model predictions        |
| `python verify_predictions.py` | Verify model outputs          |

### Run AI Technique Demos

```bash
# Search Algorithms Demo
python code/search_algorithms.py

# CSP Demo
python code/csp_resource_allocation.py

# Neural Network Demo
python code/neural_network.py

# Clustering Demo
python code/clustering.py

# Reinforcement Learning Demo
python code/reinforcement_learning.py

# Explainability Demo
python code/explainability.py
```

---

## 📁 Project Structure

```
AI-Based-Natural-Disaster/
│
├── 📱 app.py                          # Main Streamlit web application
│
├── 📂 code/                           # Source code modules
│   ├── search_algorithms.py           # A*, BFS, DFS (Week 8)
│   ├── csp_resource_allocation.py     # CSP (Week 9)
│   ├── neural_network.py              # LSTM (Week 11)
│   ├── clustering.py                  # K-Means (Week 12)
│   ├── reinforcement_learning.py      # Q-Learning (Week 12)
│   ├── explainability.py              # SHAP/LIME (Bonus)
│   ├── improved_models.py             # ML model training
│   ├── preprocessing.py               # Data preprocessing
│   ├── baseline_models.py             # Baseline ML models
│   ├── model_evaluation.py            # Evaluation metrics
│   ├── fetch_nasa_power.py            # NASA POWER API
│   ├── fetch_meteostat_weather.py     # Meteostat API
│   ├── merge_weather_data.py          # Data merging
│   └── label_historical_floods.py     # Flood labeling
│
├── 📂 data/
│   ├── raw/                           # Raw API data
│   │   ├── nasa_power_*.csv
│   │   ├── weather_*.csv
│   │   └── ndma_flood_reports.csv
│   └── processed/                     # Cleaned datasets
│       ├── flood_weather_dataset.csv  # Main training data (18,902 records)
│       ├── cleaned_swat.csv
│       └── cleaned_upper_dir.csv
│
├── 📂 results/                        # Model outputs
│   ├── best_flood_model.pkl           # Trained model
│   ├── model_metrics.csv              # Performance metrics
│   ├── feature_importance.json        # Feature rankings
│   └── evaluation_report.txt          # Detailed report
│
├── 📂 docs/                           # Documentation
├── 📂 notebooks/                      # Jupyter notebooks
├── 📂 .streamlit/                     # Streamlit config
├── 📂 .github/workflows/              # CI/CD
│
├── 🐳 Dockerfile                      # Docker config
├── 🐳 docker-compose.yml              # Docker Compose
├── 📋 requirements.txt                # Python dependencies
├── 📖 README.md                       # This file
└── 📖 AI_TECHNIQUES_SUMMARY.md        # AI techniques documentation
```

---

## ⚙️ How It Works

### Data Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   NASA POWER    │────▶│   Data Merge    │────▶│   Preprocessing │
│   (2000-2025)   │     │   & Cleaning    │     │   24 Features   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                              │
         │              ┌─────────────────┐            ▼
         └─────────────▶│   Fill Missing  │     ┌─────────────────┐
                        │   Values        │     │   ML Training   │
┌─────────────────┐     └─────────────────┘     │   (3 Models)    │
│   Meteostat     │────────────────────────────▶└─────────────────┘
│   (2018-2025)   │                                    │
└─────────────────┘                                    ▼
                                                ┌─────────────────┐
┌─────────────────┐                             │   Best Model    │
│   NDMA Reports  │────▶ Flood Labels ─────────▶│   (60% Recall)  │
│   + Historical  │      (517 events)           └─────────────────┘
└─────────────────┘
```

### Prediction Flow

```
User Input          ──▶  Feature Engineering  ──▶  Model Prediction
(Weather Data)           (24 features)             (Flood Probability)
                                                          │
                                                          ▼
                                                   Risk Assessment
                                                   LOW / MODERATE / HIGH
```

### 24 Engineered Features

| Category          | Features                                                                    |
| ----------------- | --------------------------------------------------------------------------- |
| **Temperature**   | tavg, tmin, tmax, temp_range, tavg_7day_avg                                 |
| **Precipitation** | prcp, prcp_7day_avg, prcp_3day_sum, prcp_7day_sum, heavy_rain, extreme_rain |
| **Atmospheric**   | pres, humidity, pressure_anomaly, high_humidity                             |
| **Wind**          | wspd, wpgt, wspd_7day_avg                                                   |
| **Solar**         | solar_radiation                                                             |
| **Temporal**      | month, day_of_year, quarter, is_monsoon                                     |
| **Location**      | location_encoded                                                            |

---

## 📊 Dataset

### Statistics

| Metric            | Value                          |
| ----------------- | ------------------------------ |
| **Total Records** | 18,902                         |
| **Time Range**    | January 2000 - November 2025   |
| **Flood Events**  | 517 (2.74%)                    |
| **Features**      | 24 engineered                  |
| **Locations**     | Swat, Upper Dir (KP, Pakistan) |

### Data Sources

1. **NASA POWER API** - Satellite-derived meteorological data (2000-2025)
2. **Meteostat API** - Ground station weather data (2018-2025)
3. **NDMA Reports** - Historical flood event records
4. **Historical Archives** - Major flood events database

---

## 📈 Model Performance

### 🏆 Best Model: Logistic Regression (Class Weighted)

<div align="center">

| Metric | Score | Notes |
|--------|-------|-------|
| **Recall** | 60% ⭐ | Primary optimization target |
| **Precision** | 45% | Acceptable false alarm rate |
| **F1 Score** | 51% | Balanced performance |
| **Accuracy** | 97% | Overall correctness |

</div>

### 💡 Why Recall Matters

In flood prediction, **missing a real flood is worse than a false alarm**:

- ✅ **60% of actual floods are correctly detected**
- ⚠️ Some false alarms (acceptable trade-off for safety)
- 🛡️ Prioritizes human safety over precision

### 📊 Model Comparison

| Model | Recall | Precision | F1 | Best For |
|-------|--------|-----------|-----|----------|
| **Logistic Regression** | **60%** | 45% | 51% | ⭐ Production |
| Random Forest | 53% | 52% | 52% | Balanced |
| Gradient Boosting | 43% | 58% | 49% | Low False Alarms |

---

## 🔑 API Keys

### OpenWeatherMap (For Real-time Weather)

1. Sign up at [OpenWeatherMap](https://openweathermap.org/api)
2. Get your free API key
3. Create `.streamlit/secrets.toml`:

```toml
OPENWEATHER_API_KEY = "your_api_key_here"
```

**Without API key:** The app uses demo/simulated weather data.

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
docker-compose up --build
```

### Manual Docker Build

```bash
# Build the image
docker build -t flood-prediction-app .

# Run the container
docker run -p 8501:8501 flood-prediction-app
```

Access the app at `http://localhost:8501`

---

## 🛠️ Technologies Used

<div align="center">

| Category | Technologies |
|----------|--------------|
| **Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) |
| **ML/AI** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) |
| **Data Sources** | OpenWeatherMap API • NASA POWER • Meteostat |
| **Deployment** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white) ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=github-actions&logoColor=white) |
| **Version Control** | ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white) |

</div>

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [AI_TECHNIQUES_SUMMARY.md](AI_TECHNIQUES_SUMMARY.md) | Comprehensive AI techniques documentation |
| [ML_PIPELINE_README.md](ML_PIPELINE_README.md) | Machine learning pipeline details |
| [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md) | Streamlit application guide |
| [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) | Environment setup instructions |
| [QUICK_START.md](QUICK_START.md) | Quick start guide |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📚 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
requests>=2.31.0
python-dateutil>=2.8.2
```

Full list in `requirements.txt`

---

## 👨‍💻 Author

**CS351 - Artificial Intelligence Project**  
Semester 5

---

## ⚠️ Disclaimer

> **Note:** This is an **educational project** demonstrating AI techniques for disaster prediction. For actual emergency situations, please refer to official sources:

| Resource | Link |
|----------|------|
| NDMA Pakistan | [ndma.gov.pk](https://ndma.gov.pk/) |
| PMD Pakistan | [pmd.gov.pk](https://www.pmd.gov.pk/) |
| Emergency Services | Local authorities |

---

## 🙏 Acknowledgments

- **NASA POWER** — Satellite-derived meteorological data
- **Meteostat** — Ground station weather data
- **NDMA Pakistan** — Historical flood reports
- **Streamlit** — Web application framework
- **scikit-learn** — Machine learning tools

---

<div align="center">

### 🌐 [Try the Live Demo](https://ai-based-natural-disaster-prediction.streamlit.app/)

<br>

**Made with ❤️ for CS351 - Artificial Intelligence**

<br>

[![Stars](https://img.shields.io/github/stars/zohaibkhan745/-AI-Based-Natural-Disaster-Prediction-Web-App-?style=social)](https://github.com/zohaibkhan745/-AI-Based-Natural-Disaster-Prediction-Web-App-)
[![Forks](https://img.shields.io/github/forks/zohaibkhan745/-AI-Based-Natural-Disaster-Prediction-Web-App-?style=social)](https://github.com/zohaibkhan745/-AI-Based-Natural-Disaster-Prediction-Web-App-)

</div>
