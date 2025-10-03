# car-price-checker-api

## 🚗 Project Overview

**car-price-checker** is a real-world, end-to-end system designed to predict used car prices based on vehicle features. The main goal of this project is to showcase my skills as a Data Scientist by building a production-ready solution that covers the full ML lifecycle: from data ingestion to model serving via an API.

This repository covers the **backend component** of the system, including data pipelines, model training, and a REST API for real-time inference. A separate repository [car-price-checker-webapp](https://github.com/lopezrbn/car-price-checker-webapp) contains the frontend interface used by end users.

The app is live at:  
👉 https://car-price-checker.lopezrbn.com/

---

## 📦 Repository Scope

This repository includes:

- A monthly ingestion pipeline that scrapes raw data from a real website.
- A modular preprocessing and feature engineering pipeline.
- A training pipeline that evaluates and selects the best ML model, with experiment tracking via MLflow.
- A REST API that serves predictions.
- Deployment configuration to run the API as a background service.

The frontend is not included here and can be found in the companion repo.

---

## 🛠 Tech Stack

- **Language**: Python
- **ML Libraries**: scikit-learn, XGBoost, category-encoders
- **Data Handling**: pandas, numpy
- **Web Scraping**: requests, BeautifulSoup
- **Database**: PostgreSQL
- **Experiment Tracking**: MLflow
- **Orchestration**: Apache Airflow
- **API**: FastAPI
- **Deployment**: gunicorn + uvicorn + systemd

---

## 🧭 System Architecture

```
[coches.com] → [Ingestion Pipeline] → [PostgreSQL DB]
                                 ↓
                          [Preprocessing Pipeline]
                                 ↓
                          [Training Pipeline]
                                 ↓
                           [Saved Model.pkl]
                                 ↓
                      [FastAPI REST Inference API]
```

---

## 🗄️ PostgreSQL Database

A self-hosted **PostgreSQL** instance is used as the central storage for scraped data.  
Key aspects:

- All raw car listings are inserted into a table `cars_scraped`.
- Data is partitioned and queried by `created_at` timestamps (year/month).
- Both the ingestion pipeline (writing) and the training pipeline (reading) interact with this database.
- Credentials are securely stored in a JSON file (referenced via `config.paths`).

This ensures persistence and reliability in data access across the full ML lifecycle.

---

## ⚙️ Pipelines

### 📥 Ingestion Pipeline

- **Monthly execution**: The pipeline is run on a monthly basis.
- **Scraping**: It scrapes all listings from [coches.com](https://www.coches.com) using `requests` and `BeautifulSoup`.
- **Structured output**: Extracts structured data such as manufacturer, model, year, fuel type, transmission, power, etc.
- **Validation and cleaning**: Includes functions to clean and standardize data types.
- **Storage**: Validated data is stored in PostgreSQL.
- **Resilience**: The loop state (by manufacturer/model) is saved to allow for graceful interruption and restart.

### 🔧 Preprocessing Pipeline

Custom features and preprocessing transformers are combined in a `scikit-learn` pipeline.

**Feature engineering:**
- `age` and `kms_per_year`
- Binned variables like `age_bins`, `kms_per_year_bins`
- Popularity metrics: `model_rel_freq`, `manuf_rel_freq`
- Segment and country-level aggregates
- Manufacturer metadata: `manuf_country`, `exclusivity_level`, `manuf_group`

**Transformers:**
- `StandardScaler` for numeric features.
- `OneHotEncoder` for low-cardinality categorical features.
- `TargetEncoder` for high-cardinality categorical features like `manufacturer` and `model`.

### 🧠 Training Pipeline

The training pipeline is the core of the ML workflow. It:

- Loads data for the current month from PostgreSQL.
- Builds a preprocessing pipeline dynamically based on the input data.
- Evaluates multiple models using `GridSearchCV`:
  - Linear Regression, Ridge, Lasso
  - Decision Tree
  - Random Forest
  - XGBoost
- Performs hyperparameter optimization with cross-validation.
- Logs hyperparameters, metrics, and the best model into **MLflow**.
- Persists the best model locally as a `.pkl` file for inference.

---

## 📊 Experiment Tracking with MLflow

Experiment tracking is fully integrated into the training pipeline:

- **Tracking URI** is defined in the project’s configuration (`paths.TRACKING_URI`).
- All runs are logged to the MLflow server under the experiment:  
  `"car-price-prediction-training-pipeline"`.
- Each run logs:
  - Hyperparameters of the best model (`mlflow.log_param`)
  - Metrics (R², MAE, MSE, RMSE) on both train and test sets
  - The trained model as an artifact (`mlflow.sklearn.log_model`)
- Best models are also persisted locally to allow deployment in the API.

The MLflow UI is exposed at:  
👉 http://mlflow.lopezrbn.com

---

## 🗂️ Orchestration with Airflow

The full pipeline is orchestrated using **Apache Airflow**, with the following DAG:

- **Monthly schedule**: runs on the first day of every month at midnight.
- **Tasks**:
  1. Ingestion (`ingestion_pipeline.py`)
  2. Training (`training_pipeline.py`)
- **Email notifications**: custom callbacks send emails on task start, success, and failure, including execution time.

The Airflow UI is exposed at:  
👉 http://airflow.lopezrbn.com

---

## 🚀 Inference API

A REST API is provided for real-time price prediction.

- **Framework**: FastAPI
- **Serving**: The model is loaded into memory on startup and used to serve predictions via a `/predict` endpoint.
- **Deployment stack**:
  - `uvicorn` as the ASGI server
  - `gunicorn` as process manager
  - Deployed as a `systemd` service on a self-hosted Ubuntu server
- A sample `.service` file is included for easy deployment.

---

## 🧪 Future Work

- Add unit tests and API tests
- Monitor predictions in production
- Containerize and deploy on cloud (AWS/GCP)
- Expand scraping coverage and resilience

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).  
Feel free to use, modify, and distribute the code with attribution.

---

## 📫 Contact

If you have any questions, suggestions, or feedback, feel free to reach out:

- **Rubén López**  
- Data Scientist  
- 📧 lopezrbn@gmail.com
