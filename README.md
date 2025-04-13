# car-price-checker-api

## 🚗 Project Overview

**car-price-checker** is a real-world, end-to-end system designed to predict used car prices based on vehicle features. The main goal of this project is to showcase my skills as a Data Scientist by building a production-ready solution that covers the full ML lifecycle: from data ingestion to model serving via an API.

This repository covers the **backend component** of the system, including data pipelines, model training, and a REST API for real-time inference. A separate repository [car-price-checker-webapp](https://github.com/lopezrbn/car-price-checker-webapp) contains the frontend interface used by end users.

The working webapp can be used at [https://car-price-checker.lopezrbn.com/](https://car-price-checker.lopezrbn.com/).

---

## 📦 Repository Scope

This repository includes:

- A monthly ingestion pipeline that scrapes raw data from a real website.
- A modular preprocessing and feature engineering pipeline.
- A training pipeline that evaluates and selects the best ML model.
- A REST API that serves predictions.
- Deployment configuration to run the API as a background service.

The frontend is not included here and can be found in the companion repo.

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

## 📥 Ingestion Pipeline

The ingestion pipeline performs the following tasks:

- **Monthly execution**: The pipeline is run on a monthly basis.
- **Scraping**: It scrapes all listings from [coches.com](https://www.coches.com) using `requests` and `BeautifulSoup`.
- **Structured output**: Extracts structured data such as manufacturer, model, year, fuel type, transmission, power, etc.
- **Validation and cleaning**: Includes functions to clean and standardize data types.
- **Storage**: Validated data is stored in a self-hosted **PostgreSQL** database.
- **Resilience**: The loop state (by manufacturer/model) is saved to allow for graceful interruption and restart.

---

## 🔧 Preprocessing Pipeline

The preprocessing stage consists of two major blocks:

### 1. Feature Engineering

Custom features are created based on domain knowledge, such as:

- `age` and `kms_per_year`
- Binned variables like `age_bins`, `kms_per_year_bins`
- Popularity metrics: `model_rel_freq`, `manuf_rel_freq`
- Segment-level and country-level aggregates
- Manufacturer metadata: `manuf_country`, `exclusivity_level`, `manuf_group`

### 2. Preprocessing Transformers

After feature engineering, a `sklearn` pipeline applies:

- **StandardScaler** for numeric features.
- **OneHotEncoder** for low-cardinality categorical features.
- **TargetEncoder** for high-cardinality categorical features like `manufacturer` and `model`.

---

## 🧠 Training Pipeline

The training pipeline:

- Loads data from the PostgreSQL database.
- Constructs a `sklearn` pipeline with preprocessing + model.
- Evaluates **6 different models**:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree
  - Random Forest
  - XGBoost
- Performs **GridSearchCV** with cross-validation to find the best hyperparameters for each model.
- Selects the best model based on the highest R² score.
- Saves the best pipeline using `joblib` for use in production.

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

## 🛠 Tech Stack

- **Language**: Python
- **ML Libraries**: scikit-learn, XGBoost, category-encoders
- **Data Handling**: pandas, numpy
- **Web Scraping**: requests, BeautifulSoup
- **Database**: PostgreSQL
- **API**: FastAPI
- **Deployment**: gunicorn + uvicorn + systemd

---

## 🧪 Future Work

- Add unit tests and API tests
- Add model versioning with MLflow or DVC
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

---
