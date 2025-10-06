import os
import config.paths as paths
from pipelines.b_preprocessing.preprocessing_pipeline import PreprocessingPipeline
from dotenv import load_dotenv
load_dotenv(dotenv_path=paths.DOTENV_FILE)

import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn


def create_engine_connection(db_credentials: dict):
    return create_engine(
        f"postgresql+psycopg2://{db_credentials['user']}:{db_credentials['password']}@{db_credentials['host']}:{db_credentials['port']}/{db_credentials['dbname']}"
    )

with paths.DB_CREDENTIALS_FILE.open("r") as f:
    db_credentials = json.load(f)

# Load data from database
print("Loading data from database...")
query = f"SELECT * FROM public.cars_scraped WHERE EXTRACT(YEAR FROM created_at) = 2025 AND EXTRACT(MONTH FROM created_at) = 4;"
engine = create_engine_connection(db_credentials)
df = pd.read_sql(query, engine)
print("\tData loaded\n")

# Preprocess data
print("Loading preprocessing pipeline...")
preproc = PreprocessingPipeline()
preproc_pipeline = preproc.create_pipeline(df)
print("\tPreprocessing pipeline loaded\n")

# Create X and y and train-test split
X = df.drop(columns=["price_cash"]).copy()
y = df["price_cash"].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=paths.RANDOM_SEED)

params_grid = {
    "XGBRegressor": {
        "model_instance": XGBRegressor(),
        "grid_sizes": {
            "small": {
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.1, 0.3],
                "model__n_estimators": [100, 300],
            },
            "medium": {
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.05, 0.1],
                "model__n_estimators": [100, 300],
                "model__min_child_weight": [1, 5],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
            "big": {
                "model__max_depth": [3, 6, 10, 12],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__n_estimators": [100, 300, 500],
                "model__min_child_weight": [1, 3, 7],
                "model__subsample": [0.6, 0.8, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
                "model__gamma": [0, 0.3, 1],
                "model__reg_alpha": [0, 1, 5],
                "model__reg_lambda": [1, 5, 10],
            },
        },
    },
    "RandomForestRegressor": {
        "model_instance": RandomForestRegressor(),
        "grid_sizes": {
            "small": {
                "model__n_estimators": [100, 300],
                "model__max_depth": [None, 10],
                "model__max_features": ["sqrt", "log2"],
            },
            "medium": {
                "model__n_estimators": [100, 300],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", "log2"],
            },
            "big": {
                "model__n_estimators": [100, 300, 500],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
                "model__bootstrap": [True, False],
            },
        },
    },
    "DecisionTreeRegressor": {
        "model_instance": DecisionTreeRegressor(),
        "grid_sizes": {
            "small": {
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 10],
            },
            "medium": {
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
            "big": {
                "model__max_depth": [None, 5, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4, 10],
                "model__max_features": ["sqrt", "log2", None],
                "model__ccp_alpha": [0.0, 0.01, 0.1],
            },
        },
    },
    "Lasso": {
        "model_instance": Lasso(),
        "grid_sizes": {
            "small": {
                "model__alpha": [0.01, 0.1, 1.0],
            },
            "medium": {
                "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
            },
            "big": {
                "model__alpha": [1e-6, 1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
            },
        },
    },
    "Ridge": {
        "model_instance": Ridge(),
        "grid_sizes": {
            "small": {
                "model__alpha": [0.1, 1.0, 10.0],
            },
            "medium": {
                "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 1000.0],
            },
            "big": {
                "model__alpha": [1e-6, 1e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0],
            },
        },
    },
    "LinearRegression": {
        "model_instance": LinearRegression(),
        "grid_sizes": {
            "small": {
                "model__fit_intercept": [True, False],
                "model__positive": [False]
            },
            "medium": {
                "model__fit_intercept": [True, False],
                "model__positive": [False, False],
            },
            "big": {
                "model__fit_intercept": [True, False],
                "model__positive": [False, False],
                "model__copy_X": [True, False],
            },
        },
    },
}

def evaluate_and_select_best_model(X, y, model_instance, preproc_pipeline, param_grid, scoring="r2"):
    
    pipeline = Pipeline([
        ("preprocessor", preproc_pipeline),
        ("model", model_instance)  # Modelo "base" (se cambiará en la búsqueda)
    ])

    # 4) Llamamos a GridSearchCV para probar todas las combinaciones
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=5,
        n_jobs=-1
    )

    # 5) Ajustamos la búsqueda
    search.fit(X, y)

    # 6) Extraemos el mejor pipeline y su score
    best_model = search.best_estimator_
    best_score = search.best_score_

    print("Mejor configuración encontrada:\n", search.best_params_)
    print(f"Mejor score ({scoring}): {best_score:.4f}")

    return best_model, best_score, search

# mlflow.set_tracking_uri(f"file:{paths.MLRUNS_DIR}")  # o cualquier ruta absoluta donde quieras guardar los logs
mlflow.set_tracking_uri(paths.TRACKING_URI)

# grid_sizes = ["small", "medium", "big"]
# model_names = ["LinearRegression", "Ridge", "Lasso", "DecisionTreeRegressor", "RandomForestRegressor", "XGBRegressor"]
grid_sizes = ["big"]
model_names = ["DecisionTreeRegressor", "RandomForestRegressor", "XGBRegressor"]


for grid_size in grid_sizes:
    for model_name in model_names:

        model_instance = params_grid[model_name]["model_instance"]
        param_grid = params_grid[model_name]["grid_sizes"][grid_size]

        # Set MLFlow experiment
        mlflow.set_experiment("car-price-prediction-gridsearchs")

        with mlflow.start_run(run_name=f"202504_GridSearchCV_{model_name}_{grid_size}") as run:
            
            print("Evaluating models...")
            best_model, best_score, search_obj = evaluate_and_select_best_model(X_train, y_train, model_instance=model_instance, preproc_pipeline=preproc_pipeline, param_grid=param_grid, scoring="r2")
            print("\tModels evaluated\n")

            # Log hyperparameters of the best model
            best_params = search_obj.best_params_
            for param_name, value in best_params.items():
                mlflow.log_param(param_name, value)
            # Log the best score for the best model
            mlflow.log_metric("best_score_r2", best_score)
            # Log best model
            input_example = X_train.iloc[:1]
            mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model", registered_model_name="CarPricePredictionModel", input_example=input_example)
            # Log search object
            search_obj_temp_path = "search_obj_temp.pkl"
            joblib.dump(search_obj, search_obj_temp_path)
            mlflow.log_artifact(search_obj_temp_path, artifact_path="search_object")
            os.remove(search_obj_temp_path)

            # Log metrics on the train set
            y_train_pred = best_model.predict(X_train)
            r2_train = r2_score(y_train, y_train_pred)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            mse_train = mean_squared_error(y_train, y_train_pred)
            rmse_train = np.sqrt(mse_train)
            mlflow.log_metric("r2_train", r2_train)
            mlflow.log_metric("mae_train", mae_train)
            mlflow.log_metric("mse_train", mse_train)
            mlflow.log_metric("rmse_train", rmse_train)

            # Log metrics on the test set
            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mlflow.log_metric("r2_test", r2)
            mlflow.log_metric("mae_test", mae)
            mlflow.log_metric("mse_test", mse)
            mlflow.log_metric("rmse_test", rmse)