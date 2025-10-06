import config.paths as paths
from pipelines.b_preprocessing.preprocessing_pipeline import PreprocessingPipeline
from dotenv import load_dotenv
load_dotenv(dotenv_path=paths.DOTENV_FILE, override=False)  # Load .env file, and do not override existing env vars if they exist

import pandas as pd
import numpy as np
import json
import joblib
from sqlalchemy import create_engine
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn

CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

def create_engine_connection(db_credentials: dict):
    return create_engine(
        f"postgresql+psycopg2://{db_credentials['user']}:{db_credentials['password']}@{db_credentials['host']}:{db_credentials['port']}/{db_credentials['dbname']}"
    )

def evaluate_and_select_best_model(X, y, preproc_pipeline, scoring="r2"):
    """
    Evalúa varios modelos con distintos hiperparámetros usando GridSearchCV
    y elige el mejor según una métrica (por defecto, R²).

    Parámetros:
    -----------
    - df: pd.DataFrame
        Datos crudos con todas las columnas, incluyendo la variable objetivo.
    - preproc_pipeline: Pipeline (o cualquier 'transformer' scikit-learn)
        Pipeline de feature engineering + preprocessing.
        Debe transformar X de DataFrame a la matriz lista para el modelo.
    - scoring: str o callable
        Métrica para optimizar. Por defecto 'r2'. Se podría usar también 'neg_mean_squared_error', etc.

    Retorna:
    --------
    - best_model: Pipeline
        El Pipeline completo (preprocesamiento + modelo) con los mejores hiperparámetros.
    - best_score: float
        El valor de la métrica (por ejemplo, R²) para el mejor modelo.
    - search: GridSearchCV
        El objeto de búsqueda completa, por si necesitas inspeccionar resultados.
    """

    # 1) Separamos X, y
    # X = df.drop(columns=["price_cash"])
    # y = df["price_cash"]

    # 2) Definimos un pipeline que tenga 2 pasos:
    #    - "preprocessor": tu pipeline de preprocesamiento (feature engineering + encoding)
    #    - "model": un estimador cualquiera (se sobreescribirá en el param_grid)
    pipeline = Pipeline([
        ("preprocessor", preproc_pipeline),
        ("model", RandomForestRegressor())  # Modelo "base" (se cambiará en la búsqueda)
    ])

    # 3) Definimos un param_grid que abarca varios modelos y sus hiperparámetros
    #    Observa que el "model" es un paso, así que "model: [RandomForestRegressor(), XGBRegressor()]" 
    #    permite a GridSearchCV probar ambos modelos.
    param_grid = [
        # Linear Regression (sin hiperparámetros)
        {
            "model": [LinearRegression()]
        },
        # Ridge
        {
            "model": [Ridge()],
            "model__alpha": [0.1, 1.0, 10.0, 100.0]
        },
        # Lasso
        {
            "model": [Lasso(max_iter=10000)],
            "model__alpha": [0.01, 0.1, 1.0, 10.0]
        },
        # Decision Tree
        {
            "model": [DecisionTreeRegressor()],
            "model__max_depth": [5, 10, 20, None],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        },
        # Random Forest
        {
            "model": [RandomForestRegressor()],
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [10, 20, None],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        },
        # XGBoost
        {
            "model": [XGBRegressor(use_label_encoder=False, eval_metric="rmse")],
            "model__n_estimators": [100, 200],
            "model__max_depth": [6, 10],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0]
        }
    ]

    # Si quisieras una métrica distinta a 'r2', por ejemplo 'neg_mean_squared_error', 
    # podrías pasar scoring="neg_mean_squared_error" o un make_scorer personalizado.
    # scoring = make_scorer(r2_score)  # si quisieras un callable en vez de un str

    # 4) Llamamos a GridSearchCV para probar todas las combinaciones
    #    cv=5 => 5 folds de cross-validation
    #    n_jobs=-1 => usar todos los cores disponibles para acelerar la búsqueda (opcional)
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

    
def main(year=CURRENT_YEAR, month=CURRENT_MONTH):

    MODEL_PATH_ACTUAL_MONTH = paths.MODELS_DIR / f"{year}{str(month).zfill(2)}_best_model.pkl"

    # Load database credentials
    with paths.DB_CREDENTIALS_FILE.open("r") as f:
        db_credentials = json.load(f)

    # Load data from database
    print("Loading data from database...")

    query = f"SELECT * FROM public.cars_scraped WHERE EXTRACT(YEAR FROM created_at) = {year} AND EXTRACT(MONTH FROM created_at) = {month};"
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

    # Set MLFlow tracking URI
    mlflow.set_tracking_uri(paths.TRACKING_URI)
    # Set MLFlow experiment
    mlflow.set_experiment("car-price-prediction-training-pipeline")

    with mlflow.start_run(run_name=f"training_pipeline_run_{year}{str(month).zfill(2)}") as run:
        
        print("Evaluating models...")
        best_model, best_score, search_obj = evaluate_and_select_best_model(X_train, y_train, preproc_pipeline=preproc_pipeline)
        print("\tModels evaluated\n")

        # Log hyperparameters of the best model
        best_params = search_obj.best_params_
        for param_name, value in best_params.items():
            mlflow.log_param(param_name, value)
        # Log the best score for the best model
        mlflow.log_metric("best_score_r2", best_score)
        # Log best model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

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

    print("Best score:")
    print(best_score)
    print("Best model:")
    print(best_model)

    print("Saving best model...")
    joblib.dump(best_model, MODEL_PATH_ACTUAL_MONTH)
    print("\tBest model saved\n")

if __name__ == "__main__":

    main(CURRENT_YEAR, CURRENT_MONTH)
    # for month in range(5, 1, -1):
    #     print(f"\n\n\n\nTRAINING MODEL FOR {CURRENT_YEAR}{str(month).zfill(2)}")
    #     main(CURRENT_YEAR, month)
    #     print(f"MODEL FOR {CURRENT_YEAR}-{str(month).zfill(2)} TRAINED AND SAVED\n\n\n\n")