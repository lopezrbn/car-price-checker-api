import config.paths as paths
from pipelines.b_preprocessing.preprocessing_pipeline import PreprocessingPipeline

import pandas as pd
import numpy as np
import json
import joblib
from sqlalchemy import create_engine
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, r2_score


def create_engine_connection(db_credentials: dict):
    return create_engine(
        f"postgresql+psycopg2://{db_credentials['user']}:{db_credentials['password']}@{db_credentials['host']}:{db_credentials['port']}/{db_credentials['dbname']}"
    )

def evaluate_and_select_best_model(df, preproc_pipeline, scoring="r2"):
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
    X = df.drop(columns=["price_cash"])
    y = df["price_cash"]

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

    
if __name__ == "__main__":
    with paths.DB_CREDENTIALS_FILE.open("r") as f:
        db_credentials = json.load(f)

    print("Loading data from database...")
    current_month = datetime.now().month
    query = f"SELECT * FROM public.cars_scraped WHERE EXTRACT(MONTH FROM created_at) = {current_month};"
    engine = create_engine_connection(db_credentials)
    df = pd.read_sql(query, engine)
    print("\tData loaded\n")

    print("Loading preprocessing pipeline...")
    preproc = PreprocessingPipeline()
    preproc_pipeline = preproc.create_pipeline(df)
    print("\tPreprocessing pipeline loaded\n")

    print("Evaluating models...")
    best_model, best_score, search_obj = evaluate_and_select_best_model(df, preproc_pipeline=preproc_pipeline)
    print("\tModels evaluated\n")
    
    print("Best score:")
    print(best_score)
    print("Best model:")
    print(best_model)

    print("Saving best model...")
    joblib.dump(best_model, paths.BEST_MODEL_FILE)
    print("\tBest model saved\n")