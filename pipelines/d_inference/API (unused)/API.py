"""
Run this script on terminal with:
    uvicorn API:app --host 0.0.0.0 --port 8000 --reload
"""


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


model = joblib.load("/home/ubuntu/car_price_checker_2/pipelines/best_model.pkl")

# Definir el esquema de entrada
class Coche(BaseModel):
    manufacturer: str
    model: str
    version: str
    month: int
    year: int
    kms: int
    fuel: str
    transmission: str
    power_hp: int
    no_doors: int
    color: str
    seller: str

# Crear la aplicación
app = FastAPI()

# Ruta de prueba
@app.get("/")
def home():
    return {"message": "API working correctly"}

# Ruta de predicción
@app.post("/predict")
def predecir_precio(coche: Coche):
    # Aquí deberías tener el mismo preprocesamiento que usaste al entrenar el modelo
    data = pd.DataFrame(
        [[
            coche.manufacturer,
            coche.model,
            coche.version,
            coche.month,
            coche.year,
            coche.kms,
            coche.fuel,
            coche.transmission,
            coche.power_hp,
            coche.no_doors,
            coche.color,
            coche.seller
        ]],
        columns=["manufacturer", "model", "version", "month", "year", "kms", "fuel", "transmission", "power_hp", "no_doors", "color", "seller"]
    )
    # Suponiendo que tienes un pipeline que maneja la codificación y escalado
    prediction = model.predict(data)[0]
    return {"Predicted price": float(prediction)}