"""
Run this script on terminal with:
    uvicorn API:app --host 0.0.0.0 --port 8000 --reload
"""

from config import paths as paths

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime


CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month


def find_best_model(current_year=CURRENT_YEAR, current_month=CURRENT_MONTH):
    model_path = paths.MODELS_DIR / f"{current_year}{str(current_month).zfill(2)}_best_model.pkl"
    if model_path.exists():
        print(f"Best model found for {current_year}{str(current_month).zfill(2)}")
        return model_path
    else:
        print(f"No model found for {current_year}{str(current_month).zfill(2)}. Checking previous months...")
        current_month -= 1
        if current_month == 0:
            current_month = 12
            current_year -= 1
            if current_year < 2025:
                raise ValueError("No model available before 2025.")
        return find_best_model(current_year, current_month)

# Load the model
model_path = find_best_model()
model = joblib.load(model_path)

# Define the entrypoint for the API
class Car(BaseModel):
    manufacturer: str
    model: str
    # version: str
    month: int
    year: int
    kms: int
    fuel: str
    transmission: str
    power_hp: int
    # no_doors: int
    # color: str
    # seller: str

# Create the FastAPI app
app = FastAPI()

# Test route
@app.get("/")
def home():
    return {"message": "API working correctly"}

# Predict route
@app.post("/predict")
def predict_price(car: Car):
    data = pd.DataFrame(
        [[
            car.manufacturer,
            car.model,
            # car.version,
            car.month,
            car.year,
            car.kms,
            car.fuel,
            car.transmission,
            car.power_hp,
            # car.no_doors,
            # car.color,
            # car.seller
        ]],
        columns=["manufacturer",
                 "model",
                #  "version",
                 "month",
                 "year",
                 "kms",
                 "fuel",
                 "transmission",
                 "power_hp",
                #  "no_doors",
                #  "color",
                #  "seller"
        ]
    )
    prediction = model.predict(data)[0]
    return {"Predicted price": float(prediction)}