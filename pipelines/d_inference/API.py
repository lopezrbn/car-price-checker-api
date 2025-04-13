"""
Run this script on terminal with:
    uvicorn API:app --host 0.0.0.0 --port 8000 --reload
"""

from config import paths as paths

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


# Load the model
model = joblib.load(paths.BEST_MODEL_FILE)

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