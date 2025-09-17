import config.paths as paths

import pandas as pd
import joblib
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

def make_prediction(data):
    """
    Make a prediction using the trained model.
    """

    # Load the model
    model_path = find_best_model()
    model = joblib.load(model_path)

    # Ensure data is in the correct format
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, index=[0])
    
    # Make prediction
    prediction = model.predict(data)[0]
    
    return prediction


if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        "manufacturer": "audi",
        "model": "a4",
        "month": 11,
        "year": 2017,
        "kms": 130000,
        "fuel": "d",
        "transmission": "m",
        "power_hp": 150,
    }, index=[0])
    
    prediction = make_prediction(data)
    print(f"Predicted price: {prediction:.2f} €")