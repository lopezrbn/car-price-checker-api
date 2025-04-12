import config.paths as paths

import pandas as pd
import joblib


def make_prediction(data):
    """
    Make a prediction using the trained model.
    """

    # Load the model
    model = joblib.load(paths.BEST_MODEL_FILE)

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