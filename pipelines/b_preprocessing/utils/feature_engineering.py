from pathlib import Path

import pandas as pd
import numpy as np
import json
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

PATH_THIS_FILE = Path(__file__).resolve()
PATH_MODELS_VS_SEGMENT_FILE = PATH_THIS_FILE.parent / "models_vs_segment.json"
PATH_MANUF_DETAILS_FILE = PATH_THIS_FILE.parent / "manufacturers_details.json"
# PATH_MODELS_VS_SEGMENT = "/home/ubuntu/car_price_checker_2/config/models_vs_segment.json"
# PATH_MANUF_DETAILS = "/home/ubuntu/car_price_checker_2/config/manufacturers_details.json"

class CustomFeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No need to fit, so we return self
        return self

    def transform(self, X, y=None):
        # Copy the dataframe to avoid modifying the original one
        X = X.copy()

        # Create feature "age"
        current_year = datetime.now().year + datetime.now().month / 12
        X['age'] = np.maximum(0, current_year - (X["year"] + X["month"] / 12))

        # Create feature "age_bins"
        X["age_bins"] = pd.cut(X["age"], bins=range(0, 61, 3), labels=[f"[{i}-{i+3})" for i in range(0, 60, 3)], right=False)

        # Create feature "kms_per_year"
        X["kms_per_year"] = X["kms"] / (X["age"] + 1/12)

        # Create feature "kms_per_year_bins"
        X["kms_per_year_bins"] = pd.cut(X["kms_per_year"], bins=[0, 10000, 20000, 10000000], labels=["low", "medium", "high"], right=False)

        # # Create feature "avg_price_model"
        # X["avg_price_model"] = X.groupby("model")["price_cash"].transform("mean")

        # Create feature "model_segment"
        with open(PATH_MODELS_VS_SEGMENT_FILE, "r") as f:
            models_vs_segment = json.load(f)
        X["model_segment"] = X["model"].map(models_vs_segment)
        X["model_segment"] = X["model_segment"].fillna("Unknown")
        X["model_segment"] = X["model_segment"].str.lower()

        # Create feature "manuf_rel_freq" as a proxy of the popularity of the brand
        X["manuf_rel_freq"] = X["manufacturer"].map(X["manufacturer"].value_counts(normalize=True))

        # Create feature "model_rel_freq" as a proxy of the popularity of the model
        X["model_rel_freq"] = X["model"].map(X["model"].value_counts(normalize=True))

        # Create feature "model_segment_rel_freq" as a proxy of the popularity of the model segment
        X["model_segment_rel_freq"] = X["model_segment"].map(X["model_segment"].value_counts(normalize=True))

        # # Create feature "model_segment_avg_price"
        # X["model_segment_avg_price"] = X.groupby("model_segment")["price_cash"].transform("mean")

        # Create feature "model_segment_avg_kms"
        X["model_segment_avg_kms"] = X.groupby("model_segment")["kms"].transform("mean")

        # Create feature "model_segment_avg_age"
        X["model_segment_avg_age"] = X.groupby("model_segment")["age"].transform("mean")

        # Create feature "model_segment_avg_manuf_rel_freq"
        X["model_segment_avg_manuf_rel_freq"] = X.groupby("model_segment")["manuf_rel_freq"].transform("mean")

        # Create feature "model_segment_avg_model_rel_freq"
        X["model_segment_avg_model_rel_freq"] = X.groupby("model_segment")["model_rel_freq"].transform("mean")

        # Create feature "manuf_country"
        with open(PATH_MANUF_DETAILS_FILE, "r") as f:
            manuf_details = json.load(f)
        X["manuf_country"] = X["manufacturer"].map(lambda x: manuf_details.get(x, {}).get("country", "unknown"))
        X["manuf_country"] = X["manuf_country"].fillna("unknown")
        X["manuf_country"] = X["manuf_country"].str.lower()

        # # Create feature "manuf_country_avg_price"
        # X["manuf_country_avg_price"] = X.groupby("manuf_country")["price_cash"].transform("mean")

        # Create feature "manuf_country_avg_kms"
        X["manuf_country_avg_kms"] = X.groupby("manuf_country")["kms"].transform("mean")

        # Create feature "manuf_country_avg_age"
        X["manuf_country_avg_age"] = X.groupby("manuf_country")["age"].transform("mean")

        # Create feature "exclusivity_level"
        X["exclusivity_level"] = X["manufacturer"].map(lambda x: manuf_details.get(x, {}).get("exclusivity_level", "unknown"))
        X["exclusivity_level"] = X["exclusivity_level"].fillna("unknown")
        X["exclusivity_level"] = X["exclusivity_level"].str.lower()

        # # Create feature "exclusivity_level_avg_price"
        # X["exclusivity_level_avg_price"] = X.groupby("exclusivity_level")["price_cash"].transform("mean")

        # Create feature "exclusivity_level_avg_kms"
        X["exclusivity_level_avg_kms"] = X.groupby("exclusivity_level")["kms"].transform("mean")

        # Create feature "exclusivity_level_avg_age"
        X["exclusivity_level_avg_age"] = X.groupby("exclusivity_level")["age"].transform("mean")

        # Create feature "manuf_group"
        X["manuf_group"] = X["manufacturer"].map(lambda x: manuf_details.get(x, {}).get("group", "unknown"))
        X["manuf_group"] = X["manuf_group"].fillna("unknown")
        X["manuf_group"] = X["manuf_group"].str.lower().str.replace(" ", "_").str.replace("-", "_")

        return X
