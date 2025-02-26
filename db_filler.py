import os
import pandas as pd
import json
from sqlalchemy import create_engine
from datetime import datetime
import logging
import sys
import traceback

from cars_scraper import cars_scraper


PATH_ABS = os.path.dirname(__file__)
PATH_CONFIG = os.path.join(PATH_ABS, "config")
PATH_CARS_DICT = os.path.join(PATH_CONFIG, "cars_dict.json")
PATH_DB_CREDENTIALS = os.path.join(PATH_CONFIG, "db_credentials.json")
PATH_FOR_LOOP_STATE = os.path.join(PATH_CONFIG, "for_loop_state.json")
PATH_DATA = os.path.join(PATH_ABS, "data")
PATH_CARS_SCRAPED = os.path.join(PATH_DATA, "cars_scraped")
PATH_LOGS = os.path.join(PATH_ABS, "logs", datetime.now().strftime("%Y-%m-%d") + ".log")

# Configue logging
logging.basicConfig(
    filename=PATH_LOGS,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

with open(PATH_CARS_DICT) as f:
    cars = json.load(f)

with open(PATH_DB_CREDENTIALS) as f:
    db_credentials = json.load(f)

def _create_engine_connection(db_credentials: dict):
    return create_engine(
        f"postgresql+psycopg2://{db_credentials['user']}:{db_credentials['password']}@{db_credentials['host']}:{db_credentials['port']}/{db_credentials['dbname']}"
    )

def _save_for_loop_state(initial_manufacturer_index, initial_model_index):
    with open(PATH_FOR_LOOP_STATE, "w") as f:
            json.dump({"initial_manufacturer_index": initial_manufacturer_index, "initial_model_index": initial_model_index}, f)

def _load_for_loop_state(manufacturer_param=None, model_param=None):
    if manufacturer_param and model_param:
        try:
            state = {
                "initial_manufacturer_index": list(cars.keys()).index(manufacturer_param),
                "initial_model_index": cars[manufacturer_param].index(model_param)
            }
        except ValueError as e:
            logging.error(f"Error in _load_for_loop_state: {e}")
            logging.info("Loading initial state (0, 0) for the for loop...")
            state = {"initial_manufacturer_index": 0, "initial_model_index": 0}
    else:
        try:
            with open(PATH_FOR_LOOP_STATE, "r") as f:
                state = json.load(f)
        except FileNotFoundError:
            state = {"initial_manufacturer_index": 0, "initial_model_index": 0}
    return state

def _clean_pipeline(df):

    cleaned_df = df.copy()

    def _clean_str_to_int(value: str) -> int:
        try:
            return int(value)
        except Exception:
            return None

    def _clean_kms(value: str) -> int:
        try:
            return int(value.replace(" km", "").replace(".", ""))
        except Exception:
            return None
        
    def _clean_fuel(value: str) -> str:
        value = value.lower()
        if value == "gasolina":
            return "g"
        elif value in ["diésel", "diesel"]:
            return "d"
        elif value in ["eléctrico", "electrico"]:
            return "e"
        elif value in ["híbrido", "hibrido"]:
            return "h"
        elif value in ["híbrido gasolina", "hibrido gasolina", "micro híbrido gasolina", "micro hibrido gasolina", "híbrido enchufable gasolina", "hibrido enchufable gasolina"]:
            return "hg"
        elif value in ["híbrido diésel", "hibrido diesel", "híbrido diesel", "hibrido diésel", "micro híbrido diésel", "micro hibrido diesel", "micro híbrido diesel", "micro hibrido diesel", "híbrido enchufable diésel", "hibrido enchufable diesel", "híbrido enchufable diesel", "hibrido enchufable diésel"]:
            return "hd"
        else:
            return value
    
    def _clean_transmission(value: str) -> int:
        value = value.lower()
        if value == "manual":
            return "m"
        elif value in ["automático", "automatico", "automática", "automatica"]:
            return "a"
        else:
            return None
        
    def _clean_power_hp(value: str) -> int:
        value = value.lower()
        try:
            return int(value.replace(" cv", ""))
        except Exception:
            return None
    
    def _clean_no_doors(value: str) -> int:
        value = value.lower()
        try:
            return int(value.replace(" puertas", ""))
        except Exception:
            return None
    
    def _clean_color(value: str) -> str:
        return value.lower()
    
    def _clean_seller(value: str) -> str:
        value = value.lower()
        if value == "profesional":
            return "prof"
        elif value == "particular":
            return "part"
        else:
            return None
        
    def _clean_price(value: str|int) -> float:
        try:
            if isinstance(value, int):
                return float(value)
            elif isinstance(value, str):
                return float(value.replace(" €", "").replace(".", "").replace(",", "."))
        except Exception as e:
            print(e)
            return None

    transforms = {
        "month": _clean_str_to_int,
        "year": _clean_str_to_int,
        "kms": _clean_kms,
        "fuel": _clean_fuel,
        "transmission": _clean_transmission,
        "power_hp": _clean_power_hp,
        "no_doors": _clean_no_doors,
        "color": _clean_color,
        "seller": _clean_seller,
        "price_cash": _clean_price,
        "price_financed": _clean_price,
    }

    for col, func in transforms.items():
        cleaned_df[col] = cleaned_df[col].apply(func)

    return cleaned_df


def main(manufacturer_param=None, model_param=None):

    state = _load_for_loop_state(manufacturer_param, model_param)

    initial_manufacturer_index = state["initial_manufacturer_index"]
    manufacturer_list = list(cars.keys())

    for manufacturer_index in range(initial_manufacturer_index, len(manufacturer_list)):
        manufacturer = manufacturer_list[manufacturer_index]
        manuf = manufacturer.replace(" ", "-")
        manuf = manuf.lower()

        models_list = cars[manufacturer]
        initial_model_index = state["initial_model_index"] if manufacturer_index == initial_manufacturer_index else 0

        for model_index in range(initial_model_index, len(models_list)):
            try:
                _save_for_loop_state(manufacturer_index, model_index)
                model = models_list[model_index]
                mod = model.replace(" ", "-")
                mod = mod.lower()
                logging.info(f"Starting data obtention pipeline for {manuf}_{mod}...")
                df = cars_scraper(manuf, mod)
                if df is None:
                    logging.info(f"Data obtention pipeline finished for {manuf}_{mod}.\n")
                    continue
                print(f"\tCleaning {manuf}_{mod} dataframe...")
                df_cleaned = _clean_pipeline(df)
                print(f"\tSaving {manuf}_{mod}.parquet...")
                df_cleaned.to_parquet(os.path.join(PATH_CARS_SCRAPED, f"{manuf}_{mod}.parquet"))
                print(f"\tUploading {manuf}_{mod} to database...")
                engine = _create_engine_connection(db_credentials)
                df_cleaned.to_sql("cars_scraped", engine, if_exists="append", index=False, method="multi")
                logging.info(f"Data obtention pipeline finished for {manuf}_{mod}.\n")
            except Exception as e:
                logging.error(f"Error in {manuf}_{mod}: {e}")
                logging.error(traceback.format_exc().strip())       # This will print the traceback in the log file
                logging.error("\n\n" + "="*80 + "\n")                 # This will separate the different errors in the log file
                raise                                               # This will raise the exception to be detected by systemd to reset the script

        initial_model_index = 0

    os.remove(PATH_FOR_LOOP_STATE)

if __name__ == "__main__":
    manufacturer_param = None
    model_param = None
    main(manufacturer_param, model_param)