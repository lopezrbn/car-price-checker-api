from pathlib import Path


# Global constants
RANDOM_SEED = 31415

# Define the base directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define the rest of the directories relative to the base directory, and their files

"""CONFIG"""
# Dir
CONFIG_DIR = ROOT_DIR / 'config'
# Files
DB_CREDENTIALS_FILE = CONFIG_DIR / 'db_credentials.json'
EMAIL_CREDENTIALS_FILE = CONFIG_DIR / 'email_credentials.json'
CARS_TO_SCRAPE_FILE = CONFIG_DIR / 'cars_to_scrape.json'

"""DATA"""
DATA_DIR = ROOT_DIR / 'data'

"""LOGS"""
LOGS_DIR = ROOT_DIR / 'logs'

"""MLFLOW"""
# MLRUNS_DIR = ROOT_DIR / 'mlruns'
TRACKING_URI = "http://mlflow.lopezrbn.com"

"""MODELS"""
# Dir
MODELS_DIR = ROOT_DIR / 'models'
# Files
BEST_MODEL_FILE = MODELS_DIR / 'best_model.pkl'

"""PIPELINES"""
PIPELINES_DIR = ROOT_DIR / 'pipelines'