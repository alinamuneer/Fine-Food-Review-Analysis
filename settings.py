from pathlib import Path
from dataclasses import dataclass
import json
import sys
from src.azure_pipelines.azure_ml_config import AzureMlConfig

DATA_DIR = Path(__file__).parent /'data/'
DATA_DIR.mkdir(exist_ok=True)

REVIEW_FILE_PATH = DATA_DIR / 'Reviews.csv'

AZURE_ML_CONFIG_FILE = Path(__file__).parent / 'config.json'
CONFIG_AZURE_ML = AzureMlConfig.load_config(AZURE_ML_CONFIG_FILE)