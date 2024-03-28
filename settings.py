from pathlib import Path

DATA_DIR = Path(__file__).parent /'data/'
DATA_DIR.mkdir(exist_ok=True)

REVIEW_FILE_PATH = DATA_DIR / 'Reviews.csv'
