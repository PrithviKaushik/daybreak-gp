from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SOLETE_PARQUET = RAW_DATA_DIR / "SOLETE_Pombo_60min_v4.parquet"
