"""
config.py
====================================================
Konfigurasi utama pipeline skripsi K-Means + Decision Tree (Demand Pattern).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_OUT = PROJECT_ROOT / "data_output"
PLOTS_DIR = DATA_OUT / "plots"

DATA_OUT.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILES = [
    (DATA_RAW / "REKAP PENJUALAN ONLINE 2023-2024.xlsx", "REKAP_2023_2024"),
    (DATA_RAW / "REKAP PENJUALAN ONLINE 2025.xlsx", "REKAP_2025"),
]

MASTER_XLSX = DATA_OUT / "01_MASTER_REKAP_PENJUALAN_2023_2025.xlsx"
MASTER_CSV = DATA_OUT / "01_MASTER_REKAP_PENJUALAN_2023_2025.csv"

# [BERUBAH] Nama file output dari RFM diubah jadi DEMAND_FEATURES
DEMAND_FEATURES_XLSX = DATA_OUT / "02_DEMAND_FEATURES.xlsx" 

K_EVAL_XLSX = DATA_OUT / "03_K_EVALUATION_DBI.xlsx"
CLUSTER_XLSX = DATA_OUT / "04_CLUSTERED_LABELED_SKU.xlsx"
CLASSIFICATION_XLSX = DATA_OUT / "05_CLASSIFICATION_DECISION_TREE.xlsx"
RULES_TXT = DATA_OUT / "05_RULES_DECISION_TREE.txt"

KEEP_COLS_RAW = [
    "NO", "MEREK", "NAMA BARANG", "TIPE BARANG", "NO IMEI",
    "TGL TERJUAL", "NAMA PEMBELI", "BELI DARI", "HARGA JUAL"
]

COL_MEREK = "MEREK"
COL_NAMA = "NAMA BARANG"
COL_TIPE = "TIPE BARANG"
COL_IMEI = "NO IMEI"
COL_TGL = "TGL TERJUAL"
COL_HARGA = "HARGA JUAL"

DATE_MIN = "2023-01-01"
DATE_MAX = "2025-12-31"

EXCLUDE_BRANDS = {
    "SULWHASOO", "NEW BALANCE", "NATESH", "RACHEL AJENG",
    "FONTACTIV", "PEPSODENT", "ADLV", "CHAKOLAB"
}
EXCLUDE_PRODUCTS = {"STICKER", "PHOTO", "SKINCARE", "SERUM"}

K_MIN = 2
K_MAX = 4
RANDOM_STATE = 42
N_INIT = 50

# S1 = log1p ADI & CV2 + Z-score (Baseline Terbaik)
FINAL_SCENARIO = "S1"

# [BERUBAH] Kita paksa K=4 agar masuk ke teori 4 Kuadran (Smooth, Erratic, Intermittent, Lumpy)
FINAL_K_OVERRIDE = 4