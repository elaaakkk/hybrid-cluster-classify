"""
02_preprocess_transform_demand.py
====================================================
TAHAP 2 - Preprocessing transaksi dan transformasi menjadi fitur ADI & CV2 per SKU.
"""

import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts import config as cfg

# [BERUBAH] Fitur utama HANYA ADI dan CV2
FEATURE_COLS = ["ADI", "CV2"]

def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan, "-": np.nan, "—": np.nan})
    )
def load_master() -> pd.DataFrame:
    if not cfg.MASTER_XLSX.exists():
        raise FileNotFoundError(f"Master belum ada: {cfg.MASTER_XLSX}")
    return pd.read_excel(cfg.MASTER_XLSX, sheet_name="MASTER")

def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    trx = df.copy()
    before_all = len(trx) # Menambahkan definisi before_all

    for col in [cfg.COL_MEREK, cfg.COL_NAMA, cfg.COL_TIPE]:
        trx[col] = clean_text(trx[col])
    if cfg.COL_IMEI in trx.columns:
        trx[cfg.COL_IMEI] = clean_text(trx[cfg.COL_IMEI])

    trx[cfg.COL_TGL] = pd.to_datetime(trx[cfg.COL_TGL], errors="coerce", dayfirst=True)
    trx = trx.dropna(subset=[cfg.COL_MEREK, cfg.COL_NAMA, cfg.COL_TIPE, cfg.COL_TGL]).copy()
    trx = trx[trx[cfg.COL_TGL].between(pd.Timestamp(cfg.DATE_MIN), pd.Timestamp(cfg.DATE_MAX))]

    if trx[cfg.COL_HARGA].dtype == "O":
        trx[cfg.COL_HARGA] = trx[cfg.COL_HARGA].astype(str).str.replace(r"[^\d]", "", regex=True).replace("", np.nan)
    trx[cfg.COL_HARGA] = pd.to_numeric(trx[cfg.COL_HARGA], errors="coerce")
    trx.loc[trx[cfg.COL_HARGA] <= 0, cfg.COL_HARGA] = np.nan

    # Menghilangkan tanda kurung () pada pola regex agar tidak muncul UserWarning
    mask_brand = trx[cfg.COL_MEREK].isin(cfg.EXCLUDE_BRANDS)
    if cfg.EXCLUDE_PRODUCTS:
        pattern = "|".join(map(re.escape, cfg.EXCLUDE_PRODUCTS))
        mask_product = trx[cfg.COL_NAMA].str.contains(pattern, na=False, regex=True) | trx[cfg.COL_TIPE].str.contains(pattern, na=False, regex=True)
    else:
        mask_product = False
        
    trx = trx[~(mask_brand | mask_product)].copy()

    trx["SKU"] = trx[cfg.COL_MEREK].fillna("") + "|" + trx[cfg.COL_NAMA].fillna("") + "|" + trx[cfg.COL_TIPE].fillna("")

    if cfg.COL_IMEI in trx.columns:
        has_imei = trx[cfg.COL_IMEI].notna()
        trx = pd.concat([trx[has_imei].drop_duplicates(subset=[cfg.COL_IMEI], keep="first"), trx[~has_imei]], ignore_index=True)
    else:
        trx = trx.drop_duplicates(subset=["SKU", cfg.COL_TGL, cfg.COL_HARGA], keep="first")

    print(f"[02] Baris awal master: {before_all}")
    print(f"[02] Drop duplikasi & filter: {before_all - len(trx)} baris")
    print(f"[02] Transaksi bersih: {len(trx)} baris")
    
    return trx.reset_index(drop=True)

def build_demand_features(trx: pd.DataFrame) -> pd.DataFrame:
    """Menghitung metrik ADI dan CV2 murni untuk setiap SKU."""
    total_days = (pd.Timestamp(cfg.DATE_MAX) - pd.Timestamp(cfg.DATE_MIN)).days + 1
    #buat kolom priode bulan tahun untuk transaksi
    trx["YearMonth"] = trx [cfg.COL_TGL].dt.to_period("M")
    
    #hitung total bulan kalender dari DATE_MIN sampai DATE_MAX
    total_months = len(pd.period_range(start=cfg.DATE_MIN, end=cfg.DATE_MAX, freq="M"))

    #hitung berapa bulan SKU ini laku, lalu cari presentase bulan kosongnya
    monthly_active = trx.groupby("SKU")["YearMonth"].nunique().reset_index(name="active_months")
    monthly_active["zero_month_ratio"] = 1.0 - (monthly_active["active_months"] / total_months)
    monthly_active["zero_month_ratio"] = monthly_active["zero_month_ratio"].clip(lower=0.0)

    trx.drop(columns=["YearMonth"], inplace=True)

    # 1. Agregasi penjualan harian (Kuantitas harian per barang)
    daily_trx = trx.groupby(["SKU", cfg.COL_TGL]).size().reset_index(name="daily_qty")
    
    # 2. Hitung statistik permintaan
    demand_df = daily_trx.groupby("SKU").agg(
        active_days=("daily_qty", "count"),
        qty_mean=("daily_qty", "mean"),
        qty_std=("daily_qty", "std")
    ).reset_index()

    # 3. Rumus Inti ADI & CV2
    demand_df["ADI"] = total_days / demand_df["active_days"]
    demand_df["qty_std"] = demand_df["qty_std"].fillna(0)
    demand_df["CV2"] = (demand_df["qty_std"] / (demand_df["qty_mean"] + 1e-9)) ** 2

    # 4. (Opsional) Tambahkan total penjualan untuk reporting 
    sales = trx.groupby("SKU").agg(
        Total_Terjual=(cfg.COL_HARGA, "count"),
        Total_Revenue=(cfg.COL_HARGA, "sum")
    ).reset_index()

    demand_df = pd.merge(demand_df, sales, on="SKU", how="left")

    #gabung fitur zmr ke demand_df
    demand_df = pd.merge(demand_df, monthly_active[["SKU", "zero_month_ratio"]], on="SKU", how="left")
                         
    print(f"[02] Jumlah SKU diproses: {len(demand_df)}")
    return demand_df

def make_scenario(demand_df: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Skenario transformasi untuk menormalkan distribusi ADI & CV2."""
    data = demand_df.copy()
    valid = data.dropna(subset=FEATURE_COLS).copy()
    
    if scenario in {"S1", "S2"}:
        X = np.log1p(valid[FEATURE_COLS].astype(float).values)
        transform_note = "log1p ADI & CV2"
    else:
        X = valid[FEATURE_COLS].astype(float).values
        transform_note = "raw ADI & CV2"

    if scenario in {"S1", "S3"}:
        scaler = StandardScaler()
        scaling_note = "Z-score"
        X_scaled = scaler.fit_transform(X)
    elif scenario in {"S2", "S4"}:
        scaler = MinMaxScaler()
        scaling_note = "Min-Max"
        X_scaled = scaler.fit_transform(X)
    else:
        scaling_note = "tanpa scaling"
        X_scaled = X

    scaled = pd.DataFrame(X_scaled, columns=[f"scaled_{c}" for c in FEATURE_COLS], index=valid.index)
    scaled.insert(0, "SKU", valid["SKU"].values)
    scaled["scenario"] = scenario
    scaled["transform_note"] = transform_note
    scaled["scaling_note"] = scaling_note

    data["scenario"] = scenario
    data["transform_note"] = transform_note
    data["scaling_note"] = scaling_note
    return data, scaled

def main():
    print("[02] Mulai preprocessing dan ekstraksi Demand Features (ADI & CV2)...")
    trx = preprocess_transactions(load_master())
    demand_df = build_demand_features(trx)

    scenarios = ["S1", "S2", "S3", "S4", "S5"]
    scenario_frames, scaled_frames = {}, {}

    for sc in scenarios:
        scenario_frames[sc], scaled_frames[sc] = make_scenario(demand_df, sc)

    # [BERUBAH] Simpan ke DEMAND_FEATURES_XLSX bukan RFM_XLSX
    with pd.ExcelWriter(cfg.DEMAND_FEATURES_XLSX, engine="openpyxl") as writer:
        trx.to_excel(writer, index=False, sheet_name="clean_transactions")
        demand_df.to_excel(writer, index=False, sheet_name="demand_features_raw")
        for sc in scenarios:
            scenario_frames[sc].to_excel(writer, index=False, sheet_name=f"{sc}_transformed")
            scaled_frames[sc].to_excel(writer, index=False, sheet_name=f"{sc}_scaled")
    print(f"[02] Saved: {cfg.DEMAND_FEATURES_XLSX}")

if __name__ == "__main__":
    main()