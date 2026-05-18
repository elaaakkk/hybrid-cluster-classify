"""
02_preprocess_transform_rfm.py
====================================================
TAHAP 2 - Preprocessing transaksi dan transformasi menjadi fitur RFM per SKU.

Input:
- data_output/01_MASTER_REKAP_PENJUALAN_2023_2025.xlsx

Proses:
1. Membaca master transaksi.
2. Membersihkan teks merek, nama barang, tipe barang, dan IMEI.
3. Memfilter periode penelitian 2023-2025.
4. Memfilter produk/brand di luar scope gadget dan aksesoris.
5. Membentuk SKU = MEREK | NAMA BARANG | TIPE BARANG.
6. Dedup transaksi:
   - Jika ada IMEI, dedup berdasarkan IMEI.
   - Jika tidak ada IMEI, transaksi tidak didedup agresif agar order berbeda tidak hilang.
7. Membentuk RFM per SKU:
   - Recency = jarak hari dari tanggal acuan ke tanggal terakhir terjual.
   - Frequency = jumlah transaksi per SKU.
   - Monetary = total harga jual valid per SKU.
8. Membuat 5 skenario transformasi:
   - S1: log1p F & M + Z-score
   - S2: log1p F & M + Min-Max
   - S3: raw F & M + Z-score
   - S4: raw F & M + Min-Max
   - S5: raw F & M + tanpa scaling

Output:
- data_output/02_RFM_FEATURES.xlsx
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts import config as cfg


FEATURE_COLS = ["Recency", "Frequency", "Monetary"]


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
        raise FileNotFoundError(f"Master belum ada: {cfg.MASTER_XLSX}. Jalankan 01_build_master.py dulu.")
    return pd.read_excel(cfg.MASTER_XLSX, sheet_name="MASTER")


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan transaksi sebelum agregasi RFM."""
    required = [cfg.COL_MEREK, cfg.COL_NAMA, cfg.COL_TIPE, cfg.COL_TGL, cfg.COL_HARGA]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ada: {missing}\nKolom tersedia: {list(df.columns)}")

    trx = df.copy()
    before_all = len(trx)

    # 1) Standarisasi teks.
    for col in [cfg.COL_MEREK, cfg.COL_NAMA, cfg.COL_TIPE]:
        trx[col] = clean_text(trx[col])

    if cfg.COL_IMEI in trx.columns:
        trx[cfg.COL_IMEI] = clean_text(trx[cfg.COL_IMEI])

    # 2) Parsing tanggal dan filter periode 2023-2025.
    trx[cfg.COL_TGL] = pd.to_datetime(trx[cfg.COL_TGL], errors="coerce", dayfirst=True)
    trx = trx.dropna(subset=[cfg.COL_MEREK, cfg.COL_NAMA, cfg.COL_TIPE, cfg.COL_TGL]).copy()
    trx = trx[trx[cfg.COL_TGL].between(pd.Timestamp(cfg.DATE_MIN), pd.Timestamp(cfg.DATE_MAX))].copy()

    # 3) Parsing harga. Harga invalid tetap NaN.
    if trx[cfg.COL_HARGA].dtype == "O":
        trx[cfg.COL_HARGA] = (
            trx[cfg.COL_HARGA].astype(str)
                              .str.replace(r"[^\d]", "", regex=True)
                              .replace("", np.nan)
        )
    trx[cfg.COL_HARGA] = pd.to_numeric(trx[cfg.COL_HARGA], errors="coerce")
    trx.loc[trx[cfg.COL_HARGA] <= 0, cfg.COL_HARGA] = np.nan

    # 4) Filter brand/produk di luar scope.
    before_filter = len(trx)
    mask_brand = trx[cfg.COL_MEREK].isin(cfg.EXCLUDE_BRANDS)
    if cfg.EXCLUDE_PRODUCTS:
        pattern = r"(" + "|".join(map(re.escape, cfg.EXCLUDE_PRODUCTS)) + r")"
        mask_product = (
            trx[cfg.COL_NAMA].str.contains(pattern, na=False) |
            trx[cfg.COL_TIPE].str.contains(pattern, na=False)
        )
    else:
        mask_product = False
    trx = trx[~(mask_brand | mask_product)].copy()
    print(f"[02] Filter brand/produk di luar scope: {before_filter - len(trx)} baris")

    # 5) Bentuk SKU sebagai unit analisis produk.
    trx["SKU"] = (
        trx[cfg.COL_MEREK].fillna("") + "|" +
        trx[cfg.COL_NAMA].fillna("") + "|" +
        trx[cfg.COL_TIPE].fillna("")
    )

    # 6) Dedup berbasis IMEI kalau tersedia.
    before_dedup = len(trx)
    if cfg.COL_IMEI in trx.columns:
        has_imei = trx[cfg.COL_IMEI].notna()
        trx_imei = trx[has_imei].drop_duplicates(subset=[cfg.COL_IMEI], keep="first")
        trx_noimei = trx[~has_imei].copy()
        trx = pd.concat([trx_imei, trx_noimei], ignore_index=True)
    else:
        trx = trx.drop_duplicates(subset=["SKU", cfg.COL_TGL, cfg.COL_HARGA], keep="first")

    print(f"[02] Baris awal master: {before_all}")
    print(f"[02] Drop duplikasi preprocessing: {before_dedup - len(trx)} baris")
    print(f"[02] Transaksi bersih: {len(trx)} baris")
    return trx.reset_index(drop=True)


def build_rfm(trx: pd.DataFrame) -> pd.DataFrame:
    """Agregasi transaksi menjadi fitur RFM per SKU."""
    reference_date = trx[cfg.COL_TGL].max() + pd.Timedelta(days=1)

    rfm = (
        trx.groupby("SKU", as_index=False)
           .agg(
               last_sold=(cfg.COL_TGL, "max"),
               first_sold=(cfg.COL_TGL, "min"),
               Frequency_raw=(cfg.COL_TGL, "count"),
               Monetary_raw=(cfg.COL_HARGA, "sum"),
               PriceCount=(cfg.COL_HARGA, "count"),
           )
    )

    rfm["Recency_raw"] = (reference_date - rfm["last_sold"]).dt.days.astype(int)
    rfm.loc[rfm["Recency_raw"] < 0, "Recency_raw"] = 0
    rfm["PriceCoverage"] = rfm["PriceCount"] / rfm["Frequency_raw"]

    # Jika tidak ada harga valid sama sekali, Monetary dianggap unknown, bukan 0.
    rfm.loc[rfm["PriceCount"] == 0, "Monetary_raw"] = np.nan

    # Kolom dasar yang selalu dipakai.
    rfm = rfm[[
        "SKU", "Recency_raw", "Frequency_raw", "Monetary_raw",
        "PriceCount", "PriceCoverage", "first_sold", "last_sold"
    ]].copy()

    print(f"[02] Jumlah SKU hasil RFM: {len(rfm)}")
    print(f"[02] SKU monetary unknown: {rfm['Monetary_raw'].isna().sum()}")
    return rfm


def make_scenario(rfm: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Membuat fitur siap clustering untuk satu skenario.
    Return:
    - data_scenario: data RFM dengan kolom Recency/Frequency/Monetary sesuai transformasi sebelum scaling.
    - scaled_features: fitur akhir yang masuk K-Means.
    """
    data = rfm.copy()
    data["Recency"] = data["Recency_raw"].astype(float)

    if scenario in {"S1", "S2"}:
        data["Frequency"] = np.log1p(data["Frequency_raw"].astype(float))
        data["Monetary"] = np.log1p(data["Monetary_raw"].astype(float))
        transform_note = "log1p Frequency & Monetary"
    elif scenario in {"S3", "S4", "S5"}:
        data["Frequency"] = data["Frequency_raw"].astype(float)
        data["Monetary"] = data["Monetary_raw"].astype(float)
        transform_note = "raw Frequency & Monetary"
    else:
        raise ValueError(f"Scenario tidak dikenal: {scenario}")

    valid = data.dropna(subset=FEATURE_COLS).copy()
    X = valid[FEATURE_COLS].astype(float).values

    if scenario in {"S1", "S3"}:
        scaler = StandardScaler()
        scaling_note = "Z-score"
        X_scaled = scaler.fit_transform(X)
    elif scenario in {"S2", "S4"}:
        scaler = MinMaxScaler()
        scaling_note = "Min-Max"
        X_scaled = scaler.fit_transform(X)
    else:  # S5
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
    print("[02] Mulai preprocessing dan transformasi RFM...")
    raw = load_master()
    trx = preprocess_transactions(raw)
    rfm = build_rfm(trx)

    scenarios = ["S1", "S2", "S3", "S4", "S5"]
    scenario_frames = {}
    scaled_frames = {}

    for sc in scenarios:
        scenario_frames[sc], scaled_frames[sc] = make_scenario(rfm, sc)
        print(f"[02] Skenario {sc}: {scenario_frames[sc]['transform_note'].iloc[0]} + {scenario_frames[sc]['scaling_note'].iloc[0]}")

    with pd.ExcelWriter(cfg.RFM_XLSX, engine="openpyxl") as writer:
        trx.to_excel(writer, index=False, sheet_name="clean_transactions")
        rfm.to_excel(writer, index=False, sheet_name="rfm_raw")
        for sc in scenarios:
            scenario_frames[sc].to_excel(writer, index=False, sheet_name=f"{sc}_rfm_transformed")
            scaled_frames[sc].to_excel(writer, index=False, sheet_name=f"{sc}_scaled")

    print(f"[02] Saved: {cfg.RFM_XLSX}")


if __name__ == "__main__":
    main()
