"""
04_cluster_label_interpret.py
====================================================
TAHAP 4 - Clustering final berbasis ADI & CV2, dan penentuan 4 Kuadran Manajerial.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts import config as cfg

#  Hanya pakai fitur demand
FEATURE_COLS = ["ADI", "CV2"]
FEATURE_COLS_SCALED = ["scaled_ADI", "scaled_CV2"]

def load_final_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    #  Membaca dari DEMAND_FEATURES_XLSX
    transformed = pd.read_excel(cfg.DEMAND_FEATURES_XLSX, sheet_name=f"{cfg.FINAL_SCENARIO}_transformed")
    scaled = pd.read_excel(cfg.DEMAND_FEATURES_XLSX, sheet_name=f"{cfg.FINAL_SCENARIO}_scaled")
    return transformed, scaled

def build_cluster_profile(clustered: pd.DataFrame) -> pd.DataFrame:
    """Membentuk profil cluster (rata-rata ADI, CV2, dan Revenue)."""
    profile = (
        clustered.groupby("cluster")
                 .agg(
                     n_sku=("SKU", "count"),
                     ADI_mean=("ADI", "mean"),
                     CV2_mean=("CV2", "mean"),
                     Total_Revenue_mean=("Total_Revenue", "mean") # Opsional untuk tahu omset cluster
                 )
                 .reset_index()
    )
    return profile

def assign_managerial_labels(profile: pd.DataFrame) -> pd.DataFrame:
    """Memberi 4 label kuadran murni berdasarkan pola ADI & CV2."""
    p = profile.copy()
    
    # 1. Urutkan cluster dari ADI terkecil (Paling rutin) ke terbesar (Paling jarang)
    p_sorted = p.sort_values("ADI_mean").reset_index(drop=True)
    
    # Karena FINAL_K_OVERRIDE = 4, kita asumsikan jumlah cluster pasti 4
    # Bagi dua kelompok: ADI rendah (Rutinan) dan ADI tinggi (Jarang)
    grup_rutin = p_sorted.iloc[:2].sort_values("CV2_mean").reset_index(drop=True)
    grup_jarang = p_sorted.iloc[2:].sort_values("CV2_mean").reset_index(drop=True)
    
    # 2. Eksekusi Kuadran (1-to-1 Mapping)
    grup_rutin["segment_label"] = ["Smooth", "Erratic"]
    grup_rutin["interpretation"] = [
        "Sering laku & jumlah stabil. Cocok untuk restock rutin.",
        "Sering laku tapi jumlah bergejolak. Sediakan safety stock ekstra."
    ]
    
    grup_jarang["segment_label"] = ["Intermittent", "Lumpy"]
    grup_jarang["interpretation"] = [
        "Jarang laku tapi jumlah stabil. Pesan Just-in-Time.",
        "Sangat jarang & acak. Prioritas Pre-Order (PO), hindari stok gudang."
    ]
    
    # 3. Gabung kembali
    p_final = pd.concat([grup_rutin, grup_jarang])
    p = p.merge(p_final[["cluster", "segment_label", "interpretation"]], on="cluster", how="left")
    
    return p

def main():
    print("[04] Mulai clustering K-Means (Pola Permintaan)...")
    transformed, scaled = load_final_data()
    k_final = cfg.FINAL_K_OVERRIDE # Pasti 4

    # Proses KMeans
    X = scaled[FEATURE_COLS_SCALED].astype(float).values
    km = KMeans(n_clusters=k_final, random_state=cfg.RANDOM_STATE, n_init=cfg.N_INIT)
    labels = km.fit_predict(X)

    # Gabungkan label
    valid_idx = transformed.dropna(subset=FEATURE_COLS).index
    clustered = transformed.loc[valid_idx].copy().reset_index(drop=True)
    clustered["cluster"] = labels

    profile = build_cluster_profile(clustered)
    profile = assign_managerial_labels(profile)

    label_map = profile.set_index("cluster")["segment_label"].to_dict()
    clustered["segment_label"] = clustered["cluster"].map(label_map)

    with pd.ExcelWriter(cfg.CLUSTER_XLSX, engine="openpyxl") as writer:
        clustered.to_excel(writer, index=False, sheet_name="clustered_labeled_SKU")
        profile.to_excel(writer, index=False, sheet_name="cluster_profile_labeled")

    print("[04] Profil cluster berlabel:")
    print(profile[["cluster", "n_sku", "ADI_mean", "CV2_mean", "segment_label"]])
    print(f"[04] Saved: {cfg.CLUSTER_XLSX}")

if __name__ == "__main__":
    main()