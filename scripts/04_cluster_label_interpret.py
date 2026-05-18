"""
04_cluster_label_interpret.py
====================================================
TAHAP 4 - Clustering final, labeling cluster, dan interpretasi hasil.

Input:
- data_output/02_RFM_FEATURES.xlsx
- data_output/03_K_EVALUATION_DBI.xlsx

Proses:
1. Membaca skenario final dari config.FINAL_SCENARIO.
2. Mengambil K final:
   - Jika FINAL_K_OVERRIDE diisi, pakai nilai itu.
   - Jika None, pakai K dengan DBI terbaik dari skenario final.
3. Menjalankan K-Means final.
4. Membentuk profil cluster berdasarkan nilai RFM raw dan transformed.
5. Mengurutkan cluster berdasarkan skor aktivitas demand:
   - Recency rendah = lebih aktif.
   - Frequency tinggi = lebih aktif.
   - Monetary tinggi = lebih aktif.
6. Memberi label manajerial:
   - Jika K=3: fast, medium, slow.
   - Jika K>3 dan MAP_TO_3_MANAGERIAL_LABELS=True: cluster teratas = fast, terbawah = slow, sisanya = medium.
7. Membuat rekomendasi interpretasi aksi yang aman, tanpa menentukan jumlah restock.

Output:
- data_output/04_CLUSTERED_LABELED_SKU.xlsx
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts import config as cfg


FEATURE_COLS = ["Recency", "Frequency", "Monetary", "ADI", "CV2"]
FEATURE_COLS_SCALED = ["scaled_Recency", "scaled_Frequency", "scaled_Monetary", "scaled_ADI", "scaled_CV2"]


def get_final_k() -> int:
    """Mengambil K final dari override atau hasil evaluasi DBI."""
    if cfg.FINAL_K_OVERRIDE is not None:
        print(f"[04] K final memakai override manual: K={cfg.FINAL_K_OVERRIDE}")
        return int(cfg.FINAL_K_OVERRIDE)

    if not cfg.K_EVAL_XLSX.exists():
        raise FileNotFoundError(f"File evaluasi K belum ada: {cfg.K_EVAL_XLSX}. Jalankan 03_determine_k_dbi.py dulu.")

    best_by_scenario = pd.read_excel(cfg.K_EVAL_XLSX, sheet_name="best_k_by_scenario")
    row = best_by_scenario[best_by_scenario["scenario"] == cfg.FINAL_SCENARIO]
    if row.empty:
        raise ValueError(f"Skenario {cfg.FINAL_SCENARIO} tidak ditemukan di evaluasi K.")

    k_final = int(row.iloc[0]["k"])
    print(f"[04] K final dari DBI terbaik skenario {cfg.FINAL_SCENARIO}: K={k_final}")
    return k_final


def load_final_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not cfg.RFM_XLSX.exists():
        raise FileNotFoundError(f"File RFM belum ada: {cfg.RFM_XLSX}. Jalankan 02_preprocess_transform_rfm.py dulu.")

    transformed = pd.read_excel(cfg.RFM_XLSX, sheet_name=f"{cfg.FINAL_SCENARIO}_rfm_transformed")
    scaled = pd.read_excel(cfg.RFM_XLSX, sheet_name=f"{cfg.FINAL_SCENARIO}_scaled")
    return transformed, scaled


def run_final_kmeans(scaled: pd.DataFrame, k_final: int):
    X = scaled[FEATURE_COLS_SCALED].astype(float).values
    km = KMeans(n_clusters=k_final, random_state=cfg.RANDOM_STATE, n_init=cfg.N_INIT)
    labels = km.fit_predict(X)

    metrics = pd.DataFrame([{
        "scenario": cfg.FINAL_SCENARIO,
        "k_used": k_final,
        "Davies_Bouldin": float(davies_bouldin_score(X, labels)),
        "Silhouette": float(silhouette_score(X, labels)),
        "Calinski_Harabasz": float(calinski_harabasz_score(X, labels)),
        "WSS_inertia": float(km.inertia_),
        "n_sku_clustered": int(len(scaled)),
    }])

    centers_scaled = pd.DataFrame(km.cluster_centers_, columns=FEATURE_COLS_SCALED)
    centers_scaled.insert(0, "cluster", range(k_final))
    return labels, metrics, centers_scaled


def minmax_rank_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Skor 0-1 untuk menyusun tingkat aktivitas cluster."""
    s = series.astype(float)
    if s.max() == s.min():
        scaled = pd.Series(0.5, index=s.index)
    else:
        scaled = (s - s.min()) / (s.max() - s.min())
    return scaled if higher_is_better else 1 - scaled


def build_cluster_profile(clustered: pd.DataFrame) -> pd.DataFrame:
    """Membentuk profil cluster dari rata-rata RFM, ADI, CV2 dan jumlah SKU."""
    profile = (
        clustered.groupby("cluster")
                 .agg(
                     n_sku=("SKU", "count"),
                     Recency_mean=("Recency_raw", "mean"),
                     Frequency_mean=("Frequency_raw", "mean"),
                     Monetary_mean=("Monetary_raw", "mean"),
                     ADI_mean=("ADI", "mean"),             # TAMBAHAN
                     CV2_mean=("CV2", "mean"),             # TAMBAHAN
                     Recency_median=("Recency_raw", "median"),
                     Frequency_median=("Frequency_raw", "median"),
                     Monetary_median=("Monetary_raw", "median"),
                     ADI_median=("ADI", "median"),         # TAMBAHAN
                     CV2_median=("CV2", "median"),         # TAMBAHAN
                     last_sold_min=("last_sold", "min"),
                     last_sold_max=("last_sold", "max"),
                 )
                 .reset_index()
    )

    # Skor aktivitas demand: R rendah + F tinggi + M tinggi.
    profile["score_recency"] = minmax_rank_score(profile["Recency_mean"], higher_is_better=False)
    profile["score_frequency"] = minmax_rank_score(profile["Frequency_mean"], higher_is_better=True)
    profile["score_monetary"] = minmax_rank_score(profile["Monetary_mean"], higher_is_better=True)
    profile["demand_activity_score"] = (
        profile["score_recency"] + profile["score_frequency"] + profile["score_monetary"]
    ) / 3

    profile = profile.sort_values("demand_activity_score", ascending=False).reset_index(drop=True)
    profile["activity_rank"] = np.arange(1, len(profile) + 1)
    return profile


def assign_managerial_labels(profile: pd.DataFrame) -> pd.DataFrame:
    """Memberi label adaptif berdasarkan pola ADI & CV2 secara komparatif."""
    p = profile.copy()
    k = len(p)
    
    # 1. Urutkan cluster dari ADI terkecil (paling rutin) ke terbesar (paling jarang)
    p_sorted = p.sort_values("ADI_mean").reset_index(drop=True)
    
    if k == 2:
        p_sorted["segment_label"] = ["Routine Demand", "Lumpy Demand"]
        p_sorted["interpretation"] = [
            "Aktivitas tinggi & rutin. Terapkan Continuous Replenishment.",
            "Aktivitas rendah & acak. Terapkan Pre-Order (PO) atau Dropship."
        ]
    elif k == 3:
        p_sorted["segment_label"] = ["Routine Demand", "Intermittent Demand", "Lumpy Demand"]
        p_sorted["interpretation"] = [
            "Aktivitas tinggi & rutin. Terapkan Continuous Replenishment.",
            "Jarang laku tapi stabil. Terapkan pemesanan Just-in-Time.",
            "Aktivitas rendah & acak. Terapkan Pre-Order (PO) atau Dropship."
        ]
    elif k == 4:
        # Bagi dua kelompok berdasarkan ADI, lalu urutkan masing-masing berdasarkan CV2 (kestabilan jumlah)
        grup_rendah = p_sorted.iloc[:2].sort_values("CV2_mean").reset_index(drop=True)
        grup_tinggi = p_sorted.iloc[2:].sort_values("CV2_mean").reset_index(drop=True)
        
        grup_rendah["segment_label"] = ["Smooth", "Erratic"]
        grup_rendah["interpretation"] = [
            "Rutin & stabil. Stok otomatis secukupnya.",
            "Rutin tapi bergejolak. Sediakan safety stock ekstra."
        ]
        
        grup_tinggi["segment_label"] = ["Intermittent", "Lumpy"]
        grup_tinggi["interpretation"] = [
            "Jarang laku tapi stabil. Terapkan pemesanan Just-in-Time.",
            "Paling jarang & acak. Risiko deadstock tinggi, gunakan PO."
        ]
        
        p_sorted = pd.concat([grup_rendah, grup_tinggi])
    else:
        p_sorted["segment_label"] = [f"Segment-{i+1}" for i in range(k)]
        p_sorted["interpretation"] = "Segmen teknis."
        
    # 2. Gabungkan kembali ke urutan index aslinya menggunakan 'cluster' ID
    p = p.merge(p_sorted[["cluster", "segment_label", "interpretation"]], on="cluster", how="left")
    
    return p


def main():
    print("[04] Mulai clustering final dan labeling interpretasi...")
    transformed, scaled = load_final_data()
    k_final = get_final_k()

    labels, metrics, centers_scaled = run_final_kmeans(scaled, k_final)

    # Gabungkan fitur transformed/raw dengan label cluster.
    valid_idx = transformed.dropna(subset=FEATURE_COLS).index
    clustered = transformed.loc[valid_idx].copy().reset_index(drop=True)
    clustered["cluster"] = labels

    profile = build_cluster_profile(clustered)
    profile = assign_managerial_labels(profile)

    # Mapping label ke data SKU.
    label_map = profile.set_index("cluster")["segment_label"].to_dict()
    rank_map = profile.set_index("cluster")["activity_rank"].to_dict()
    clustered["segment_label"] = clustered["cluster"].map(label_map)
    clustered["activity_rank"] = clustered["cluster"].map(rank_map)

    # SKU yang tidak ikut clustering karena monetary unknown.
    all_transformed = transformed.copy()
    unknown = all_transformed[all_transformed["Monetary"].isna()].copy()
    if len(unknown) > 0:
        unknown["cluster"] = -1
        unknown["segment_label"] = "unknown-price"
        unknown["activity_rank"] = np.nan

    with pd.ExcelWriter(cfg.CLUSTER_XLSX, engine="openpyxl") as writer:
        clustered.to_excel(writer, index=False, sheet_name="clustered_labeled_SKU")
        profile.to_excel(writer, index=False, sheet_name="cluster_profile_labeled")
        metrics.to_excel(writer, index=False, sheet_name="summary_metrics")
        centers_scaled.to_excel(writer, index=False, sheet_name="centers_scaled")
        if len(unknown) > 0:
            unknown.to_excel(writer, index=False, sheet_name="unknown_price_SKU")

    print("[04] Summary metrics:")
    print(metrics)
    print("[04] Profil cluster berlabel:")
    print(profile[["cluster", "n_sku", "Recency_mean", "Frequency_mean", "Monetary_mean", "demand_activity_score", "segment_label"]])
    print(f"[04] Saved: {cfg.CLUSTER_XLSX}")


if __name__ == "__main__":
    main()
