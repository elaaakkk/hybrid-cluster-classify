"""
04_cluster_label_interpret.py
====================================================
TAHAP 4 - Clustering final berbasis ADI & CV2,
dan penentuan label 4 pola permintaan:
Smooth, Erratic, Intermittent, dan Lumpy.

Catatan penting:
- Clustering dilakukan menggunakan fitur scaled_ADI dan scaled_CV2.
- Interpretasi label dilakukan menggunakan nilai asli ADI dan CV2.
- Label diberikan berdasarkan posisi relatif centroid/profil cluster,
  bukan berdasarkan threshold absolut ADI=1.32 dan CV2=0.49.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts import config as cfg


# Kolom identitas SKU
SKU_COL = "SKU"

# Fitur asli untuk interpretasi
FEATURE_COLS = ["ADI", "CV2"]

# Fitur scaled untuk K-Means
FEATURE_COLS_SCALED = ["scaled_ADI", "scaled_CV2"]


def require_columns(df: pd.DataFrame, required_cols: list[str], df_name: str) -> None:
    """
    Mengecek apakah kolom yang dibutuhkan tersedia di dataframe.
    """
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Kolom berikut tidak ditemukan pada {df_name}: {missing_cols}"
        )


def get_final_k() -> int:
    """
    Mengambil nilai K final dari config.

    Prioritas:
    1. FINAL_K_OVERRIDE
    2. K_OVERRIDE

    Untuk penelitian berbasis 4 pola permintaan ADI-CV2,
    nilai K final harus 4.
    """
    k_final = getattr(cfg, "FINAL_K_OVERRIDE", None)

    if k_final is None:
        k_final = getattr(cfg, "K_OVERRIDE", None)

    if k_final is None:
        raise ValueError(
            "Nilai K final belum ditemukan. "
            "Tambahkan FINAL_K_OVERRIDE = 4 atau K_OVERRIDE = 4 di config.py."
        )

    k_final = int(k_final)

    if k_final != 4:
        raise ValueError(
            f"Pelabelan 4 kuadran membutuhkan K=4, tetapi K saat ini = {k_final}"
        )

    return k_final


def load_final_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Membaca data transformed dan scaled dari file DEMAND_FEATURES_XLSX.

    transformed:
        Berisi fitur asli seperti ADI dan CV2.

    scaled:
        Berisi fitur hasil scaling seperti scaled_ADI dan scaled_CV2.
    """
    if not cfg.DEMAND_FEATURES_XLSX.exists():
        raise FileNotFoundError(
            f"File Demand Features belum ditemukan: {cfg.DEMAND_FEATURES_XLSX}"
        )

    transformed = pd.read_excel(
        cfg.DEMAND_FEATURES_XLSX,
        sheet_name=f"{cfg.FINAL_SCENARIO}_transformed"
    )

    scaled = pd.read_excel(
        cfg.DEMAND_FEATURES_XLSX,
        sheet_name=f"{cfg.FINAL_SCENARIO}_scaled"
    )

    require_columns(transformed, [SKU_COL] + FEATURE_COLS, "data transformed")
    require_columns(scaled, [SKU_COL] + FEATURE_COLS_SCALED, "data scaled")

    return transformed, scaled


def validate_input_data(transformed: pd.DataFrame, scaled: pd.DataFrame) -> None:
    """
    Validasi keamanan data sebelum clustering.
    """
    if transformed[SKU_COL].duplicated().any():
        duplicated_sku = transformed.loc[
            transformed[SKU_COL].duplicated(), SKU_COL
        ].head(10).tolist()

        raise ValueError(
            "Terdapat SKU duplikat di data transformed. "
            f"Contoh SKU duplikat: {duplicated_sku}"
        )

    if scaled[SKU_COL].duplicated().any():
        duplicated_sku = scaled.loc[
            scaled[SKU_COL].duplicated(), SKU_COL
        ].head(10).tolist()

        raise ValueError(
            "Terdapat SKU duplikat di data scaled. "
            f"Contoh SKU duplikat: {duplicated_sku}"
        )

    if scaled[FEATURE_COLS_SCALED].isna().any().any():
        raise ValueError(
            "Masih terdapat nilai kosong pada scaled_ADI atau scaled_CV2."
        )

    if transformed[FEATURE_COLS].isna().any().any():
        raise ValueError(
            "Masih terdapat nilai kosong pada ADI atau CV2 di data transformed."
        )


def build_cluster_profile(clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Membentuk profil cluster berdasarkan nilai asli ADI dan CV2.

    Profil ini digunakan untuk interpretasi label.
    """
    agg_spec = {
        "n_sku": (SKU_COL, "count"),

        "ADI_mean": ("ADI", "mean"),
        "ADI_median": ("ADI", "median"),
        "ADI_min": ("ADI", "min"),
        "ADI_max": ("ADI", "max"),

        "CV2_mean": ("CV2", "mean"),
        "CV2_median": ("CV2", "median"),
        "CV2_min": ("CV2", "min"),
        "CV2_max": ("CV2", "max"),
    }

    if "zero_month_ratio" in clustered.columns:
        agg_spec["ZMR_mean"] = ("zero_month_ratio", "mean")

    # Revenue tidak digunakan untuk clustering.
    if "Total_Revenue" in clustered.columns:
        agg_spec["Total_Revenue_mean"] = ("Total_Revenue", "mean")
        agg_spec["Total_Revenue_sum"] = ("Total_Revenue", "sum")

    profile = (
        clustered.groupby("cluster")
        .agg(**agg_spec)
        .reset_index()
    )
    
    if "Total_Revenue_sum" in profile.columns:
        total_all_revenue = profile["Total_Revenue_sum"].sum()
        # Membuat kolom persentase (dalam skala 0-100%)
        profile["Revenue_Percentage"] = (profile["Total_Revenue_sum"] / total_all_revenue) * 100

    return profile


def assign_managerial_labels(profile: pd.DataFrame) -> pd.DataFrame:
    """
    Memberikan label Smooth, Erratic, Intermittent, dan Lumpy
    berdasarkan posisi relatif cluster pada ADI_mean dan CV2_mean.

    Logika:
    - Dua cluster dengan ADI_mean terendah dianggap relatif lebih rutin.
    - Dua cluster dengan ADI_mean tertinggi dianggap relatif lebih jarang.

    Pada masing-masing kelompok:
    - CV2_mean rendah  -> lebih stabil
    - CV2_mean tinggi  -> lebih berfluktuasi

    Maka:
    - ADI rendah + CV2 rendah  -> Smooth
    - ADI rendah + CV2 tinggi  -> Erratic
    - ADI tinggi + CV2 rendah  -> Intermittent
    - ADI tinggi + CV2 tinggi  -> Lumpy
    """
    p = profile.copy()

    if len(p) != 4:
        raise ValueError(
            f"Pelabelan Smooth, Erratic, Intermittent, dan Lumpy "
            f"membutuhkan tepat 4 cluster. Jumlah cluster saat ini = {len(p)}"
        )

    p["segment_label"] = None
    p["interpretation"] = None
    p["label_basis"] = "relative_centroid_ranking_ADI_CV2"

    # Urutkan berdasarkan ADI.
    # ADI kecil = permintaan relatif lebih rutin.
    # ADI besar = permintaan relatif lebih jarang.
    p_sorted = p.sort_values(["ADI_mean", "CV2_mean", "cluster"]).reset_index(drop=True)

    low_adi_group = p_sorted.iloc[:2].copy()
    high_adi_group = p_sorted.iloc[2:].copy()

    # Pada kelompok ADI rendah:
    # CV2 rendah = Smooth, CV2 tinggi = Erratic.
    low_adi_by_cv2 = low_adi_group.sort_values(
        ["CV2_mean", "cluster"]
    ).reset_index(drop=True)

    smooth_cluster = int(low_adi_by_cv2.iloc[0]["cluster"])
    erratic_cluster = int(low_adi_by_cv2.iloc[1]["cluster"])

    # Pada kelompok ADI tinggi:
    # CV2 rendah = Intermittent, CV2 tinggi = Lumpy.
    high_adi_by_cv2 = high_adi_group.sort_values(
        ["CV2_mean", "cluster"]
    ).reset_index(drop=True)

    intermittent_cluster = int(high_adi_by_cv2.iloc[0]["cluster"])
    lumpy_cluster = int(high_adi_by_cv2.iloc[1]["cluster"])

    label_map = {
        smooth_cluster: "Smooth",
        erratic_cluster: "Erratic",
        intermittent_cluster: "Intermittent",
        lumpy_cluster: "Lumpy",
    }

    interpretation_map = {
        smooth_cluster: (
            "Permintaan relatif lebih rutin dan ukuran permintaan relatif stabil "
            "dibanding cluster lain."
        ),
        erratic_cluster: (
            "Permintaan relatif lebih rutin, tetapi ukuran permintaan lebih berfluktuasi "
            "dibanding cluster smooth."
        ),
        intermittent_cluster: (
            "Permintaan relatif lebih jarang, tetapi ukuran permintaan relatif stabil "
            "dibanding cluster lumpy."
        ),
        lumpy_cluster: (
            "Permintaan relatif lebih jarang dan ukuran permintaan lebih berfluktuasi "
            "dibanding cluster intermittent."
        ),
    }

    p["segment_label"] = p["cluster"].map(label_map)
    p["interpretation"] = p["cluster"].map(interpretation_map)

    return p


def compute_final_cluster_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    k_final: int
) -> pd.DataFrame:
    """
    Menghitung metrik evaluasi untuk hasil clustering final.
    """
    metrics = {
        "scenario": cfg.FINAL_SCENARIO,
        "k": k_final,
        "n_sku": int(len(X)),
        "Davies_Bouldin": float(davies_bouldin_score(X, labels)),
        "Silhouette": float(silhouette_score(X, labels)),
        "Calinski_Harabasz": float(calinski_harabasz_score(X, labels)),
    }

    return pd.DataFrame([metrics])


def main():
    print("[04] File script aktif:", __file__)
    print("[04] Mulai clustering K-Means berbasis ADI & CV2...")

    transformed, scaled = load_final_data()
    validate_input_data(transformed, scaled)

    k_final = get_final_k()

    print(f"[04] Final scenario: {cfg.FINAL_SCENARIO}")
    print(f"[04] K final: {k_final}")
    print(f"[04] Jumlah data transformed: {len(transformed)}")
    print(f"[04] Jumlah data scaled: {len(scaled)}")

    # ======================================================
    # 1. Proses K-Means menggunakan fitur scaled
    # ======================================================
    X = scaled[FEATURE_COLS_SCALED].astype(float).to_numpy()

    if not np.isfinite(X).all():
        raise ValueError(
            "Masih terdapat nilai NaN atau inf pada fitur scaled_ADI atau scaled_CV2."
        )

    km = KMeans(
        n_clusters=k_final,
        random_state=cfg.RANDOM_STATE,
        n_init=cfg.N_INIT
    )

    labels = km.fit_predict(X)

    # ======================================================
    # 2. Gabungkan label cluster berdasarkan SKU
    # ======================================================
    cluster_mapping = scaled[[SKU_COL]].copy()
    cluster_mapping["cluster"] = labels

    clustered = transformed.merge(
        cluster_mapping,
        on=SKU_COL,
        how="inner",
        validate="one_to_one"
    )

    print(f"[04] Jumlah data clustered setelah merge: {len(clustered)}")

    if len(clustered) != len(cluster_mapping):
        raise ValueError(
            f"Jumlah data setelah merge tidak sama. "
            f"cluster_mapping={len(cluster_mapping)}, clustered={len(clustered)}"
        )

    # ======================================================
    # 3. Buat profil cluster dan beri label demand pattern
    # ======================================================
    profile = build_cluster_profile(clustered)
    profile = assign_managerial_labels(profile)

    label_map = profile.set_index("cluster")["segment_label"].to_dict()
    interpretation_map = profile.set_index("cluster")["interpretation"].to_dict()

    clustered["segment_label"] = clustered["cluster"].map(label_map)
    clustered["interpretation"] = clustered["cluster"].map(interpretation_map)

    # ======================================================
    # 4. Hitung metrik evaluasi clustering final
    # ======================================================
    final_metrics = compute_final_cluster_metrics(X, labels, k_final)

    # ======================================================
    # 5. Simpan hasil ke Excel
    # ======================================================
    cfg.CLUSTER_XLSX.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(cfg.CLUSTER_XLSX, engine="openpyxl") as writer:
        clustered.to_excel(
            writer,
            index=False,
            sheet_name="clustered_labeled_SKU"
        )

        profile.to_excel(
            writer,
            index=False,
            sheet_name="cluster_profile_labeled"
        )

        final_metrics.to_excel(
            writer,
            index=False,
            sheet_name="final_cluster_metrics"
        )

    # ======================================================
    # 6. Print ringkasan hasil
    # ======================================================
    print("[04] Profil cluster berlabel:")
    print(
        profile[
            [
                "cluster",
                "n_sku",
                "ADI_mean",
                "ADI_median",
                "CV2_mean",
                "CV2_median",
                "segment_label",
            ]
        ]
    )

    print("[04] Evaluasi cluster final:")
    print(final_metrics)

    print(f"[04] Saved: {cfg.CLUSTER_XLSX}")


if __name__ == "__main__":
    main()  