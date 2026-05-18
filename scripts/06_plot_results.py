"""
06_plot_results.py
====================================================
TAHAP 6 - Visualisasi hasil K dan clustering.

Input:
- data_output/03_K_EVALUATION_DBI.xlsx
- data_output/04_CLUSTERED_LABELED_SKU.xlsx

Proses:
1. Membuat plot evaluasi K per skenario final:
   - DBI, Silhouette, CHI dalam panel terpisah.
2. Membuat plot scatter RFM 3D berdasarkan label segmen.
3. Membuat bar chart jumlah SKU per label.

Output:
- data_output/plots/k_validity_FINAL_SCENARIO.png
- data_output/plots/rfm_3d_labeled_FINAL_SCENARIO.png
- data_output/plots/sku_count_by_label.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts import config as cfg


def plot_k_validity():
    if not cfg.K_EVAL_XLSX.exists():
        raise FileNotFoundError(f"File evaluasi K belum ada: {cfg.K_EVAL_XLSX}")

    scores = pd.read_excel(cfg.K_EVAL_XLSX, sheet_name="k_scores_all")
    scores = scores[scores["scenario"] == cfg.FINAL_SCENARIO].copy()

    best_dbi_k = int(scores.loc[scores["Davies_Bouldin"].idxmin(), "k"])
    best_sil_k = int(scores.loc[scores["Silhouette"].idxmax(), "k"])
    best_ch_k = int(scores.loc[scores["Calinski_Harabasz"].idxmax(), "k"])

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(scores["k"], scores["Davies_Bouldin"], marker="o")
    axes[0].axvline(best_dbi_k, linestyle="--", alpha=0.7)
    axes[0].set_ylabel("DBI\n(kecil=baik)")
    axes[0].set_title(f"K Validity - {cfg.FINAL_SCENARIO}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(scores["k"], scores["Silhouette"], marker="o")
    axes[1].axvline(best_sil_k, linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Silhouette\n(besar=baik)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(scores["k"], scores["Calinski_Harabasz"], marker="o")
    axes[2].axvline(best_ch_k, linestyle="--", alpha=0.7)
    axes[2].set_ylabel("CHI\n(besar=baik)")
    axes[2].set_xlabel("Jumlah Cluster (K)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out = cfg.PLOTS_DIR / f"k_validity_{cfg.FINAL_SCENARIO}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[06] Saved: {out}")


def plot_rfm_3d():
    if not cfg.CLUSTER_XLSX.exists():
        raise FileNotFoundError(f"File clustered belum ada: {cfg.CLUSTER_XLSX}")

    # Label hasil clustering final
    labeled = pd.read_excel(cfg.CLUSTER_XLSX, sheet_name="clustered_labeled_SKU")

    # Data scaled S1 yang benar-benar dipakai K-Means
    scaled = pd.read_excel(cfg.RFM_XLSX, sheet_name=f"{cfg.FINAL_SCENARIO}_scaled")

    df = labeled[["SKU", "segment_label"]].merge(scaled, on="SKU", how="inner")
    df = df.dropna(
        subset=["scaled_Recency", "scaled_Frequency", "scaled_Monetary", "segment_label"]
    ).copy()

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    labels = sorted(df["segment_label"].unique())
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for label in labels:
        subset = df[df["segment_label"] == label]
        ax.scatter(
            subset["scaled_Recency"],
            subset["scaled_Frequency"],
            subset["scaled_Monetary"],
            s=16,
            alpha=0.75,
            label=label
        )

    ax.set_title(f"Sebaran SKU berdasarkan RFM Scaled dan Label Segmen ({cfg.FINAL_SCENARIO})")
    ax.set_xlabel("Scaled Recency")
    ax.set_ylabel("Scaled Frequency")
    ax.set_zlabel("Scaled Monetary")
    ax.legend()
    fig.tight_layout()

    out = cfg.PLOTS_DIR / f"rfm_3d_labeled_scaled_{cfg.FINAL_SCENARIO}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[06] Saved: {out}")

def plot_rfm_3d_by_cluster():
    """
    Plot 3D RFM scaled berdasarkan cluster teknis K-Means.
    Ini dipakai untuk melihat pemisahan hasil cluster asli sebelum dipetakan
    ke label manajerial fast/medium/slow.
    """
    if not cfg.CLUSTER_XLSX.exists():
        raise FileNotFoundError(f"File clustered belum ada: {cfg.CLUSTER_XLSX}")

    # Data hasil clustering final, berisi SKU dan cluster teknis
    labeled = pd.read_excel(cfg.CLUSTER_XLSX, sheet_name="clustered_labeled_SKU")

    # Data scaled sesuai skenario final, misalnya S1_scaled
    scaled = pd.read_excel(cfg.RFM_XLSX, sheet_name=f"{cfg.FINAL_SCENARIO}_scaled")

    # Gabungkan agar setiap SKU punya cluster + fitur scaled
    df = labeled[["SKU", "cluster", "segment_label"]].merge(scaled, on="SKU", how="inner")

    df = df.dropna(
        subset=["scaled_Recency", "scaled_Frequency", "scaled_Monetary", "cluster"]
    ).copy()

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    clusters = sorted(df["cluster"].unique())

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for cluster in clusters:
        subset = df[df["cluster"] == cluster]
        ax.scatter(
            subset["scaled_Recency"],
            subset["scaled_Frequency"],
            subset["scaled_Monetary"],
            s=16,
            alpha=0.75,
            label=f"Cluster {cluster}"
        )

    ax.set_title(f"Sebaran SKU berdasarkan RFM Scaled dan Cluster Teknis ({cfg.FINAL_SCENARIO})")
    ax.set_xlabel("Scaled Recency")
    ax.set_ylabel("Scaled Frequency")
    ax.set_zlabel("Scaled Monetary")
    ax.legend()
    fig.tight_layout()

    out = cfg.PLOTS_DIR / f"rfm_3d_by_cluster_scaled_{cfg.FINAL_SCENARIO}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[06] Saved: {out}")

def plot_count_by_label():
    df = pd.read_excel(cfg.CLUSTER_XLSX, sheet_name="clustered_labeled_SKU")
    count = df["segment_label"].value_counts().reset_index()
    count.columns = ["segment_label", "n_sku"]

    fig = plt.figure(figsize=(7, 4.5))
    plt.bar(count["segment_label"], count["n_sku"])
    plt.xlabel("Label Segmen")
    plt.ylabel("Jumlah SKU")
    plt.title("Jumlah SKU per Label Segmen")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()

    out = cfg.PLOTS_DIR / "sku_count_by_label.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[06] Saved: {out}")


def main():
    print("[06] Mulai membuat visualisasi...")
    plot_k_validity()
    plot_rfm_3d()
    plot_rfm_3d_by_cluster()
    plot_count_by_label()
    print(f"[06] Semua plot tersimpan di: {cfg.PLOTS_DIR}")


if __name__ == "__main__":
    main()
