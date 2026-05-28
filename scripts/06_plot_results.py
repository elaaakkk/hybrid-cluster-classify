"""
06_plot_results.py
====================================================
TAHAP 6 - Visualisasi hasil evaluasi K dan scatter plot ADI-CV2.

Input:
- data_output/03_K_EVALUATION_DBI.xlsx
- data_output/04_CLUSTERED_LABELED_SKU.xlsx

Output:
- data_output/plots/k_validity_FINAL_SCENARIO.png
- data_output/plots/demand_2d_labeled_FINAL_SCENARIO.png
- data_output/plots/demand_2d_by_cluster_FINAL_SCENARIO.png
- data_output/plots/demand_2d_labeled_unscaled.png  <-- (BARU)
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
    axes[0].axvline(best_dbi_k, linestyle="--", alpha=0.7, color='red')
    axes[0].set_ylabel("DBI\n(kecil=baik)")
    axes[0].set_title(f"K Validity - {cfg.FINAL_SCENARIO}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(scores["k"], scores["Silhouette"], marker="o")
    axes[1].axvline(best_sil_k, linestyle="--", alpha=0.7, color='red')
    axes[1].set_ylabel("Silhouette\n(besar=baik)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(scores["k"], scores["Calinski_Harabasz"], marker="o")
    axes[2].axvline(best_ch_k, linestyle="--", alpha=0.7, color='red')
    axes[2].set_ylabel("CHI\n(besar=baik)")
    axes[2].set_xlabel("Jumlah Cluster (K)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out = cfg.PLOTS_DIR / f"k_validity_{cfg.FINAL_SCENARIO}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[06] Saved: {out}")


def plot_demand_2d_unscaled():
    """Plot 2D scatter plot ADI dan CV2 (RAW / UNSCALED) untuk membuktikan skewness."""
    if not cfg.CLUSTER_XLSX.exists():
        raise FileNotFoundError(f"File clustered belum ada: {cfg.CLUSTER_XLSX}")

    labeled = pd.read_excel(cfg.CLUSTER_XLSX, sheet_name="clustered_labeled_SKU")
    # Mengambil nilai mentah asli sebelum di-log dan di-scale
    raw_data = pd.read_excel(cfg.DEMAND_FEATURES_XLSX, sheet_name="demand_features_raw")

    df = labeled[["SKU", "segment_label"]].merge(raw_data, on="SKU", how="inner")
    df = df.dropna(subset=["ADI", "CV2", "segment_label"]).copy()

    labels = sorted(df["segment_label"].unique())
    fig, ax = plt.subplots(figsize=(8, 6))

    for label in labels:
        subset = df[df["segment_label"] == label]
        ax.scatter(subset["ADI"], subset["CV2"], s=25, alpha=0.8, label=label)

    ax.set_title("Kuadran Pola Permintaan (Data Mentah / Unscaled)")
    ax.set_xlabel("Raw ADI (Jarak antar transaksi dalam Hari)")
    ax.set_ylabel("Raw CV² (Fluktuasi Kuantitas)")
    ax.legend(title="Segment Label")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = cfg.PLOTS_DIR / "demand_2d_labeled_unscaled.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[06] Saved: {out}")


def plot_demand_2d():
    """Plot 2D scatter plot ADI dan CV2 berdasarkan 4 Kuadran Manajerial."""
    if not cfg.CLUSTER_XLSX.exists():
        raise FileNotFoundError(f"File clustered belum ada: {cfg.CLUSTER_XLSX}")

    labeled = pd.read_excel(cfg.CLUSTER_XLSX, sheet_name="clustered_labeled_SKU")
    scaled = pd.read_excel(cfg.DEMAND_FEATURES_XLSX, sheet_name=f"{cfg.FINAL_SCENARIO}_scaled")

    df = labeled[["SKU", "segment_label"]].merge(scaled, on="SKU", how="inner")
    df = df.dropna(subset=["scaled_ADI", "scaled_CV2", "segment_label"]).copy()

    labels = sorted(df["segment_label"].unique())
    fig, ax = plt.subplots(figsize=(8, 6))

    for label in labels:
        subset = df[df["segment_label"] == label]
        ax.scatter(subset["scaled_ADI"], subset["scaled_CV2"], s=25, alpha=0.8, label=label)

    ax.set_title(f"Kuadran Pola Permintaan (Scaled) - {cfg.FINAL_SCENARIO}")
    ax.set_xlabel("Scaled ADI (Jarak antar transaksi)")
    ax.set_ylabel("Scaled CV² (Fluktuasi Kuantitas)")
    ax.legend(title="Segment Label")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = cfg.PLOTS_DIR / f"demand_2d_labeled_{cfg.FINAL_SCENARIO}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[06] Saved: {out}")


def plot_demand_2d_by_cluster():
    """Plot 2D scatter plot ADI dan CV2 berdasarkan Cluster Teknis."""
    if not cfg.CLUSTER_XLSX.exists():
        raise FileNotFoundError(f"File clustered belum ada: {cfg.CLUSTER_XLSX}")

    labeled = pd.read_excel(cfg.CLUSTER_XLSX, sheet_name="clustered_labeled_SKU")
    scaled = pd.read_excel(cfg.DEMAND_FEATURES_XLSX, sheet_name=f"{cfg.FINAL_SCENARIO}_scaled")

    df = labeled[["SKU", "cluster"]].merge(scaled, on="SKU", how="inner")
    df = df.dropna(subset=["scaled_ADI", "scaled_CV2", "cluster"]).copy()

    clusters = sorted(df["cluster"].unique())
    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster in clusters:
        subset = df[df["cluster"] == cluster]
        ax.scatter(subset["scaled_ADI"], subset["scaled_CV2"], s=25, alpha=0.8, label=f"Cluster {cluster}")

    ax.set_title(f"Cluster Teknis K-Means (Scaled ADI & CV2) - {cfg.FINAL_SCENARIO}")
    ax.set_xlabel("Scaled ADI")
    ax.set_ylabel("Scaled CV²")
    ax.legend(title="Cluster ID")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = cfg.PLOTS_DIR / f"demand_2d_by_cluster_{cfg.FINAL_SCENARIO}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[06] Saved: {out}")


def plot_count_by_label():
    df = pd.read_excel(cfg.CLUSTER_XLSX, sheet_name="clustered_labeled_SKU")
    count = df["segment_label"].value_counts().reset_index()
    count.columns = ["segment_label", "n_sku"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(count["segment_label"], count["n_sku"], color=['skyblue', 'lightgreen', 'salmon', 'orange'])
    ax.set_xlabel("Label Segmen")
    ax.set_ylabel("Jumlah SKU")
    ax.set_title("Distribusi SKU per Pola Permintaan")
    plt.xticks(rotation=0)
    
    # Tambahkan angka di atas bar
    for i, v in enumerate(count["n_sku"]):
        ax.text(i, v + (v*0.01), str(v), ha='center', va='bottom', fontweight='bold')

    fig.tight_layout()
    out = cfg.PLOTS_DIR / "sku_count_by_label.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[06] Saved: {out}")


def main():
    print("[06] Mulai membuat visualisasi 2D Kuadran Permintaan...")
    plot_k_validity()
    plot_demand_2d_unscaled()  # Panggil grafik unscaled di sini
    plot_demand_2d()
    plot_demand_2d_by_cluster()
    plot_count_by_label()
    print(f"[06] Semua plot tersimpan di: {cfg.PLOTS_DIR}")

if __name__ == "__main__":
    main()
