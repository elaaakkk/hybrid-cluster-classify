"""
03_determine_k_dbi.py
====================================================
TAHAP 3 - Penentuan jumlah cluster K dengan DBI sebagai kriteria utama.

Input:
- data_output/02_RFM_FEATURES.xlsx

Proses:
1. Membaca fitur scaled dari tiap skenario S1-S5.
2. Menjalankan K-Means untuk K = K_MIN sampai K_MAX.
3. Menghitung:
   - Davies-Bouldin Index / DBI: kecil = baik, menjadi kriteria utama.
   - Silhouette Score: besar = baik, metrik pendukung.
   - Calinski-Harabasz Index: besar = baik, metrik pendukung.
   - WSS/Inertia: untuk melihat elbow.
4. Memilih K terbaik per skenario berdasarkan DBI terkecil.
5. Jika DBI seri, tie-break memakai Silhouette terbesar, lalu CHI terbesar.

Output:
- data_output/03_K_EVALUATION_DBI.xlsx
  Sheet:
  - k_scores_all
  - best_k_by_scenario
  - best_overall
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts import config as cfg


FEATURE_COLS_SCALED = ["scaled_Recency", "scaled_Frequency", "scaled_Monetary"]
SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]


def load_scaled_features(scenario: str) -> pd.DataFrame:
    if not cfg.RFM_XLSX.exists():
        raise FileNotFoundError(f"File RFM belum ada: {cfg.RFM_XLSX}. Jalankan 02_preprocess_transform_rfm.py dulu.")
    return pd.read_excel(cfg.RFM_XLSX, sheet_name=f"{scenario}_scaled")


def evaluate_k_for_scenario(scenario: str, df_scaled: pd.DataFrame) -> pd.DataFrame:
    """Menghitung DBI, Silhouette, CHI, dan WSS untuk rentang K tertentu."""
    X = df_scaled[FEATURE_COLS_SCALED].astype(float).values
    rows = []

    for k in range(cfg.K_MIN, cfg.K_MAX + 1):
        print(f"[03] Evaluasi {scenario} | K={k}")
        km = KMeans(n_clusters=k, random_state=cfg.RANDOM_STATE, n_init=cfg.N_INIT)
        labels = km.fit_predict(X)

        rows.append({
            "scenario": scenario,
            "k": k,
            "WSS_inertia": float(km.inertia_),
            "Davies_Bouldin": float(davies_bouldin_score(X, labels)),
            "Silhouette": float(silhouette_score(X, labels)),
            "Calinski_Harabasz": float(calinski_harabasz_score(X, labels)),
            "n_sku_evaluated": int(len(df_scaled)),
        })

    return pd.DataFrame(rows)


def pick_best_k_dbi(df_scores: pd.DataFrame) -> pd.DataFrame:
    """Memilih K terbaik per skenario: DBI minimum, lalu Silhouette maksimum, lalu CHI maksimum."""
    best_rows = []

    for scenario, group in df_scores.groupby("scenario"):
        g = group.copy()
        cand = g[g["Davies_Bouldin"] == g["Davies_Bouldin"].min()].copy()
        if len(cand) > 1:
            cand = cand[cand["Silhouette"] == cand["Silhouette"].max()].copy()
        if len(cand) > 1:
            cand = cand[cand["Calinski_Harabasz"] == cand["Calinski_Harabasz"].max()].copy()
        best_rows.append(cand.sort_values("k").iloc[0])

    return pd.DataFrame(best_rows).reset_index(drop=True)


def pick_best_overall(best_by_scenario: pd.DataFrame, final_scenario: str = "S1") -> pd.DataFrame:
    """
    Memilih hasil final berdasarkan skenario utama penelitian.
    Default: S1 = log1p Frequency & Monetary + Z-score.
    
    Catatan:
    Skenario lain tetap disimpan sebagai pembanding, tetapi tidak dipakai
    sebagai model final karena S5/tanpa scaling bisa bias akibat perbedaan
    skala fitur RFM.
    """
    cand = best_by_scenario[best_by_scenario["scenario"] == final_scenario].copy()

    if cand.empty:
        raise ValueError(f"Skenario final {final_scenario} tidak ditemukan di best_by_scenario.")

    return cand.sort_values(["Davies_Bouldin", "k"], ascending=[True, True]).head(1).reset_index(drop=True)


def main():
    print("[03] Mulai evaluasi K berbasis DBI...")

    all_scores = []
    for scenario in SCENARIOS:
        df_scaled = load_scaled_features(scenario)
        all_scores.append(evaluate_k_for_scenario(scenario, df_scaled))

    k_scores_all = pd.concat(all_scores, ignore_index=True)
    best_k_by_scenario = pick_best_k_dbi(k_scores_all)
    best_overall = pick_best_overall(best_k_by_scenario, final_scenario="S1")

    with pd.ExcelWriter(cfg.K_EVAL_XLSX, engine="openpyxl") as writer:
        k_scores_all.to_excel(writer, index=False, sheet_name="k_scores_all")
        best_k_by_scenario.to_excel(writer, index=False, sheet_name="best_k_by_scenario")
        best_overall.to_excel(writer, index=False, sheet_name="best_overall")

    print("[03] Best K per skenario:")
    print(best_k_by_scenario[["scenario", "k", "Davies_Bouldin", "Silhouette", "Calinski_Harabasz"]])
    print("[03] Best overall:")
    print(best_overall[["scenario", "k", "Davies_Bouldin", "Silhouette", "Calinski_Harabasz"]])
    print(f"[03] Saved: {cfg.K_EVAL_XLSX}")


if __name__ == "__main__":
    main()
