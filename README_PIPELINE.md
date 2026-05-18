# Pipeline Skripsi RFM + K-Means + Decision Tree

Folder ini berisi kode versi baru yang dipisah per tahap supaya hasil sementara dan hasil final tidak tercampur.

## Urutan Pipeline

### 1. `01_build_master.py` — Gabung Sheet
Menggabungkan seluruh sheet dari file Excel penjualan 2023-2024 dan 2025 menjadi satu master transaksi.

Output:
- `data_output/01_MASTER_REKAP_PENJUALAN_2023_2025.xlsx`
- `data_output/01_MASTER_REKAP_PENJUALAN_2023_2025.csv`

### 2. `02_preprocess_transform_rfm.py` — Preprocess + Transformasi RFM
Membersihkan transaksi, filter periode 2023-2025, bentuk SKU, dedup, dan bentuk fitur RFM.

Fitur:
- Recency = jarak hari sejak SKU terakhir terjual
- Frequency = jumlah transaksi SKU
- Monetary = total nilai penjualan SKU

Script ini juga membuat 5 skenario transformasi:
- S1 = log1p Frequency & Monetary + Z-score
- S2 = log1p Frequency & Monetary + Min-Max
- S3 = raw Frequency & Monetary + Z-score
- S4 = raw Frequency & Monetary + Min-Max
- S5 = raw Frequency & Monetary + tanpa scaling

Output:
- `data_output/02_RFM_FEATURES.xlsx`

### 3. `03_determine_k_dbi.py` — Penentuan Jumlah K dengan DBI
Menjalankan K-Means untuk rentang K=2 sampai K=8 pada tiap skenario.

Metrik:
- Davies-Bouldin Index / DBI: semakin kecil semakin baik. Ini kriteria utama.
- Silhouette: semakin besar semakin baik.
- Calinski-Harabasz: semakin besar semakin baik.
- WSS/Inertia: pendukung elbow.

Output:
- `data_output/03_K_EVALUATION_DBI.xlsx`

Sheet penting:
- `k_scores_all`
- `best_k_by_scenario`
- `best_overall`

### 4. `04_cluster_label_interpret.py` — Clustering Final + Label Interpretasi
Mengambil skenario final dari `config.py`, lalu memakai K terbaik berdasarkan DBI, kecuali kamu isi `FINAL_K_OVERRIDE`.

Label manajerial:
- fast-moving = cluster dengan aktivitas demand tertinggi
- medium-moving = cluster tengah
- slow-moving = cluster dengan aktivitas demand terendah

Kalau K teknis = 4, script tetap bisa memetakan hasilnya ke 3 label:
- cluster paling aktif -> fast-moving
- dua cluster tengah -> medium-moving
- cluster paling rendah -> slow-moving

Output:
- `data_output/04_CLUSTERED_LABELED_SKU.xlsx`

### 5. `05_classification_decision_tree.py` — Klasifikasi Label
Melatih Decision Tree berbasis entropy untuk memprediksi `segment_label` dari fitur RFM.

Catatan:
Scikit-learn tidak menyediakan C4.5 murni. Script ini memakai `DecisionTreeClassifier(criterion="entropy")` sebagai pendekatan C4.5-like. Kalau dosen meminta C4.5 murni, bisa memakai Weka J48 atau library tambahan.

Output:
- `data_output/05_CLASSIFICATION_DECISION_TREE.xlsx`
- `data_output/05_RULES_DECISION_TREE.txt`

### 6. `06_plot_results.py` — Visualisasi
Membuat plot evaluasi K, scatter RFM, dan jumlah SKU per label.

Output:
- `data_output/plots/k_validity_S1.png`
- `data_output/plots/rfm_3d_labeled_S1.png`
- `data_output/plots/sku_count_by_label.png`

## Cara Menjalankan

Dari folder root project:

```bash
python scripts/01_build_master.py
python scripts/02_preprocess_transform_rfm.py
python scripts/03_determine_k_dbi.py
python scripts/04_cluster_label_interpret.py
python scripts/05_classification_decision_tree.py
python scripts/06_plot_results.py
```

Atau langsung semua:

```bash
python run_all.py
```

## Mengulang Eksperimen K = 3 atau K = 4

Buka `scripts/config.py`, lalu ubah:

```python
FINAL_K_OVERRIDE = 3
```

atau:

```python
FINAL_K_OVERRIDE = 4
```

Kalau ingin otomatis kembali berdasarkan DBI:

```python
FINAL_K_OVERRIDE = None
```

## Mengganti Skenario Final

Buka `scripts/config.py`, lalu ubah:

```python
FINAL_SCENARIO = "S1"
```

Pilihan: `S1`, `S2`, `S3`, `S4`, `S5`.
