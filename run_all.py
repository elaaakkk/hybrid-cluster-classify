"""
run_all.py
====================================================
Menjalankan seluruh pipeline dari awal sampai akhir.

Urutan:
1. Gabung semua sheet transaksi menjadi master.
2. Preprocess transaksi dan bentuk RFM.
3. Tentukan K dengan DBI.
4. Clustering final + label interpretasi.
5. Klasifikasi label dengan Decision Tree entropy.
6. Visualisasi hasil.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "01_build_master.py",
    "02_preprocess_transform_rfm.py",
    "03_determine_k_dbi.py",
    "04_cluster_label_interpret.py",
    "05_classification_decision_tree.py",
    "06_plot_results.py",
]

for script in SCRIPTS:
    print("=" * 80)
    print(f"RUNNING: scripts/{script}")
    print("=" * 80)
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)

print("Selesai semua ✅")
