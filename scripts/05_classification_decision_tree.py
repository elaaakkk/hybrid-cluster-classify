"""
05_classification_decision_tree.py
====================================================
TAHAP 5 - Klasifikasi label segmen menggunakan Decision Tree berbasis entropy.

Input:
- data_output/04_CLUSTERED_LABELED_SKU.xlsx

Proses:
1. Membaca data SKU yang sudah memiliki cluster dan segment_label.
2. Menggunakan fitur RFM sebagai input klasifikasi.
3. Menggunakan segment_label sebagai target.
4. Membagi data menjadi train dan test.
5. Melatih DecisionTreeClassifier dengan criterion='entropy'.
6. Mengevaluasi hasil dengan confusion matrix, classification report, dan accuracy.
7. Mengekspor aturan pohon keputusan dalam bentuk teks.

Catatan metodologis penting:
- Scikit-learn tidak menyediakan implementasi C4.5 murni.
- DecisionTreeClassifier(criterion='entropy') adalah pendekatan decision tree berbasis information gain/entropy
  yang sering dipakai sebagai implementasi praktis C4.5-like untuk penelitian Python.
- Jika dosen meminta C4.5 murni, bisa dipertimbangkan library tambahan seperti chefboost atau implementasi J48/Weka.

Output:
- data_output/05_CLASSIFICATION_DECISION_TREE.xlsx
- data_output/05_RULES_DECISION_TREE.txt
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts import config as cfg


# Gunakan raw RFM agar aturan lebih mudah dibaca secara bisnis.
# Kalau ingin memakai transformed RFM, ganti menjadi ["Recency", "Frequency", "Monetary"].
# Masukkan ADI dan CV2 ke dalam pohon klasifikasi
FEATURE_COLS = ["Recency_raw", "Frequency_raw", "Monetary_raw", "ADI", "CV2"]
TARGET_COL = "segment_label"


def load_clustered() -> pd.DataFrame:
    if not cfg.CLUSTER_XLSX.exists():
        raise FileNotFoundError(f"File clustered belum ada: {cfg.CLUSTER_XLSX}. Jalankan 04_cluster_label_interpret.py dulu.")
    return pd.read_excel(cfg.CLUSTER_XLSX, sheet_name="clustered_labeled_SKU")


def main():
    print("[05] Mulai klasifikasi label segmen dengan Decision Tree entropy...")
    df = load_clustered()

    # Ambil hanya baris valid.
    data = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

    # Jika cuma ada 1 label, model klasifikasi tidak bermakna.
    n_classes = data[TARGET_COL].nunique()
    if n_classes < 2:
        raise ValueError(f"Target hanya memiliki {n_classes} kelas. Klasifikasi butuh minimal 2 kelas.")

    X = data[FEATURE_COLS].astype(float)
    y = data[TARGET_COL].astype(str)

    # Stratify dipakai agar proporsi label train-test tetap mirip.
    # Jika ada kelas dengan anggota terlalu sedikit, stratify bisa gagal; fallback tanpa stratify.
    try:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, data.index,
            test_size=0.20,
            random_state=cfg.RANDOM_STATE,
            stratify=y
        )
    except ValueError:
        print("[05] Stratify gagal karena ada kelas terlalu kecil. Split dilakukan tanpa stratify.")
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, data.index,
            test_size=0.20,
            random_state=cfg.RANDOM_STATE,
            stratify=None
        )

    model = DecisionTreeClassifier(
        criterion="entropy",
        random_state=cfg.RANDOM_STATE,
        max_depth=None,
        min_samples_leaf=3
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    labels_order = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=[f"actual_{c}" for c in labels_order], columns=[f"pred_{c}" for c in labels_order])

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "class"})

    summary = pd.DataFrame([{
        "model": "DecisionTreeClassifier_entropy_C45_like",
        "features": ", ".join(FEATURE_COLS),
        "target": TARGET_COL,
        "n_data": int(len(data)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_classes": int(n_classes),
        "accuracy": float(acc),
    }])

    test_result = data.loc[idx_test, ["SKU", TARGET_COL, "cluster"] + FEATURE_COLS].copy()
    test_result["predicted_label"] = y_pred
    test_result["is_correct"] = test_result[TARGET_COL].astype(str).values == test_result["predicted_label"].astype(str).values

    feature_importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    rules = export_text(model, feature_names=FEATURE_COLS)
    cfg.RULES_TXT.write_text(rules, encoding="utf-8")

    with pd.ExcelWriter(cfg.CLASSIFICATION_XLSX, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="summary")
        cm_df.to_excel(writer, sheet_name="confusion_matrix")
        report_df.to_excel(writer, index=False, sheet_name="classification_report")
        test_result.to_excel(writer, index=False, sheet_name="test_predictions")
        feature_importance.to_excel(writer, index=False, sheet_name="feature_importance")

    print("[05] Summary:")
    print(summary)
    print("[05] Confusion matrix:")
    print(cm_df)
    print(f"[05] Saved: {cfg.CLASSIFICATION_XLSX}")
    print(f"[05] Rules saved: {cfg.RULES_TXT}")


if __name__ == "__main__":
    main()
