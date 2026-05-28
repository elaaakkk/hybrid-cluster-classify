"""
05_classification_decision_tree.py
====================================================
TAHAP 5 - Klasifikasi label segmen menggunakan Decision Tree murni dengan ADI & CV2.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts import config as cfg

# [BERUBAH] Fitur Decision Tree hanya 2 ini saja
FEATURE_COLS = ["ADI", "CV2"]
TARGET_COL = "segment_label"

def main():
    print("[05] Mulai klasifikasi Decision Tree untuk ADI & CV2...")
    df = pd.read_excel(cfg.CLUSTER_XLSX, sheet_name="clustered_labeled_SKU")
    data = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

    X = data[FEATURE_COLS].astype(float)
    y = data[TARGET_COL].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20,
        random_state=cfg.RANDOM_STATE,
        stratify=y
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
    cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels_order), index=[f"actual_{c}" for c in labels_order], columns=[f"pred_{c}" for c in labels_order])

    rules = export_text(model, feature_names=FEATURE_COLS)
    cfg.RULES_TXT.write_text(rules, encoding="utf-8")

    print(f"[05] Accuracy: {acc:.4f}")
    print("[05] Rules berhasil diekstrak! Cek folder data_output.")

if __name__ == "__main__":
    main()