"""
05_classification_decision_tree.py
====================================================
TAHAP 5 - Klasifikasi label segmen menggunakan Decision Tree dengan fitur ADI & CV2.

Label yang diprediksi adalah pseudo-label dari K-Means tahap 04, bukan ground truth manual.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts import config as cfg


FEATURE_COLS = ["ADI", "CV2"]
TARGET_COL = "segment_label"


def main() -> None:
    print("[05] Mulai klasifikasi Decision Tree untuk ADI & CV2...")

    if not cfg.CLUSTER_XLSX.exists():
        raise FileNotFoundError(f"File cluster belum ada: {cfg.CLUSTER_XLSX}")

    df = pd.read_excel(cfg.CLUSTER_XLSX, sheet_name="clustered_labeled_SKU")
    missing = [col for col in FEATURE_COLS + [TARGET_COL] if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan pada clustered_labeled_SKU: {missing}")

    data = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    if data[TARGET_COL].nunique() < 2:
        raise ValueError("Target label kurang dari 2 kelas. Decision Tree tidak bisa dievaluasi.")

    class_counts = data[TARGET_COL].value_counts()
    if (class_counts < 2).any():
        raise ValueError(
            "Ada kelas dengan jumlah data < 2, train_test_split stratify tidak valid. "
            f"Distribusi kelas: {class_counts.to_dict()}"
        )

    X = data[FEATURE_COLS].astype(float)
    y = data[TARGET_COL].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=cfg.RANDOM_STATE,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        criterion="entropy",
        random_state=cfg.RANDOM_STATE,
        max_depth=None,
        min_samples_leaf=3,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    labels_order = sorted(y.unique())

    cm_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels_order),
        index=[f"actual_{c}" for c in labels_order],
        columns=[f"pred_{c}" for c in labels_order],
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels_order,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})

    predictions = X_test.copy()
    predictions["actual_label"] = y_test.values
    predictions["predicted_label"] = y_pred
    predictions["is_correct"] = predictions["actual_label"].eq(predictions["predicted_label"])

    summary_df = pd.DataFrame([
        {
            "accuracy": float(acc),
            "n_total": int(len(data)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features": ", ".join(FEATURE_COLS),
            "target": TARGET_COL,
            "note": "Accuracy measures replication of K-Means pseudo-labels, not external ground truth.",
        }
    ])

    feature_importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    rules = export_text(model, feature_names=FEATURE_COLS)
    cfg.RULES_TXT.write_text(rules, encoding="utf-8")

    cfg.CLASSIFICATION_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(cfg.CLASSIFICATION_XLSX, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        class_counts.rename_axis("segment_label").reset_index(name="n_sku").to_excel(
            writer,
            index=False,
            sheet_name="class_distribution",
        )
        cm_df.to_excel(writer, sheet_name="confusion_matrix")
        report_df.to_excel(writer, index=False, sheet_name="classification_report")
        feature_importance_df.to_excel(writer, index=False, sheet_name="feature_importance")
        predictions.to_excel(writer, index=True, sheet_name="test_predictions")

    print(f"[05] Accuracy: {acc:.4f}")
    print("[05] Confusion matrix:")
    print(cm_df)
    print(f"[05] Saved: {cfg.CLASSIFICATION_XLSX}")
    print(f"[05] Rules saved: {cfg.RULES_TXT}")


if __name__ == "__main__":
    main()
