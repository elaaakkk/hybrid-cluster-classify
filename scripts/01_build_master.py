"""
01_build_master.py
====================================================
TAHAP 1 - Gabung sheet transaksi menjadi master data.

Input:
- File Excel mentah pada folder data_raw/.

Proses:
1. Membaca semua sheet dari setiap file Excel.
2. Mendeteksi baris header yang benar.
3. Menyamakan nama kolom menjadi uppercase dan rapi.
4. Menyimpan hanya kolom inti yang diperlukan.
5. Menggabungkan seluruh sheet menjadi satu master transaksi.
6. Membersihkan tanggal dan harga jual awal.
7. Menghapus duplikasi awal.

Output:
- data_output/01_MASTER_REKAP_PENJUALAN_2023_2025.xlsx
- data_output/01_MASTER_REKAP_PENJUALAN_2023_2025.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts import config as cfg


def norm_text(value: object) -> str:
    """Normalisasi nama kolom: uppercase, strip, dan rapikan spasi dobel."""
    return " ".join(("" if value is None else str(value)).strip().upper().split())


KEEP_COLS = [norm_text(c) for c in cfg.KEEP_COLS_RAW]
EXPECTED_HEADERS = set(KEEP_COLS)


def clean_columns(columns) -> list[str]:
    return [norm_text(c) for c in columns]


def detect_header_row(excel_path: Path, sheet_name: str, max_scan_rows: int = 8) -> int:
    """Mencari baris header terbaik dari beberapa baris awal sheet."""
    preview = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, nrows=max_scan_rows)
    best_i, best_score = 0, -1

    for i in range(min(max_scan_rows, len(preview))):
        row = preview.iloc[i].map(norm_text)
        row_set = {x for x in row.tolist() if x not in {"", "NAN", "NONE"}}
        score = len(row_set.intersection(EXPECTED_HEADERS))
        if score > best_score:
            best_i, best_score = i, score

    return best_i if best_score >= 2 else 0


def read_all_sheets(path: Path, source_label: str) -> pd.DataFrame:
    """Membaca semua sheet dari satu workbook Excel lalu menyatukannya."""
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    frames = []
    xls = pd.ExcelFile(path)

    for sheet in xls.sheet_names:
        header_row = detect_header_row(path, sheet)
        df = pd.read_excel(path, sheet_name=sheet, header=header_row)
        if df.empty:
            continue

        df = df.copy()
        df.columns = clean_columns(df.columns)

        # Hapus kolom kosong/unnamed/simbol tidak perlu.
        drop_cols = [c for c in df.columns if c.startswith("UNNAMED") or c in {"", "*"}]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Pertahankan hanya kolom yang dibutuhkan penelitian.
        existing_keep = [c for c in KEEP_COLS if c in df.columns]
        df = df[existing_keep].copy()
        df = df.dropna(how="all")

        # Metadata untuk tracking asal baris transaksi.
        df["__SOURCE_FILE"] = source_label
        df["__SOURCE_SHEET"] = sheet
        frames.append(df)

        print(f"[01] Baca {source_label} | sheet={sheet} | rows={len(df)}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def clean_master(master: pd.DataFrame) -> pd.DataFrame:
    """Pembersihan ringan pada master transaksi hasil gabungan."""
    m = master.copy()

    # Rapikan kolom teks.
    for col in [cfg.COL_MEREK, cfg.COL_NAMA, cfg.COL_TIPE, "NAMA PEMBELI", "BELI DARI"]:
        if col in m.columns:
            m[col] = (
                m[col].astype(str)
                      .str.strip()
                      .str.upper()
                      .str.replace(r"\s+", " ", regex=True)
                      .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
            )

    # Harga jual -> numerik. Nilai yang tidak valid dibiarkan NaN, bukan diisi 0.
    if cfg.COL_HARGA in m.columns:
        m[cfg.COL_HARGA] = (
            m[cfg.COL_HARGA].astype(str)
                             .str.replace(r"[^\d]", "", regex=True)
                             .replace("", np.nan)
        )
        m[cfg.COL_HARGA] = pd.to_numeric(m[cfg.COL_HARGA], errors="coerce")
        m.loc[m[cfg.COL_HARGA] <= 0, cfg.COL_HARGA] = np.nan

    # Tanggal terjual -> datetime.
    if cfg.COL_TGL in m.columns:
        m[cfg.COL_TGL] = pd.to_datetime(m[cfg.COL_TGL], errors="coerce", dayfirst=True).dt.normalize()

    # Baris wajib minimal punya tanggal, merek, dan nama barang.
    required = [c for c in [cfg.COL_TGL, cfg.COL_MEREK, cfg.COL_NAMA] if c in m.columns]
    before = len(m)
    m = m.dropna(subset=required)
    print(f"[01] Drop baris wajib kosong: {before - len(m)}")

    # Dedup awal: masih akan ada dedup lanjutan di tahap preprocessing.
    subset = [c for c in [cfg.COL_IMEI, cfg.COL_TGL, cfg.COL_HARGA, cfg.COL_MEREK, cfg.COL_NAMA, cfg.COL_TIPE] if c in m.columns]
    before = len(m)
    m = m.drop_duplicates(subset=subset) if subset else m.drop_duplicates()
    print(f"[01] Drop duplikasi awal: {before - len(m)}")

    return m.reset_index(drop=True)


def main():
    print("[01] Mulai gabung sheet transaksi...")
    frames = []
    for path, label in cfg.RAW_FILES:
        print(f"[01] Cek file: {path} | exists={path.exists()}")
        frames.append(read_all_sheets(path, label))

    master = pd.concat(frames, ignore_index=True)
    master = clean_master(master)

    with pd.ExcelWriter(cfg.MASTER_XLSX, engine="openpyxl") as writer:
        master.to_excel(writer, index=False, sheet_name="MASTER")
    master.to_csv(cfg.MASTER_CSV, index=False, encoding="utf-8-sig")

    print(f"[01] Final master shape: {master.shape}")
    print(f"[01] Saved: {cfg.MASTER_XLSX}")
    print(f"[01] Saved: {cfg.MASTER_CSV}")


if __name__ == "__main__":
    main()
