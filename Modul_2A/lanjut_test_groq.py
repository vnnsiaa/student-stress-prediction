# ==========================================================
# MODUL 2A - TEST FINAL GROQ DENGAN CHECKPOINT
# STUDENT STRESS PREDICTION
# ==========================================================

import json
import os
import sys
import time
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from classifier import NAMA_FITUR, classify


# ==========================================================
# 1. LOKASI FILE DAN PENGATURAN
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR.parent / "dataset" / "student_lifestyle_dataset.csv"
)

KANDIDAT_PATH = BASE_DIR / "kandidat_terbaik.json"
PROGRES_PATH = BASE_DIR / "progres_test_groq.csv"
HASIL_PATH = BASE_DIR / "hasil_evaluasi_test.csv"
CONFIG_PATH = BASE_DIR / "config.json"
MODEL_INFO_PATH = BASE_DIR / "model_info.json"

FITUR = list(NAMA_FITUR)
TARGET = "Stress_Level"
LABELS = ["High", "Low", "Moderate"]

JEDA_API = 4
TARGET_F1 = 0.70
MAKS_INVALID = 0.05


# ==========================================================
# 2. FUNGSI SIMPAN CHECKPOINT
# ==========================================================

def simpan_checkpoint(data_progres, lokasi):
    """
    Menyimpan seluruh prediksi yang telah berhasil.

    Menggunakan file sementara agar checkpoint lama
    tidak langsung tertimpa jika penyimpanan gagal.
    """

    lokasi = Path(lokasi)

    hasil = pd.DataFrame(
        [
            {
                "baris_asli": int(idx),
                "prediksi": label,
            }
            for idx, label in sorted(data_progres.items())
        ],
        columns=["baris_asli", "prediksi"],
    )

    file_sementara = lokasi.with_suffix(".tmp")

    hasil.to_csv(
        file_sementara,
        index=False,
    )

    os.replace(
        file_sementara,
        lokasi,
    )


# ==========================================================
# 3. TEST CHECKPOINT TANPA GROQ API
# ==========================================================

def test_checkpoint_offline():
    """
    Menguji fungsi simpan dan baca checkpoint.
    Tidak menghubungi Groq dan tidak memakai token.
    Tidak menyentuh checkpoint asli.
    """

    print("\n=== TEST CHECKPOINT OFFLINE ===")

    with tempfile.TemporaryDirectory() as folder:
        lokasi_test = Path(folder) / "checkpoint_test.csv"

        # Simulasi tiga prediksi
        simulasi = {
            10: "High",
            20: "Moderate",
            30: "Low",
        }

        simpan_checkpoint(simulasi, lokasi_test)

        assert lokasi_test.exists(), (
            "GAGAL: File checkpoint tidak dibuat."
        )

        hasil_baca = pd.read_csv(lokasi_test)

        assert len(hasil_baca) == 3, (
            "GAGAL: Jumlah prediksi tidak sesuai."
        )

        # Simulasi melanjutkan proses
        simulasi[40] = "High"

        simpan_checkpoint(simulasi, lokasi_test)

        hasil_lanjut = pd.read_csv(lokasi_test)

        assert len(hasil_lanjut) == 4, (
            "GAGAL: Checkpoint tidak dapat diperbarui."
        )

        assert set(hasil_lanjut["prediksi"]).issubset(LABELS), (
            "GAGAL: Terdapat label tidak valid."
        )

    print("BERHASIL: File checkpoint bisa dibuat.")
    print("BERHASIL: Prediksi bisa disimpan.")
    print("BERHASIL: Checkpoint bisa diperbarui.")
    print("BERHASIL: Checkpoint bisa dibaca kembali.")
    print("Tidak ada permintaan ke Groq API.")


# Jika menjalankan --self-test,
# hanya menguji checkpoint dan langsung keluar.

if "--self-test" in sys.argv:
    test_checkpoint_offline()
    raise SystemExit(0)


# ==========================================================
# 4. MEMUAT KONFIGURASI GROQ
# ==========================================================

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY tidak ditemukan di Modul_2A/.env"
    )

if not KANDIDAT_PATH.exists():
    raise FileNotFoundError(
        "kandidat_terbaik.json tidak ditemukan."
    )

with open(KANDIDAT_PATH, encoding="utf-8") as f:
    cfg = json.load(f)

client = Groq(api_key=api_key)


# ==========================================================
# 5. MEMUAT DATA TESTING YANG SAMA DENGAN MODUL 1A
# ==========================================================

df = pd.read_csv(DATASET_PATH)

_, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[TARGET],
)

test = test.copy()

# Menyimpan indeks asli untuk mencocokkan checkpoint
test["baris_asli"] = test.index.astype(int)

test = test.reset_index(drop=True)

TOTAL_TEST = len(test)


# ==========================================================
# 6. MEMBACA CHECKPOINT SEBELUM MELANJUTKAN
# ==========================================================

if PROGRES_PATH.exists():

    progres = pd.read_csv(PROGRES_PATH)

    if not {"baris_asli", "prediksi"}.issubset(progres.columns):
        raise ValueError(
            "Format progres_test_groq.csv tidak sesuai."
        )

    if progres["baris_asli"].duplicated().any():
        raise ValueError(
            "Checkpoint memiliki indeks duplikat."
        )

    sudah = dict(
        zip(
            progres["baris_asli"].astype(int),
            progres["prediksi"].astype(str),
        )
    )

else:
    sudah = {}


# Validasi checkpoint
indeks_test = set(test["baris_asli"].astype(int))

if not set(sudah).issubset(indeks_test):
    raise ValueError(
        "Checkpoint tidak cocok dengan dataset testing."
    )

if any(label not in LABELS for label in sudah.values()):
    raise ValueError(
        "Checkpoint mengandung label invalid."
    )


print("\n=== TEST FINAL GROQ DENGAN CHECKPOINT ===")

print("Model:", cfg["model"])
print("Metode:", cfg.get("metode", "Konfigurasi terbaik"))
print(f"Progres tersimpan: {len(sudah)}/{TOTAL_TEST}")
print("File checkpoint:", PROGRES_PATH)


# ==========================================================
# 7. PREDIKSI DENGAN SIMPAN OTOMATIS
# ==========================================================

try:

    for nomor, (_, baris) in enumerate(
        test.iterrows(),
        start=1,
    ):

        idx = int(baris["baris_asli"])

        # Lewati data yang sudah berhasil diprediksi
        if idx in sudah:
            continue

        print(
            f"\nMemproses {nomor}/{TOTAL_TEST} ...",
            flush=True,
        )

        try:

            label = classify(
                client,
                cfg,
                baris[FITUR].to_dict(),
            )

        except KeyboardInterrupt:
            raise

        except Exception as error:

            print(
                f"\nAPI error: {type(error).__name__}: {error}"
            )

            print("Proses dihentikan.")
            raise SystemExit(1)

        # Jangan simpan prediksi invalid
        if label not in LABELS:

            print(
                "\nPrediksi invalid atau kuota Groq habis."
            )

            print(
                f"Progres aman: {len(sudah)}/{TOTAL_TEST}"
            )

            raise SystemExit(1)

        # ==================================================
        # BAGIAN TERPENTING:
        # SIMPAN SETELAH SETIAP PREDIKSI BERHASIL
        # ==================================================

        sudah[idx] = label

        simpan_checkpoint(
            sudah,
            PROGRES_PATH,
        )

        print(
            f"TERSIMPAN: {len(sudah)}/{TOTAL_TEST} | {label}",
            flush=True,
        )


        # Jeda hanya jika masih ada data
        if len(sudah) < TOTAL_TEST:
            time.sleep(JEDA_API)

except KeyboardInterrupt:

    print("\nProses dihentikan pengguna.")

    print(
        f"Checkpoint terakhir: {len(sudah)}/{TOTAL_TEST}"
    )

    raise SystemExit(0)


# ==========================================================
# 8. MEMASTIKAN 400 DATA SUDAH LENGKAP
# ==========================================================

if len(sudah) != TOTAL_TEST:

    print(
        f"\nTesting belum selesai: "
        f"{len(sudah)}/{TOTAL_TEST}"
    )

    raise SystemExit(0)


print("\nSEMUA DATA TESTING BERHASIL DIPROSES!")


# ==========================================================
# 9. MENGHITUNG HASIL EVALUASI FINAL
# ==========================================================

test["prediksi"] = test["baris_asli"].map(sudah)

y_true = test[TARGET].tolist()
y_pred = test["prediksi"].tolist()

accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=LABELS,
    average="macro",
    zero_division=0,
)

hasil_test = {
    "accuracy": round(float(accuracy), 4),
    "f1_macro": round(float(macro_f1), 4),
    "invalid": 0,
    "n": TOTAL_TEST,
}

print("\n=== HASIL EVALUASI GROQ ===")
print(hasil_test)

print("\nCLASSIFICATION REPORT:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
)

print("\nCONFUSION MATRIX:")
print(
    pd.DataFrame(
        confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS,
        ),
        index=[f"Asli_{label}" for label in LABELS],
        columns=[f"Pred_{label}" for label in LABELS],
    )
)


# ==========================================================
# 10. PERBANDINGAN DENGAN DNN
# ==========================================================

try:

    with open(MODEL_INFO_PATH, encoding="utf-8") as f:
        dnn = json.load(f)["metrik"]

    print("\n=== PERBANDINGAN DNN VS GROQ ===")

    print(
        pd.DataFrame(
            {
                "DNN Modul 1A": [
                    dnn["accuracy"],
                    dnn["f1_macro"],
                ],
                "Groq Modul 2A": [
                    hasil_test["accuracy"],
                    hasil_test["f1_macro"],
                ],
            },
            index=["Accuracy", "Macro F1"],
        )
    )

    hasil_test["f1_dnn_pembanding"] = dnn["f1_macro"]

except (FileNotFoundError, KeyError, json.JSONDecodeError) as error:

    print("Perbandingan DNN belum tersedia:", error)


# ==========================================================
# 11. MENYIMPAN HASIL TESTING
# ==========================================================

test["benar"] = (
    test[TARGET] == test["prediksi"]
)

test.to_csv(
    HASIL_PATH,
    index=False,
)

print("\nHasil testing tersimpan:", HASIL_PATH.name)


# ==========================================================
# 12. GERBANG EVALUASI
# ==========================================================

lolos_f1 = hasil_test["f1_macro"] >= TARGET_F1

lolos_invalid = (
    hasil_test["invalid"] / TOTAL_TEST
) <= MAKS_INVALID

if lolos_f1 and lolos_invalid:

    config_final = {
        k: v
        for k, v in cfg.items()
        if k not in ["metode", "hasil_validasi"]
    }

    config_final["evaluasi"] = {
        **hasil_test,
        "metode": cfg.get("metode", "-"),
        "dataset": DATASET_PATH.name,
        "tanggal": date.today().isoformat(),
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            config_final,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nLOLOS EVALUASI!")
    print("config.json berhasil dibuat.")

else:

    print("\nBELUM LOLOS EVALUASI.")
    print("config.json tidak dibuat.")

print("\nPROGRAM SELESAI.")