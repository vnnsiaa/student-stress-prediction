# ============================================================
# MODUL 2A - STUDENT STRESS PREDICTION
# EVALUASI AI MODEL SERVICE MENGGUNAKAN GROQ API
#
# Metode:
# 1. Zero-shot
# 2. Few-shot
# 3. Few-shot + Profil Median Kelas
# 4. Few-shot + Decision Tree
#
# Dataset: student_lifestyle_dataset.csv
# Target : Stress_Level (High, Moderate, Low)
# ============================================================

import json
import os
import time
from datetime import date
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv
from groq import Groq

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from sklearn.model_selection import train_test_split

from sklearn.tree import (
    DecisionTreeClassifier,
    export_text,
)

from classifier import NAMA_FITUR, classify


# ============================================================
# 1. KONFIGURASI UTAMA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR.parent
    / "dataset"
    / "student_lifestyle_dataset.csv"
)

MODEL_INFO_PATH = BASE_DIR / "model_info.json"

KANDIDAT_PATH = BASE_DIR / "kandidat_terbaik.json"

CONFIG_PATH = BASE_DIR / "config.json"

HASIL_VALIDASI_PATH = (
    BASE_DIR / "hasil_validasi_groq.csv"
)

HASIL_TEST_PATH = (
    BASE_DIR / "hasil_evaluasi_test.csv"
)


# ============================================================
# PENGATURAN TAHAP
# ============================================================

# TAHAP = "validasi"
# Untuk membandingkan 4 iterasi menggunakan 30 data.
#
# TAHAP = "test"
# Untuk menguji konfigurasi terbaik pada 400 data.
#
# JALANKAN VALIDASI TERLEBIH DAHULU.

TAHAP = "test"


# Model Groq yang sudah berhasil digunakan

MODEL_GROQ = "qwen/qwen3.8-27b"


# Jeda antarpemanggilan API
# Sesuaikan jika mengalami batas kuota.

JEDA_API = 4


# Kriteria evaluasi final

TARGET_F1 = 0.70

MAKS_INVALID = 0.05


# ============================================================
# 2. KONEKSI KE GROQ API
# ============================================================

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY tidak ditemukan. "
        "Periksa file .env di folder Modul_2A."
    )

client = Groq(api_key=API_KEY)

print("\n==========================================")
print("MODUL 2A - STUDENT STRESS PREDICTION")
print("==========================================")

print("Koneksi Groq siap")
print("Model:", MODEL_GROQ)


# ============================================================
# 3. MEMUAT DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

FITUR = list(NAMA_FITUR.keys())

TARGET = "Stress_Level"

LABELS = [
    "High",
    "Low",
    "Moderate",
]


# Pastikan seluruh kolom tersedia

KOLOM_WAJIB = FITUR + [TARGET]

KOLOM_HILANG = [
    kolom
    for kolom in KOLOM_WAJIB
    if kolom not in df.columns
]

if KOLOM_HILANG:
    raise ValueError(
        f"Kolom tidak ditemukan: {KOLOM_HILANG}"
    )


# ============================================================
# PEMBAGIAN DATA SAMA DENGAN MODUL 1A
# ============================================================

# Training : 1600
# Testing  : 400

train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[TARGET],
)


# Validasi: 10 mahasiswa setiap kelas
# Total 30 data

val = train.groupby(TARGET).sample(
    n=10,
    random_state=7,
)


# Pool training untuk contoh dan profil
# Tidak menggunakan data validasi maupun testing

pool = train.drop(val.index)


print("\n===== INFORMASI DATASET =====")

print("Nama dataset :", DATASET_PATH.name)
print("Total data   :", len(df))
print("Training     :", len(train))
print("Validasi     :", len(val))
print("Pool         :", len(pool))
print("Testing      :", len(test))
print("Jumlah fitur :", len(FITUR))
print("Target       :", TARGET)
print("Label        :", LABELS)


# ============================================================
# 4. FUNGSI EVALUASI
# ============================================================

def evaluasi(cfg, data, nama_pengujian):

    print("\n" + "=" * 55)
    print(nama_pengujian)
    print("=" * 55)

    prediksi = []

    for i, (_, baris) in enumerate(
        data.iterrows(),
        start=1,
    ):

        data_mahasiswa = (
            baris[FITUR].to_dict()
        )

        hasil_prediksi = classify(
            client,
            cfg,
            data_mahasiswa,
        )

        prediksi.append(
            hasil_prediksi
        )

        print(
            f"\rMemproses {i}/{len(data)} "
            f"| Hasil: {hasil_prediksi}",
            end="",
            flush=True,
        )

        if i < len(data):
            time.sleep(JEDA_API)

    print("\n")

    y_true = data[TARGET].tolist()

    accuracy = accuracy_score(
        y_true,
        prediksi,
    )

    macro_f1 = f1_score(
        y_true,
        prediksi,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    jumlah_invalid = prediksi.count(
        "invalid"
    )

    persen_invalid = (
        jumlah_invalid / len(data)
    )

    hasil = {
        "accuracy": round(
            float(accuracy),
            4,
        ),
        "f1_macro": round(
            float(macro_f1),
            4,
        ),
        "invalid": jumlah_invalid,
        "persen_invalid": round(
            persen_invalid,
            4,
        ),
        "n": len(data),
    }

    print("===== HASIL EVALUASI =====")
    print(hasil)

    print("\nCLASSIFICATION REPORT:")

    print(
        classification_report(
            y_true,
            prediksi,
            labels=LABELS,
            zero_division=0,
        )
    )

    print("\nCONFUSION MATRIX:")

    cm = confusion_matrix(
        y_true,
        prediksi,
        labels=LABELS,
    )

    tabel_cm = pd.DataFrame(
        cm,
        index=[
            f"Asli_{label}"
            for label in LABELS
        ],
        columns=[
            f"Pred_{label}"
            for label in LABELS
        ],
    )

    print(tabel_cm)

    return hasil, prediksi


# ============================================================
# 5. ITERASI 1 - ZERO-SHOT
# ============================================================

PROMPT_DASAR = """
Kamu adalah sistem klasifikasi tingkat stres mahasiswa
berdasarkan pola gaya hidup sehari-hari.

Tugasmu adalah memprediksi Stress_Level berdasarkan
enam fitur berikut:

1. Study_Hours_Per_Day:
   Jumlah jam belajar per hari.

2. Extracurricular_Hours_Per_Day:
   Jumlah jam ekstrakurikuler per hari.

3. Sleep_Hours_Per_Day:
   Jumlah jam tidur per hari.

4. Social_Hours_Per_Day:
   Jumlah jam bersosialisasi per hari.

5. Physical_Activity_Hours_Per_Day:
   Jumlah jam aktivitas fisik per hari.

6. GPA:
   Indeks prestasi akademik mahasiswa.

Terdapat tiga kelas:

High:
Tingkat stres tinggi.

Moderate:
Tingkat stres sedang.

Low:
Tingkat stres rendah.

Gunakan kombinasi seluruh indikator
untuk memprediksi kategori yang sesuai.

Klasifikasi ini berdasarkan data gaya hidup,
bukan diagnosis medis atau psikologis.

ATURAN OUTPUT:

Jawab HANYA dengan salah satu label berikut:

High
Moderate
Low

Jangan memberikan penjelasan.
Jangan menambahkan tanda baca.
Jangan menulis teks selain label.
""".strip()


cfg_v1 = {
    "model": MODEL_GROQ,
    "temperature": 0,
    "max_tokens": 128,
    "labels": LABELS,
    "system_prompt": PROMPT_DASAR,
    "few_shot": [],
}


# ============================================================
# 6. ITERASI 2 - FEW-SHOT
# ============================================================

# Mengambil 2 contoh dari setiap kelas
# menggunakan data pool training.

contoh = pool.groupby(TARGET).sample(
    n=2,
    random_state=1,
)


few_shot = []

for _, baris in contoh.iterrows():

    few_shot.append(
        {
            "data": baris[FITUR].to_dict(),
            "label": str(baris[TARGET]),
        }
    )


cfg_v2 = {
    **cfg_v1,
    "few_shot": few_shot,
}


# ============================================================
# 7. ITERASI 3 - FEW-SHOT + PROFIL MEDIAN
# ============================================================

# Menghitung median setiap fitur
# untuk masing-masing kelas.
#
# Sumber data hanya pool training.

median = (
    pool.groupby(TARGET)[FITUR]
    .median()
)


profil_baris = []

for label in LABELS:

    profil_baris.append(
        f"\nKelas {label}:"
    )

    for fitur in FITUR:

        nilai = median.loc[
            label,
            fitur,
        ]

        profil_baris.append(
            f"- {NAMA_FITUR[fitur]}: "
            f"{nilai:.2f}"
        )


PROFIL_KELAS = "\n".join(
    profil_baris
)


cfg_v3 = {
    **cfg_v2,

    "system_prompt": (
        PROMPT_DASAR
        + "\n\nPROFIL MEDIAN KELAS "
        + "BERDASARKAN DATA TRAINING:\n"
        + PROFIL_KELAS
        + "\n\nBandingkan seluruh fitur mahasiswa "
        + "dengan profil median setiap kelas. "
        + "Gunakan juga contoh few-shot "
        + "untuk memilih label yang sesuai."
    ),
}


# ============================================================
# 8. ITERASI 4 - FEW-SHOT + DECISION TREE
# ============================================================

# Decision Tree hanya mempelajari pool training.
# Aturannya dimasukkan ke prompt LLM.
#
# Prediksi akhir tetap dihasilkan oleh Groq,
# bukan langsung oleh Decision Tree.

tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=15,
    random_state=42,
)

tree.fit(
    pool[FITUR],
    pool[TARGET],
)


aturan = export_text(
    tree,
    feature_names=FITUR,
)


print("\n===== ATURAN DECISION TREE =====")

print(aturan)


cfg_v4 = {
    **cfg_v2,

    "system_prompt": (
        PROMPT_DASAR
        + "\n\nATURAN KLASIFIKASI "
        + "BERDASARKAN DATA TRAINING:\n"
        + aturan
        + "\n\nGunakan percabangan aturan di atas "
        + "sesuai nilai fitur mahasiswa. "
        + "Contoh few-shot dapat membantu "
        + "menentukan label akhir. "
        + "Jawab hanya High, Moderate, atau Low."
    ),
}


# ============================================================
# 9. DAFTAR KONFIGURASI
# ============================================================

KANDIDAT = [
    (
        "Iterasi 1 - Zero-shot",
        cfg_v1,
    ),
    (
        "Iterasi 2 - Few-shot",
        cfg_v2,
    ),
    (
        "Iterasi 3 - Few-shot + Profil Median",
        cfg_v3,
    ),
    (
        "Iterasi 4 - Few-shot + Decision Tree",
        cfg_v4,
    ),
]


# ============================================================
# 10. TAHAP VALIDASI
# ============================================================

if TAHAP == "validasi":

    print("\n==========================================")
    print("MULAI VALIDASI 4 ITERASI")
    print("==========================================")

    semua_hasil = []

    for nama_iterasi, cfg in KANDIDAT:

        hasil, _ = evaluasi(
            cfg,
            val,
            nama_iterasi,
        )

        semua_hasil.append(
            {
                "iterasi": nama_iterasi,
                "model": cfg["model"],
                **hasil,
            }
        )


    # ========================================================
    # TABEL PERBANDINGAN VALIDASI
    # ========================================================

    tabel_validasi = pd.DataFrame(
        semua_hasil
    )

    print("\n===== PERBANDINGAN VALIDASI =====")

    print(
        tabel_validasi.to_string(
            index=False
        )
    )


    # Simpan seluruh hasil validasi

    tabel_validasi.to_csv(
        HASIL_VALIDASI_PATH,
        index=False,
    )


    # ========================================================
    # PILIH KONFIGURASI TERBAIK
    # ========================================================

    # Prioritas:
    # 1. Macro F1 tertinggi
    # 2. Invalid paling sedikit
    # 3. Accuracy tertinggi

    idx_terbaik = max(
        range(len(semua_hasil)),
        key=lambda i: (
            semua_hasil[i]["f1_macro"],
            -semua_hasil[i]["invalid"],
            semua_hasil[i]["accuracy"],
        ),
    )


    nama_terbaik, cfg_terbaik = (
        KANDIDAT[idx_terbaik]
    )

    hasil_terbaik = (
        semua_hasil[idx_terbaik]
    )


    print("\n==========================================")
    print("KONFIGURASI TERBAIK")
    print("==========================================")

    print("Metode   :", nama_terbaik)

    print(
        "Accuracy :",
        hasil_terbaik["accuracy"],
    )

    print(
        "Macro F1 :",
        hasil_terbaik["f1_macro"],
    )

    print(
        "Invalid  :",
        hasil_terbaik["invalid"],
    )


    # Jangan menyimpan pilihan yang
    # semuanya menghasilkan prediksi invalid.

    if hasil_terbaik["invalid"] == len(val):

        print(
            "\nSEMUA PREDIKSI INVALID."
        )

        print(
            "Periksa model, max_tokens, "
            "dan keluaran classifier.py."
        )

        print(
            "Tidak membuat kandidat_terbaik.json."
        )

    else:

        # Simpan konfigurasi terpilih.
        # File ini BELUM config final Modul 2B.

        kandidat_simpan = {
            **cfg_terbaik,
            "metode": nama_terbaik,
            "hasil_validasi": hasil_terbaik,
        }

        with open(
            KANDIDAT_PATH,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                kandidat_simpan,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "\nKonfigurasi terbaik disimpan:"
        )

        print(
            KANDIDAT_PATH.name
        )


    print("\nVALIDASI SELESAI.")

    print(
        "File hasil:",
        HASIL_VALIDASI_PATH.name,
    )

    print(
        "\nPeriksa hasil sebelum "
        "melanjutkan ke 400 data testing."
    )


# ============================================================
# 11. TAHAP TEST FINAL
# ============================================================

elif TAHAP == "test":

    # Pastikan validasi sudah dilakukan

    if not KANDIDAT_PATH.exists():

        raise FileNotFoundError(
            "kandidat_terbaik.json tidak ditemukan. "
            "Jalankan TAHAP = 'validasi' dahulu."
        )


    # Muat prompt terbaik hasil validasi

    with open(
        KANDIDAT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        cfg_final = json.load(f)


    print("\n==========================================")
    print("PENGUJIAN FINAL - 400 DATA")
    print("==========================================")

    print(
        "Metode terpilih:",
        cfg_final["metode"],
    )


    # ========================================================
    # EVALUASI TEST
    # ========================================================

    hasil_test, pred_test = evaluasi(
        cfg_final,
        test,
        "PENGUJIAN FINAL GROQ",
    )


    # Simpan hasil prediksi test

    hasil_df = test.copy()

    hasil_df["prediksi"] = pred_test

    hasil_df["benar"] = (
        hasil_df[TARGET]
        == hasil_df["prediksi"]
    )

    hasil_df.to_csv(
        HASIL_TEST_PATH,
        index=False,
    )


    print(
        "\nHasil test disimpan:",
        HASIL_TEST_PATH.name,
    )


    # ========================================================
    # 12. PERBANDINGAN DNN VS GROQ
    # ========================================================

    print("\n==========================================")
    print("PERBANDINGAN MODUL 1A VS MODUL 2A")
    print("==========================================")

    try:

        with open(
            MODEL_INFO_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            model_info = json.load(f)

        dnn = model_info["metrik"]


        tabel_perbandingan = pd.DataFrame(
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

            index=[
                "Accuracy",
                "Macro F1",
            ],
        )


        print(
            tabel_perbandingan
        )


        hasil_test["f1_dnn_pembanding"] = (
            dnn["f1_macro"]
        )


    except (
        FileNotFoundError,
        KeyError,
        json.JSONDecodeError,
    ) as error:

        print(
            "Gagal membaca model_info.json:",
            error,
        )


    # ========================================================
    # 13. GERBANG EVALUASI
    # ========================================================

    print("\n==========================================")
    print("GERBANG EVALUASI")
    print("==========================================")

    print(
        "Target Macro F1:",
        TARGET_F1,
    )

    print(
        "Macro F1 Groq:",
        hasil_test["f1_macro"],
    )

    print(
        "Maks invalid:",
        f"{MAKS_INVALID:.0%}",
    )

    print(
        "Invalid Groq:",
        f"{hasil_test['persen_invalid']:.2%}",
    )


    lolos_f1 = (
        hasil_test["f1_macro"]
        >= TARGET_F1
    )

    lolos_invalid = (
        hasil_test["persen_invalid"]
        <= MAKS_INVALID
    )


    # ========================================================
    # 14. SIMPAN CONFIG JIKA LOLOS
    # ========================================================

    if lolos_f1 and lolos_invalid:

        # Buang metadata internal validasi

        config_simpan = {
            k: v
            for k, v in cfg_final.items()
            if k not in [
                "metode",
                "hasil_validasi",
            ]
        }


        # Tambahkan hasil evaluasi final

        config_simpan["evaluasi"] = {
            **hasil_test,
            "metode": cfg_final["metode"],
            "dataset": DATASET_PATH.name,
            "tanggal": date.today().isoformat(),
        }


        # Simpan config untuk Streamlit Modul 2B

        with open(
            CONFIG_PATH,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                config_simpan,
                f,
                indent=2,
                ensure_ascii=False,
            )


        print("\n==========================================")
        print("MODEL GROQ LOLOS EVALUASI")
        print("==========================================")

        print(
            "config.json berhasil dibuat."
        )

        print(
            "Konfigurasi siap digunakan "
            "pada Modul 2B."
        )


    else:

        print("\n==========================================")
        print("MODEL GROQ BELUM LOLOS EVALUASI")
        print("==========================================")

        print(
            "Perlu memperbaiki prompt "
            "atau mencoba model lain."
        )

        print(
            "config.json tidak dibuat."
        )


# ============================================================
# 15. VALIDASI PILIHAN TAHAP
# ============================================================

else:

    raise ValueError(
        "TAHAP harus 'validasi' atau 'test'."
    )


print("\nPROGRAM SELESAI.")