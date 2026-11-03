# ==================================================
# CLASSIFIER.PY
# Klasifikasi Tingkat Stres Mahasiswa menggunakan
# AI Model Service (Groq API)
#
# Digunakan pada Modul 2A dan Modul 2B
# ==================================================

import json
import re
import time
from pathlib import Path

import groq


# ==================================================
# 1. PEMETAAN FITUR DATASET STUDENT LIFESTYLE
# ==================================================

NAMA_FITUR = {
    "Study_Hours_Per_Day": "Jam Belajar per Hari",
    "Extracurricular_Hours_Per_Day": "Jam Ekstrakurikuler per Hari",
    "Sleep_Hours_Per_Day": "Jam Tidur per Hari",
    "Social_Hours_Per_Day": "Jam Bersosialisasi per Hari",
    "Physical_Activity_Hours_Per_Day": "Jam Aktivitas Fisik per Hari",
    "GPA": "Indeks Prestasi Akademik (GPA)"
}


# ==================================================
# 2. MEMUAT KONFIGURASI MODEL LLM
# ==================================================

def load_config(path=None):
    """
    Memuat konfigurasi model Groq dari config.json.

    File config.json akan dibuat setelah proses
    evaluasi Modul 2A berhasil.
    """

    if path is None:
        path = Path(__file__).parent / "config.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==================================================
# 3. MENGUBAH DATA MAHASISWA MENJADI TEKS
# ==================================================

def format_data(data):
    """
    Mengubah enam fitur numerik mahasiswa
    menjadi teks yang dapat dipahami LLM.
    """

    return "; ".join(
        f"{nama}: {data[kolom]}"
        for kolom, nama in NAMA_FITUR.items()
    )


# ==================================================
# 4. MENYUSUN PROMPT UNTUK GROQ
# ==================================================

def build_messages(cfg, data):
    """
    Menyusun pesan untuk LLM:
    - System prompt
    - Contoh few-shot jika tersedia
    - Data mahasiswa yang akan diprediksi
    """

    msgs = [
        {
            "role": "system",
            "content": cfg["system_prompt"]
        }
    ]

    # Menambahkan contoh klasifikasi dari data training
    for ex in cfg.get("few_shot", []):

        msgs.append({
            "role": "user",
            "content": format_data(ex["data"])
        })

        msgs.append({
            "role": "assistant",
            "content": ex["label"]
        })

    # Menambahkan data mahasiswa yang akan diklasifikasi
    msgs.append({
        "role": "user",
        "content": format_data(data)
    })

    return msgs


# ==================================================
# 5. MEMVALIDASI HASIL KLASIFIKASI LLM
# ==================================================

def parse_label(output, labels):
    """
    Merapikan hasil prediksi Groq.

    Contoh:
    'high.'     -> High
    'LOW'       -> Low
    'moderate.' -> Moderate

    Jawaban di luar daftar label -> invalid
    """

    if not isinstance(output, str):
        return "invalid"

    jawab = re.sub(r"[^a-z]", "", output.lower())

    peta = {
        re.sub(r"[^a-z]", "", label.lower()): label
        for label in labels
    }

    return peta.get(jawab, "invalid")


# ==================================================
# 6. PROSES KLASIFIKASI MENGGUNAKAN GROQ API
# ==================================================

def classify(client, cfg, data, max_retry=3):
    """
    Mengirim data mahasiswa ke Groq API
    dan mengembalikan label prediksi.

    Output:
    - Low
    - Moderate
    - High
    - invalid
    """

    for percobaan in range(max_retry):

        try:

            resp = client.chat.completions.create(
                model=cfg["model"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                messages=build_messages(cfg, data)
            )

            output = resp.choices[0].message.content

            hasil = parse_label(
                output,
                cfg["labels"]
            )

            return hasil

        except groq.RateLimitError:

            print(
                f"Batas kuota Groq tercapai. "
                f"Percobaan {percobaan + 1}/{max_retry}"
            )

            if percobaan < max_retry - 1:
                time.sleep(15 * (percobaan + 1))

    return "invalid"