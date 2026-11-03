
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ==================================================
# 1. KONFIGURASI APLIKASI
# ==================================================

st.set_page_config(
    page_title="Student Stress Prediction",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "model_student_stress.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
INFO_PATH = BASE_DIR / "model_info.json"
SAMPLE_PATH = BASE_DIR / "contoh_data_stres.csv"

# Dataset Kaggle berada satu folder di atas Modul_1B
DATASET_PATH = (
    BASE_DIR.parent / "dataset" / "student_lifestyle_dataset.csv"
)


# ==================================================
# 2. MEMUAT MODEL DAN INFORMASI MODEL
# ==================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    with open(INFO_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)

    return model, encoder, info


# Memeriksa ketersediaan file model
file_model = [
    MODEL_PATH,
    ENCODER_PATH,
    INFO_PATH
]

if not all(path.exists() for path in file_model):
    st.error(
        "File model tidak ditemukan. Pastikan "
        "model_student_stress.pkl, label_encoder.pkl, "
        "dan model_info.json berada dalam folder yang sama "
        "dengan app.py."
    )
    st.stop()


try:
    model, encoder, info = load_model()

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


FITUR = info["fitur"]
RENTANG = info["rentang_fitur"]

NAMA_FITUR = {
    "Study_Hours_Per_Day": "Jam Belajar per Hari",
    "Extracurricular_Hours_Per_Day": "Jam Ekstrakurikuler per Hari",
    "Sleep_Hours_Per_Day": "Jam Tidur per Hari",
    "Social_Hours_Per_Day": "Jam Bersosialisasi per Hari",
    "Physical_Activity_Hours_Per_Day": "Jam Aktivitas Fisik per Hari",
    "GPA": "GPA"
}


# ==================================================
# 3. SIDEBAR: INFORMASI MODEL
# ==================================================

with st.sidebar:

    st.header("🧠 Tentang Model")

    st.write(info["nama_model"])

    st.metric(
        "Accuracy (Data Uji)",
        f"{info['metrik']['accuracy']:.1%}"
    )

    st.metric(
        "F1 Macro (Data Uji)",
        f"{info['metrik']['f1_macro']:.4f}"
    )

    st.caption(
        f"Dievaluasi pada {info['metrik']['n_test']} data uji"
    )

    st.divider()

    st.write("**Dataset:** Student Stress Prediction")

    st.write(f"**Jumlah fitur:** {len(FITUR)}")

    st.caption(
        f"Scikit-learn {info['versi']['scikit-learn']}"
    )


# ==================================================
# 4. HALAMAN UTAMA
# ==================================================

st.title("🧠 Student Stress Prediction")

st.write(
    "Aplikasi untuk memprediksi tingkat stres mahasiswa "
    "berdasarkan kebiasaan belajar, tidur, aktivitas sosial, "
    "aktivitas fisik, kegiatan ekstrakurikuler, dan GPA "
    "menggunakan model Deep Neural Network (DNN)."
)

st.info(
    "Masukkan data mahasiswa untuk memperoleh "
    "prediksi tingkat stres secara otomatis."
)


# ==================================================
# 5. MEMBUAT TAB
# ==================================================

tab1, tab2 = st.tabs([
    "👤 Prediksi Satu Mahasiswa",
    "📁 Prediksi dari File CSV"
])


# ==================================================
# 6. TAB 1: PREDIKSI SATU MAHASISWA
# ==================================================

with tab1:

    st.subheader("Data Mahasiswa")

    st.write(
        "Masukkan enam indikator gaya hidup mahasiswa "
        "pada formulir berikut."
    )

    with st.form("form_prediksi"):

        col1, col2 = st.columns(2)

        data_input = {}

        for i, fitur in enumerate(FITUR):

            batas_min = float(RENTANG[fitur][0])
            batas_max = float(RENTANG[fitur][1])

            nilai_default = round(
                (batas_min + batas_max) / 2, 2
            )

            langkah = 0.01 if fitur == "GPA" else 0.1

            kolom = col1 if i % 2 == 0 else col2

            with kolom:

                data_input[fitur] = st.number_input(
                    NAMA_FITUR.get(fitur, fitur),
                    min_value=batas_min,
                    max_value=batas_max,
                    value=nilai_default,
                    step=langkah,
                    key=f"input_{fitur}"
                )

        kirim = st.form_submit_button(
            "🔍 Prediksi Tingkat Stres",
            use_container_width=True
        )

    if kirim:

        data = pd.DataFrame([data_input])[FITUR]

        try:

            hasil_encoded = model.predict(data).astype(int)

            hasil = encoder.inverse_transform(
                hasil_encoded
            )[0]

            st.divider()
            st.subheader("Hasil Prediksi")

            if hasil == "Low":

                st.success(
                    f"Prediksi Tingkat Stres: **{hasil}**"
                )

            elif hasil == "Moderate":

                st.warning(
                    f"Prediksi Tingkat Stres: **{hasil}**"
                )

            else:

                st.error(
                    f"Prediksi Tingkat Stres: **{hasil}**"
                )

            st.caption(
                "Hasil merupakan prediksi model berdasarkan "
                "pola data, bukan diagnosis klinis."
            )

            # Probabilitas prediksi
            if hasattr(model, "predict_proba"):

                probabilitas = model.predict_proba(data)[0]

                nama_kelas = encoder.inverse_transform(
                    model.classes_.astype(int)
                )

                hasil_probabilitas = pd.DataFrame({
                    "Tingkat Stres": nama_kelas,
                    "Probabilitas (%)": [
                        round(float(p) * 100, 2)
                        for p in probabilitas
                    ]
                })

                st.subheader("Probabilitas Prediksi")

                st.bar_chart(
                    hasil_probabilitas.set_index(
                        "Tingkat Stres"
                    )
                )

                st.dataframe(
                    hasil_probabilitas,
                    hide_index=True,
                    use_container_width=True
                )

            with st.expander("Lihat Data Input"):

                st.dataframe(
                    data,
                    hide_index=True,
                    use_container_width=True
                )

        except Exception as e:

            st.error(f"Prediksi gagal dilakukan: {e}")


# ==================================================
# 7. TAB 2: PREDIKSI BANYAK DATA DARI CSV
# ==================================================

with tab2:

    st.subheader("Prediksi Data Mahasiswa melalui CSV")

    st.write(
        "Unggah file CSV untuk memprediksi tingkat stres "
        "mahasiswa secara otomatis."
    )

    st.caption(
        "File harus memiliki enam kolom fitur sesuai "
        "dataset Student Stress Prediction."
    )


    # --------------------------------------------------
    # A. DOWNLOAD SAMPEL DATA UNTUK PENGUJIAN
    # --------------------------------------------------

    st.write("### Sampel Data untuk Pengujian")

    if SAMPLE_PATH.exists():

        contoh_data = pd.read_csv(SAMPLE_PATH)

        st.write(
            f"Tersedia {len(contoh_data)} sampel mahasiswa "
            "untuk mencoba fitur prediksi."
        )

        st.download_button(
            label="⬇️ Download Sampel Data Uji",
            data=SAMPLE_PATH.read_bytes(),
            file_name="contoh_data_stres.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.warning(
            "File contoh_data_stres.csv tidak ditemukan."
        )

    st.divider()

    # --------------------------------------------------
    # B. UPLOAD FILE CSV
    # --------------------------------------------------

    st.write("### Upload Data Mahasiswa")

    file = st.file_uploader(
        "Pilih file CSV",
        type=["csv"]
    )

    if file is not None:

        try:

            df = pd.read_csv(file)

            st.write("### Preview Data yang Diunggah")

            st.dataframe(
                df,
                use_container_width=True
            )

            st.caption(
                f"Jumlah data: {len(df)} mahasiswa"
            )

            # Memeriksa kolom wajib
            kurang = [
                fitur for fitur in FITUR
                if fitur not in df.columns
            ]

            if df.empty:

                st.warning(
                    "File CSV tidak memiliki data."
                )

            elif kurang:

                st.error(
                    "Kolom berikut tidak ditemukan: "
                    + ", ".join(kurang)
                )

            else:

                data_csv = df[FITUR].copy()

                # Memastikan semua fitur berupa angka
                for fitur in FITUR:

                    data_csv[fitur] = pd.to_numeric(
                        data_csv[fitur],
                        errors="coerce"
                    )

                # Memeriksa nilai kosong dan bukan angka
                if (
                    data_csv.isna().any().any()
                    or not np.isfinite(
                        data_csv.to_numpy(dtype=float)
                    ).all()
                ):

                    st.error(
                        "Terdapat nilai kosong, bukan angka, "
                        "atau nilai tidak valid pada data. "
                        "Periksa kembali file CSV."
                    )

                else:

                    # Memeriksa rentang fitur
                    kesalahan = []

                    for fitur in FITUR:

                        batas_min = float(RENTANG[fitur][0])
                        batas_max = float(RENTANG[fitur][1])

                        if (
                            (data_csv[fitur] < batas_min).any()
                            or
                            (data_csv[fitur] > batas_max).any()
                        ):

                            kesalahan.append(
                                f"{fitur}: rentang harus "
                                f"{batas_min} - {batas_max}"
                            )

                    if kesalahan:

                        for pesan in kesalahan:
                            st.warning(pesan)

                        st.error(
                            "Periksa kembali nilai pada file CSV."
                        )

                    elif st.button(
                        "🔍 Prediksi Semua Data",
                        use_container_width=True
                    ):

                        # ----------------------------------
                        # C. MELAKUKAN PREDIKSI
                        # ----------------------------------

                        hasil_encoded = model.predict(
                            data_csv
                        ).astype(int)

                        hasil_label = encoder.inverse_transform(
                            hasil_encoded
                        )

                        df_hasil = df.copy()

                        df_hasil["prediksi_stres"] = hasil_label

                        st.success(
                            "Prediksi seluruh data berhasil!"
                        )

                        # Menampilkan hasil prediksi
                        st.write("### Hasil Prediksi")

                        st.dataframe(
                            df_hasil,
                            use_container_width=True
                        )


                        # ----------------------------------
                        # D. RINGKASAN HASIL PREDIKSI
                        # ----------------------------------

                        st.write(
                            "### Ringkasan Tingkat Stres Mahasiswa"
                        )

                        jumlah_status = (
                            df_hasil["prediksi_stres"]
                            .value_counts()
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.metric(
                                "High",
                                int(jumlah_status.get("High", 0))
                            )

                        with col2:

                            st.metric(
                                "Moderate",
                                int(jumlah_status.get("Moderate", 0))
                            )

                        with col3:

                            st.metric(
                                "Low",
                                int(jumlah_status.get("Low", 0))
                            )


                        # Grafik distribusi hasil
                        distribusi = (
                            df_hasil["prediksi_stres"]
                            .value_counts()
                            .reindex(
                                ["High", "Moderate", "Low"],
                                fill_value=0
                            )
                            .rename_axis("Tingkat Stres")
                            .reset_index(name="Jumlah Mahasiswa")
                        )

                        st.bar_chart(
                            distribusi.set_index(
                                "Tingkat Stres"
                            )
                        )


                        # ----------------------------------
                        # E. DOWNLOAD HASIL PREDIKSI
                        # ----------------------------------

                        st.write("### Download Hasil Prediksi")

                        hasil_csv = df_hasil.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="⬇️ Download Hasil Prediksi CSV",
                            data=hasil_csv,
                            file_name="hasil_prediksi_student_stress.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as e:

            st.error(
                f"Terjadi kesalahan saat memproses CSV: {e}"
            )
