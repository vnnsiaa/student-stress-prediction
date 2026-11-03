import time
from pathlib import Path

import pandas as pd
import streamlit as st
from groq import Groq

from classifier import NAMA_FITUR, classify, load_config


# =====================================================
# 1. KONFIGURASI APLIKASI
# =====================================================

st.set_page_config(
    page_title="Prediksi Tingkat Stres Mahasiswa",
    page_icon="🧠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
SAMPLE_PATH = BASE_DIR / "contoh_data_stres.csv"

FITUR = list(NAMA_FITUR)
LABEL_VALID = ["Low", "Moderate", "High"]

BATAS_CSV = 30


# =====================================================
# 2. MEMUAT KONFIGURASI MODEL
# =====================================================

try:
    cfg = load_config(str(CONFIG_PATH))

except Exception as e:
    st.error(f"Gagal membaca config.json: {e}")
    st.stop()


# =====================================================
# 3. MEMUAT API KEY
# =====================================================

try:
    API_KEY = st.secrets["GROQ_API_KEY"]

    if not API_KEY:
        raise ValueError("API Key kosong")

except Exception:
    st.error(
        "GROQ_API_KEY belum diatur. "
        "Tambahkan API Key pada Streamlit Secrets."
    )
    st.stop()


@st.cache_resource
def get_client(api_key):
    return Groq(api_key=api_key)


client = get_client(API_KEY)


# =====================================================
# 4. FUNGSI PREDIKSI
# =====================================================

@st.cache_data(show_spinner=False)
def prediksi(
    study_hours,
    extracurricular_hours,
    sleep_hours,
    social_hours,
    physical_hours,
    gpa,
):
    """
    Melakukan klasifikasi tingkat stres mahasiswa
    menggunakan konfigurasi LLM terbaik Modul 2A.

    Input yang sama disimpan dalam cache agar
    tidak perlu mengulangi request Groq.
    """

    data = {
        "Study_Hours_Per_Day": float(study_hours),
        "Extracurricular_Hours_Per_Day": float(
            extracurricular_hours
        ),
        "Sleep_Hours_Per_Day": float(sleep_hours),
        "Social_Hours_Per_Day": float(social_hours),
        "Physical_Activity_Hours_Per_Day": float(
            physical_hours
        ),
        "GPA": float(gpa),
    }

    hasil = classify(client, cfg, data)

    if hasil not in LABEL_VALID:
        raise ValueError(
            "Prediksi LLM tidak valid atau API sedang "
            "mengalami pembatasan. Silakan coba kembali."
        )

    return hasil


def prediksi_dari_data(data):
    """
    Mengambil enam fitur sesuai urutan NAMA_FITUR.
    """

    nilai = [float(data[nama]) for nama in FITUR]

    return prediksi(*nilai)


# =====================================================
# 5. SIDEBAR INFORMASI MODEL
# =====================================================

ev = cfg.get("evaluasi", {})

with st.sidebar:

    st.header("🧠 Informasi Model")

    st.write("**Metode:** Large Language Model (LLM)")

    st.write(
        f"**Model:** `{cfg.get('model', 'Tidak diketahui')}`"
    )

    st.write("**API:** Groq")

    st.write(
        f"**Teknik:** {ev.get('metode', 'Few-shot + Decision Tree')}"
    )

    st.divider()

    st.subheader("Hasil Evaluasi")

    if "accuracy" in ev:
        st.metric(
            "Accuracy LLM",
            f"{float(ev['accuracy']) * 100:.2f}%"
        )

    if "f1_macro" in ev:
        st.metric(
            "F1 Macro LLM",
            f"{float(ev['f1_macro']):.4f}"
        )

    if "f1_dnn_pembanding" in ev:
        st.metric(
            "F1 Macro DNN",
            f"{float(ev['f1_dnn_pembanding']):.4f}"
        )

    st.caption(
        f"Jumlah data testing: {ev.get('n', 400)}"
    )

    st.caption(
        f"Tanggal evaluasi: {ev.get('tanggal', '-')}"
    )

    st.divider()

    with st.expander("Lihat System Prompt"):
        st.code(
            cfg.get("system_prompt", ""),
            language=None,
        )


# =====================================================
# 6. HALAMAN UTAMA
# =====================================================

st.title("🧠 Prediksi Tingkat Stres Mahasiswa")

st.write(
    "Aplikasi ini menggunakan Large Language Model (LLM) "
    "melalui Groq API untuk memprediksi tingkat stres "
    "mahasiswa berdasarkan pola aktivitas sehari-hari."
)

st.info(
    "Hasil prediksi terdiri dari tiga kategori: "
    "**Low (Rendah), Moderate (Sedang), "
    "dan High (Tinggi)**."
)

st.caption(
    "Prediksi ini merupakan hasil model berdasarkan dataset "
    "Student Lifestyle, bukan diagnosis medis atau psikologis."
)


tab1, tab2 = st.tabs(
    [
        "👤 Prediksi Manual",
        "📂 Prediksi File CSV",
    ]
)


# =====================================================
# 7. TAB PREDIKSI MANUAL
# =====================================================

with tab1:

    st.subheader("Input Data Mahasiswa")

    st.write(
        "Masukkan data aktivitas mahasiswa "
        "untuk mengetahui prediksi tingkat stres."
    )

    with st.form("form_prediksi"):

        col1, col2 = st.columns(2)

        with col1:

            study_hours = st.number_input(
                "Jam Belajar per Hari",
                min_value=0.0,
                max_value=24.0,
                value=6.0,
                step=0.5,
            )

            extracurricular_hours = st.number_input(
                "Jam Kegiatan Ekstrakurikuler per Hari",
                min_value=0.0,
                max_value=24.0,
                value=2.0,
                step=0.5,
            )

            sleep_hours = st.number_input(
                "Jam Tidur per Hari",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.5,
            )

        with col2:

            social_hours = st.number_input(
                "Jam Bersosialisasi per Hari",
                min_value=0.0,
                max_value=24.0,
                value=2.0,
                step=0.5,
            )

            physical_hours = st.number_input(
                "Jam Aktivitas Fisik per Hari",
                min_value=0.0,
                max_value=24.0,
                value=1.0,
                step=0.5,
            )

            gpa = st.number_input(
                "IPK (GPA)",
                min_value=0.0,
                max_value=4.0,
                value=3.20,
                step=0.01,
            )

        kirim = st.form_submit_button(
            "🔍 Prediksi Tingkat Stres",
            use_container_width=True,
        )

    if kirim:

        total_jam = (
            study_hours
            + extracurricular_hours
            + sleep_hours
            + social_hours
            + physical_hours
        )

        if total_jam > 24:

            st.error(
                "Total jam aktivitas tidak boleh "
                "melebihi 24 jam dalam satu hari."
            )

        else:

            try:

                with st.spinner(
                    "Memproses prediksi menggunakan Groq LLM..."
                ):

                    hasil = prediksi(
                        study_hours,
                        extracurricular_hours,
                        sleep_hours,
                        social_hours,
                        physical_hours,
                        gpa,
                    )

                st.subheader("Hasil Prediksi")

                if hasil == "Low":

                    st.success(
                        "🟢 Tingkat Stres: LOW (Rendah)"
                    )

                elif hasil == "Moderate":

                    st.warning(
                        "🟡 Tingkat Stres: MODERATE (Sedang)"
                    )

                elif hasil == "High":

                    st.error(
                        "🔴 Tingkat Stres: HIGH (Tinggi)"
                    )

                st.write(
                    f"**Hasil klasifikasi LLM:** {hasil}"
                )

            except Exception as e:

                st.error(
                    f"Prediksi gagal: {e}"
                )


# =====================================================
# 8. TAB PREDIKSI CSV
# =====================================================

with tab2:

    st.subheader("Prediksi Banyak Mahasiswa")

    st.write(
        "Unggah file CSV yang memiliki "
        "enam kolom fitur berikut:"
    )

    for fitur in FITUR:
        st.write(f"- `{fitur}`")

    # ==========================================
    # A. DOWNLOAD SAMPEL DATA UNTUK PENGUJIAN
    # ==========================================

    st.write("### Sampel Data untuk Pengujian")

    if SAMPLE_PATH.exists():

        try:
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
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Gagal membaca file sampel: {e}")

    else:

        st.warning(
            "File contoh_data_stres.csv tidak ditemukan "
            "di folder Modul_2B."
        )

    st.divider()

    # ==========================================
    # B. UPLOAD FILE CSV
    # ==========================================

    st.write("### Upload File CSV")

    file = st.file_uploader(
        "Pilih File CSV",
        type=["csv"],
    )

    if file is not None:

        try:

            df = pd.read_csv(file)

            st.write("### Preview Dataset")
            st.dataframe(
                df.head(),
                use_container_width=True,
            )

            kolom_kurang = [
                nama
                for nama in FITUR
                if nama not in df.columns
            ]

            if kolom_kurang:

                st.error(
                    "Kolom berikut tidak ditemukan: "
                    + ", ".join(kolom_kurang)
                )

            elif df.empty:

                st.warning(
                    "File CSV tidak memiliki data."
                )

            else:

                df_sub = df.head(BATAS_CSV).copy()

                st.write(
                    f"Jumlah data yang akan diprediksi: "
                    f"{len(df_sub)}"
                )

                if st.button(
                    "🚀 Prediksi Data CSV",
                    use_container_width=True,
                ):

                    hasil_list = []

                    progress_bar = st.progress(0)

                    for i, (_, baris) in enumerate(
                        df_sub.iterrows(),
                        start=1,
                    ):

                        try:

                            hasil = prediksi_dari_data(
                                baris
                            )

                        except Exception as e:

                            st.error(
                                f"Prediksi baris ke-{i} gagal: {e}. "
                                "Proses dihentikan."
                            )

                            break

                        hasil_list.append(hasil)

                        progress_bar.progress(
                            i / len(df_sub)
                        )

                        if i < len(df_sub):
                            time.sleep(4)

                    if hasil_list:

                        hasil_df = df_sub.iloc[
                            :len(hasil_list)
                        ].copy()

                        hasil_df["Prediksi_Stress_Level"] = (
                            hasil_list
                        )

                        st.session_state["hasil_csv_llm"] = (
                            hasil_df
                        )

        except Exception as e:

            st.error(
                f"Gagal membaca CSV: {e}"
            )

    # ==========================================
    # C. MENAMPILKAN HASIL PREDIKSI
    # ==========================================

    if "hasil_csv_llm" in st.session_state:

        hasil_df = st.session_state["hasil_csv_llm"]

        st.write("### Hasil Prediksi")

        st.dataframe(
            hasil_df,
            use_container_width=True,
        )

        st.write("### Ringkasan Kategori Stres")

        st.bar_chart(
            hasil_df[
                "Prediksi_Stress_Level"
            ].value_counts()
        )

        st.download_button(
            label="⬇️ Unduh Hasil Prediksi",
            data=hasil_df.to_csv(index=False),
            file_name="hasil_prediksi_stres_llm.csv",
            mime="text/csv",
        )


# =====================================================
# 9. FOOTER
# =====================================================

st.divider()

st.caption(
    "Student Stress Prediction | "
    "Modul 2B - Deployment LLM Groq"
)