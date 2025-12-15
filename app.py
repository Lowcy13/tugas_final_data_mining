import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Evaluasi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Evaluasi Kelulusan Mata Kuliah Matematika")
st.caption("Berbasis Dataset Student Performance (Kaggle)")

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("student-mat.csv", sep=";")
df.columns = df.columns.str.strip()

# =========================
# PENYESUAIAN DATASET
# =========================
# Konversi nilai dari skala 0–20 → 0–100
df["UTS"] = df["G1"] * 5
df["UAS"] = df["G2"] * 5

# Label kelulusan sesuai dataset
df["pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

X = df[["absences", "UTS", "UAS"]]
y = df["pass"]

# =========================
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

st.metric("📈 Akurasi Model (Dataset Asli)", f"{accuracy*100:.2f}%")
st.divider()

# =========================
# FORM INPUT MAHASISWA
# =========================
st.subheader("📝 Input Data Mahasiswa")

with st.form("form_mahasiswa"):
    nama = st.text_input("Nama Mahasiswa")
    nim = st.text_input("NIM")

    uts = st.number_input("Nilai UTS (0–100)", 0, 100, 75)
    uas = st.number_input("Nilai UAS (0–100)", 0, 100, 80)
    tugas = st.number_input("Nilai Tugas (0–100)", 0, 100, 85)
    absences = st.number_input("Jumlah Ketidakhadiran", 0, 50, 3)

    submit = st.form_submit_button("🔍 Hitung Kelulusan")

# =========================
# PROSES PENILAIAN
# =========================
if submit:
    # Nilai kehadiran
    nilai_kehadiran = ((50 - absences) / 50) * 100
    nilai_kehadiran = max(0, min(nilai_kehadiran, 100))

    # Nilai akhir gabungan
    nilai_akhir = (
        0.30 * uts +
        0.40 * uas +
        0.20 * tugas +
        0.10 * nilai_kehadiran
    )

    st.divider()
    st.subheader("📊 Hasil Evaluasi")

    st.write(f"👤 Nama: **{nama}**")
    st.write(f"🆔 NIM: **{nim}**")
    st.write(f"📌 Nilai Kehadiran: **{nilai_kehadiran:.2f}**")
    st.write(f"📌 Nilai Akhir: **{nilai_akhir:.2f}**")

    if nilai_akhir >= 70:
        st.success("✅ **MAHASISWA DINYATAKAN LULUS** 🎓")
    else:
        st.error("❌ **MAHASISWA DINYATAKAN TIDAK LULUS**")

    # =========================
    # PREDIKSI MODEL ML
    # =========================
    pred_prob = model.predict_proba([[absences, uts, uas]])

    st.subheader("🤖 Analisis Model Machine Learning")
    st.dataframe(pd.DataFrame({
        "Status": ["Tidak Lulus", "Lulus"],
        "Probabilitas (%)": [
            round(pred_prob[0][0] * 100, 2),
            round(pred_prob[0][1] * 100, 2)
        ]
    }))

    st.info(
        "📌 Keputusan kelulusan ditentukan oleh nilai akhir gabungan (≥ 70). "
        "Model Machine Learning digunakan sebagai analisis pendukung berdasarkan dataset."
    )
