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
st.caption("Dataset: Student Performance (Kaggle)")

# =========================
# LOAD DATASET (ANTI ERROR)
# =========================
try:
    df = pd.read_csv("student-mat.csv")
except:
    st.error("❌ File student-mat.csv tidak ditemukan")
    st.stop()

# Perbaiki delimiter jika terbaca 1 kolom
if df.shape[1] == 1:
    df = pd.read_csv("student-mat.csv", sep=",")
if df.shape[1] == 1:
    df = pd.read_csv("student-mat.csv", sep=";")

# Normalisasi nama kolom
df.columns = df.columns.str.strip().str.replace('"', '').str.lower()

# Debug kolom
st.caption("Kolom dataset yang terbaca:")
st.write(df.columns.tolist())

# =========================
# VALIDASI KOLUMN
# =========================
required_cols = ["g1", "g2", "g3", "absences"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"❌ Kolom '{col}' tidak ditemukan dalam dataset")
        st.stop()

# =========================
# PENYESUAIAN DATASET
# =========================
df["uts"] = df["g1"] * 5   # 0–20 → 0–100
df["uas"] = df["g2"] * 5

df["pass"] = df["g3"].apply(lambda x: 1 if x >= 10 else 0)

X = df[["absences", "uts", "uas"]]
y = df["pass"]

# =========================
# TRAIN MODEL ML
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
    absences = st.number_input("Jumlah Ketidakhadiran (0–16)", 0, 16, 2)

    submit = st.form_submit_button("🔍 Hitung Kelulusan")

# =========================
# PROSES PENILAIAN
# =========================
if submit:
    TOTAL_PERTEMUAN = 16

    nilai_kehadiran = ((TOTAL_PERTEMUAN - absences) / TOTAL_PERTEMUAN) * 100
    nilai_kehadiran = max(0, min(nilai_kehadiran, 100))

    nilai_akhir = (
        0.30 * uts +
        0.45 * uas +
        0.15 * tugas +
        0.10 * nilai_kehadiran
    )

    st.divider()
    st.subheader("📊 Hasil Evaluasi Kelulusan")

    st.write(f"👤 **Nama**: {nama}")
    st.write(f"🆔 **NIM**: {nim}")
    st.write(f"📌 **Nilai Kehadiran**: {nilai_kehadiran:.2f}")
    st.write(f"📌 **Nilai Akhir**: {nilai_akhir:.2f}")

    if nilai_akhir >= 70:
        st.success("✅ **MAHASISWA DINYATAKAN LULUS** 🎓")
    else:
        st.error("❌ **MAHASISWA DINYATAKAN TIDAK LULUS**")

    # =========================
    # ANALISIS ML (PENDUKUNG)
    # =========================
    prob = model.predict_proba([[absences, uts, uas]])

    st.subheader("🤖 Analisis Machine Learning (Pendukung)")
    st.dataframe(pd.DataFrame({
        "Status": ["Tidak Lulus", "Lulus"],
        "Probabilitas (%)": [
            round(prob[0][0] * 100, 2),
            round(prob[0][1] * 100, 2)
        ]
    }))

    st.info(
        "📌 Keputusan kelulusan ditentukan oleh nilai akhir gabungan (≥ 70). "
        "Model Machine Learning digunakan sebagai analisis pendukung."
    )
