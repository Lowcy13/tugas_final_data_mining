import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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

# Perbaiki delimiter jika perlu
if df.shape[1] == 1:
    df = pd.read_csv("student-mat.csv", sep=",")
if df.shape[1] == 1:
    df = pd.read_csv("student-mat.csv", sep=";")

# Normalisasi nama kolom
df.columns = df.columns.str.strip().str.replace('"', '').str.lower()

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
df["uts"] = df["g1"] * 5        # 0–20 → 0–100
df["uas"] = df["g2"] * 5
df["pass"] = df["g3"].apply(lambda x: 1 if x >= 10 else 0)

X = df[["absences", "uts", "uas"]]
y = df["pass"]

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# =========================
# AKURASI MODEL
# =========================
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
knn_acc = accuracy_score(y_test, knn_model.predict(X_test))

st.subheader("📈 Akurasi Model")
st.metric("Random Forest", f"{rf_acc*100:.2f}%")
st.metric("KNN", f"{knn_acc*100:.2f}%")

# =========================
# VISUALISASI AKURASI
# =========================
st.subheader("📊 Perbandingan Akurasi Algoritma")

akurasi_df = pd.DataFrame({
    "Algoritma": ["Random Forest", "KNN"],
    "Akurasi (%)": [rf_acc * 100, knn_acc * 100]
})

st.bar_chart(akurasi_df.set_index("Algoritma"))

st.divider()

# =========================
# CONFUSION MATRIX
# =========================
st.subheader("🧩 Confusion Matrix")

rf_cm = confusion_matrix(y_test, rf_model.predict(X_test))
knn_cm = confusion_matrix(y_test, knn_model.predict(X_test))

col1, col2 = st.columns(2)

with col1:
    st.write("### Random Forest")
    fig, ax = plt.subplots()
    sns.heatmap(
        rf_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Tidak Lulus", "Lulus"],
        yticklabels=["Tidak Lulus", "Lulus"]
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig)

with col2:
    st.write("### KNN")
    fig, ax = plt.subplots()
    sns.heatmap(
        knn_cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Tidak Lulus", "Lulus"],
        yticklabels=["Tidak Lulus", "Lulus"]
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig)

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

    st.subheader("📊 Hasil Evaluasi Kelulusan")

    st.write(f"👤 **Nama**: {nama}")
    st.write(f"🆔 **NIM**: {nim}")
    st.write(f"📌 **Nilai Akhir**: {nilai_akhir:.2f}")

    if nilai_akhir >= 70:
        st.success("✅ MAHASISWA DINYATAKAN LULUS 🎓")
    else:
        st.error("❌ MAHASISWA DINYATAKAN TIDAK LULUS")

    # =========================
    # ANALISIS ML (PENDUKUNG)
    # =========================
    input_ml = [[absences, uts, uas]]

    rf_prob = rf_model.predict_proba(input_ml)[0][1]
    knn_prob = knn_model.predict_proba(input_ml)[0][1]

    st.subheader("🤖 Analisis Machine Learning (Pendukung)")

    st.dataframe(pd.DataFrame({
        "Algoritma": ["Random Forest", "KNN"],
        "Probabilitas Lulus (%)": [
            round(rf_prob * 100, 2),
            round(knn_prob * 100, 2)
        ]
    }))

    st.info(
        "📌 Keputusan kelulusan ditentukan oleh nilai akhir (≥ 70). "
        "Model Machine Learning digunakan sebagai pembanding dan analisis pendukung."
    )
