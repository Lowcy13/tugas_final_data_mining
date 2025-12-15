import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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
# LOAD DATASET
# =========================
try:
    df = pd.read_csv("student-mat.csv")
except:
    st.error("❌ File student-mat.csv tidak ditemukan")
    st.stop()

# Perbaiki delimiter
if df.shape[1] == 1:
    df = pd.read_csv("student-mat.csv", sep=";")

# Normalisasi kolom
df.columns = df.columns.str.strip().str.lower()

# =========================
# VALIDASI KOLOM
# =========================
required_cols = ["g1", "g2", "g3", "absences"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"❌ Kolom '{col}' tidak ditemukan")
        st.stop()

# =========================
# PREPARASI DATA
# =========================
df["uts"] = df["g1"] * 5     # 0–20 → 0–100
df["uas"] = df["g2"] * 5
df["lulus"] = df["g3"].apply(lambda x: 1 if x >= 10 else 0)

X = df[["absences", "uts", "uas"]]
y = df["lulus"]

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

knn = KNeighborsClassifier(
    n_neighbors=7,
    weights="distance"
)

rf.fit(X_train, y_train)
knn.fit(X_train, y_train)

# =========================
# EVALUASI MODEL
# =========================
acc_rf = accuracy_score(y_test, rf.predict(X_test))
acc_knn = accuracy_score(y_test, knn.predict(X_test))

st.subheader("📈 Perbandingan Akurasi Model")
st.bar_chart({
    "Random Forest": acc_rf,
    "KNN": acc_knn
})

# =========================
# CONFUSION MATRIX
# =========================
st.subheader("📊 Confusion Matrix")

def plot_cm(cm, title):
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Tidak Lulus", "Lulus"])
    ax.set_yticklabels(["Tidak Lulus", "Lulus"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)

cm_rf = confusion_matrix(y_test, rf.predict(X_test))
cm_knn = confusion_matrix(y_test, knn.predict(X_test))

plot_cm(cm_rf, "Random Forest")
plot_cm(cm_knn, "KNN")

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
# PROSES NILAI
# =========================
if submit:
    TOTAL_PERTEMUAN = 16

    nilai_kehadiran = ((TOTAL_PERTEMUAN - absences) / TOTAL_PERTEMUAN) * 100
    nilai_kehadiran = max(0, min(nilai_kehadiran, 100))

    nilai_akhir = (
        0.45 * uas +
        0.30 * uts +
        0.15 * tugas +
        0.10 * nilai_kehadiran
    )

    st.divider()
    st.subheader("📊 Hasil Evaluasi")

    st.write(f"👤 Nama: **{nama}**")
    st.write(f"🆔 NIM: **{nim}**")
    st.write(f"📌 Nilai Kehadiran: **{nilai_kehadiran:.2f}**")
    st.write(f"📌 Nilai Akhir: **{nilai_akhir:.2f}**")

    if nilai_akhir >= 70:
        st.success("✅ **MAHASISWA DINYATAKAN LULUS**")
    else:
        st.error("❌ **MAHASISWA DINYATAKAN TIDAK LULUS**")

    # =========================
    # PROBABILITAS ML
    # =========================
    rf_prob = rf.predict_proba([[absences, uts, uas]])
    knn_prob = knn.predict_proba([[absences, uts, uas]])

    st.subheader("🤖 Probabilitas Kelulusan (Machine Learning)")

    st.write("### 🌲 Random Forest")
    st.dataframe(pd.DataFrame({
        "Status": ["Tidak Lulus", "Lulus"],
        "Probabilitas (%)": [
            round(rf_prob[0][0] * 100, 2),
            round(rf_prob[0][1] * 100, 2)
        ]
    }))

    st.write("### 📍 KNN")
    st.dataframe(pd.DataFrame({
        "Status": ["Tidak Lulus", "Lulus"],
        "Probabilitas (%)": [
            round(knn_prob[0][0] * 100, 2),
            round(knn_prob[0][1] * 100, 2)
        ]
    }))

    st.info(
        "📌 Keputusan kelulusan ditentukan oleh nilai akhir (≥ 70). "
        "Model Machine Learning digunakan sebagai analisis pendukung."
    )
