import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa (ML)",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.caption("Machine Learning sebagai Penentu Utama")

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(
    "student-mat.csv",
    sep=",",
    quotechar='"'
)

# Bersihkan nama kolom
df.columns = df.columns.str.strip().str.lower()

# Debug
st.write("Kolom yang terbaca:", df.columns.tolist())



# =========================
# VALIDASI KOLOM
# =========================
required_cols = ["g1", "g2", "g3", "absences"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Kolom {col} tidak ditemukan")
        st.stop()

# =========================
# PREPARASI DATA
# =========================
df["uts"] = df["g1"] * 5
df["uas"] = df["g2"] * 5
df["lulus"] = (df["g3"] >= 10).astype(int)

X = df[["absences", "uts", "uas"]]
y = df["lulus"]

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    random_state=42
)

knn = KNeighborsClassifier(
    n_neighbors=9,
    weights="distance"
)

rf.fit(X_train, y_train)
knn.fit(X_train, y_train)

# =========================
# AKURASI
# =========================
st.subheader("📈 Akurasi Model")
st.write("Random Forest:", accuracy_score(y_test, rf.predict(X_test)))
st.write("KNN:", accuracy_score(y_test, knn.predict(X_test)))

# =========================
# CONFUSION MATRIX
# =========================
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

st.subheader("📊 Confusion Matrix")
plot_cm(confusion_matrix(y_test, rf.predict(X_test)), "Random Forest")
plot_cm(confusion_matrix(y_test, knn.predict(X_test)), "KNN")

# =========================
# INPUT MAHASISWA
# =========================
st.subheader("📝 Input Data Mahasiswa")

uts = st.number_input("Nilai UTS (0–100)", 0, 100, 60)
uas = st.number_input("Nilai UAS (0–100)", 0, 100, 60)
absences = st.number_input("Jumlah Ketidakhadiran (0–16)", 0, 16, 5)

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi Kelulusan"):
    data_input = np.array([[absences, uts, uas]])

    rf_prob = rf.predict_proba(data_input)[0]
    knn_prob = knn.predict_proba(data_input)[0]

    # =========================
    # BATAS ABSENSI DATASET
    # =========================
    BATAS_ABSENSI = 8

    if absences > BATAS_ABSENSI:
        rf_prob[1] = min(rf_prob[1], 0.70)
        knn_prob[1] = min(knn_prob[1], 0.70)

    st.subheader("🤖 Prediksi Kelulusan (ML Penentu Utama)")

    st.write("### 🌲 Random Forest")
    st.write(f"Lulus: {rf_prob[1]*100:.2f}%")
    st.write(f"Tidak Lulus: {rf_prob[0]*100:.2f}%")

    st.write("### 📍 KNN")
    st.write(f"Lulus: {knn_prob[1]*100:.2f}%")
    st.write(f"Tidak Lulus: {knn_prob[0]*100:.2f}%")

    if rf_prob[1] >= 0.5:
        st.success("✅ MAHASISWA DIPREDIKSI LULUS")
    else:
        st.error("❌ MAHASISWA DIPREDIKSI TIDAK LULUS")
