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

# =========================
# LOAD DATASET (ANTI G1 ERROR)
# =========================
try:
    df = pd.read_csv("student-mat.csv")
except:
    st.error("❌ File student-mat.csv tidak ditemukan")
    st.stop()

# Jika terbaca 1 kolom → perbaiki delimiter
if df.shape[1] == 1:
    df = pd.read_csv("student-mat.csv", sep=",")

if df.shape[1] == 1:
    df = pd.read_csv("student-mat.csv", sep=";")

# Normalisasi nama kolom
df.columns = (
    df.columns
    .str.strip()
    .str.replace('"', '')
    .str.lower()
)

st.caption("Kolom dataset yang terbaca:")
st.write(df.columns.tolist())

# =========================
# VALIDASI KOLOM WAJIB
# =========================
required_cols = ["g1", "g2", "g3", "absences"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"❌ Kolom berikut tidak ditemukan: {missing}")
    st.stop()

# =========================
# PREPARASI DATA
# =========================
df["uts"] = df["g1"] * 5   # 0–20 → 0–100
df["uas"] = df["g2"] * 5

# Target ML
df["lulus"] = (df["g3"] >= 10).astype(int)

X = df[["absences", "uts", "uas"]]
y = df["lulus"]

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MODEL ML
# =========================
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    min_samples_leaf=5,
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

plot_cm(confusion_matrix(y_test, rf.predict(X_test)), "Random Forest")
plot_cm(confusion_matrix(y_test, knn.predict(X_test)), "KNN")

st.divider()

# =========================
# INPUT MAHASISWA
# =========================
st.subheader("📝 Input Data Mahasiswa")

with st.form("form_mahasiswa"):
    uts = st.number_input("Nilai UTS (0–100)", 0, 100, 75)
    uas = st.number_input("Nilai UAS (0–100)", 0, 100, 80)
    tugas = st.number_input("Nilai Tugas (0–100)", 0, 100, 85)
    absences = st.number_input("Jumlah Ketidakhadiran (0–16)", 0, 16, 2)
    submit = st.form_submit_button("🔍 Prediksi Kelulusan")

# =========================
# PREDIKSI ML (PENENTU UTAMA)
# =========================
if submit:
    input_data = [[absences, uts, uas]]

    rf_prob = rf.predict_proba(input_data)[0]
    knn_prob = knn.predict_proba(input_data)[0]

    st.subheader("🤖 Prediksi Kelulusan (ML sebagai PENENTU UTAMA)")

    st.write("### 🌲 Random Forest")
    st.write(f"Lulus: **{rf_prob[1]*100:.2f}%**")
    st.write(f"Tidak Lulus: **{rf_prob[0]*100:.2f}%**")

    st.write("### 📍 KNN")
    st.write(f"Lulus: **{knn_prob[1]*100:.2f}%**")
    st.write(f"Tidak Lulus: **{knn_prob[0]*100:.2f}%**")

    if rf_prob[1] >= 0.5:
        st.success("✅ **PREDIKSI UTAMA: MAHASISWA LULUS**")
    else:
        st.error("❌ **PREDIKSI UTAMA: MAHASISWA TIDAK LULUS**")
