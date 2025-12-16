import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Prediksi Kelulusan Mata Kuliah Matematika")
st.caption("Machine Learning sebagai penentu utama kelulusan")

# =========================
# LOAD DATASET
# =========================
try:
    df = pd.read_csv("student-mat.csv", sep=";")
except:
    st.error("❌ File student-mat.csv tidak ditemukan")
    st.stop()

df.columns = df.columns.str.lower().str.strip()

# =========================
# VALIDASI KOLOM
# =========================
required_cols = ["g1", "g2", "g3", "absences"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"❌ Kolom {col} tidak ditemukan")
        st.stop()

# =========================
# PREPARASI DATA
# =========================
TOTAL_PERTEMUAN = 16

df["uts"] = df["g1"] * 5
df["uas"] = df["g2"] * 5
df["kehadiran"] = ((TOTAL_PERTEMUAN - df["absences"]) / TOTAL_PERTEMUAN) * 100
df["kehadiran"] = df["kehadiran"].clip(0, 100)

# Nilai tugas disimulasikan agar realistis
df["tugas"] = (df["uts"] * 0.4 + df["uas"] * 0.6)

# Label kelulusan (ground truth)
df["lulus"] = df["g3"].apply(lambda x: 1 if x >= 10 else 0)

X = df[["uts", "uas", "tugas", "kehadiran"]]
y = df["lulus"]

# Normalisasi
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# TRAIN MODEL
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

plot_cm(confusion_matrix(y_test, rf.predict(X_test)), "Random Forest")
plot_cm(confusion_matrix(y_test, knn.predict(X_test)), "KNN")

st.divider()

# =========================
# FORM INPUT MAHASISWA
# =========================
st.subheader("📝 Input Data Mahasiswa")

with st.form("form_mahasiswa"):
    nama = st.text_input("Nama Mahasiswa")
    nim = st.text_input("NIM")

    uts = st.number_input("Nilai UTS (0–100)", 0, 100, 70)
    uas = st.number_input("Nilai UAS (0–100)", 0, 100, 75)
    tugas = st.number_input("Nilai Tugas (0–100)", 0, 100, 80)
    absences = st.number_input("Jumlah Ketidakhadiran (0–16)", 0, 16, 2)

    submit = st.form_submit_button("🔍 Prediksi Kelulusan")

# =========================
# PREDIKSI MAHASISWA BARU
# =========================
if submit:
    kehadiran = ((TOTAL_PERTEMUAN - absences) / TOTAL_PERTEMUAN) * 100
    kehadiran = max(0, min(kehadiran, 100))

    input_data = pd.DataFrame([[
        uts, uas, tugas, kehadiran
    ]], columns=["uts", "uas", "tugas", "kehadiran"])

    input_scaled = scaler.transform(input_data)

    pred = rf.predict(input_scaled)[0]
    prob = rf.predict_proba(input_scaled)[0]

    st.divider()
    st.subheader("📊 Hasil Prediksi Machine Learning")

    st.write(f"👤 Nama: **{nama}**")
    st.write(f"🆔 NIM: **{nim}**")

    if pred == 1:
        st.success("✅ **MAHASISWA DINYATAKAN LULUS (ML)**")
    else:
        st.error("❌ **MAHASISWA DINYATAKAN TIDAK LULUS (ML)**")

    st.subheader("🤖 Probabilitas Prediksi (Random Forest)")
    st.dataframe(pd.DataFrame({
        "Status": ["Tidak Lulus", "Lulus"],
        "Probabilitas (%)": [
            round(prob[0] * 100, 2),
            round(prob[1] * 100, 2)
        ]
    }))

    st.info(
        "📌 Keputusan kelulusan ditentukan sepenuhnya oleh "
        "Machine Learning (Random Forest) berdasarkan pola data historis."
    )
