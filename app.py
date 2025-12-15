import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.caption("Nilai input disesuaikan dengan skala perkuliahan (0–100)")

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("student-mat.csv")
df.columns = df.columns.str.strip()

# =========================
# PREPROCESSING
# =========================
df["pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

features = ["studytime", "failures", "absences", "G1", "G2"]
X = df[features]
y = df["pass"]

# =========================
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
st.metric("📈 Akurasi Model", f"{accuracy*100:.2f}%")
st.divider()

# =========================
# FUNGSI KONVERSI
# =========================
def convert_studytime(jam):
    if jam <= 2:
        return 1
    elif jam <= 5:
        return 2
    elif jam <= 10:
        return 3
    else:
        return 4

def convert_score(nilai):
    return round((nilai / 100) * 20, 2)

# =========================
# FORM INPUT MAHASISWA
# =========================
st.subheader("📝 Input Data Mahasiswa")

with st.form("form_mahasiswa"):
    nama = st.text_input("Nama Mahasiswa")
    nim = st.text_input("NIM")

    jam_belajar = st.number_input(
        "Jam Belajar per Hari",
        min_value=0.0, max_value=24.0, value=3.0
    )

    failures = st.number_input(
        "Jumlah Mata Kuliah Gagal",
        min_value=0, max_value=10, value=0
    )

    absences = st.number_input(
        "Jumlah Ketidakhadiran",
        min_value=0, max_value=50, value=3
    )

    nilai_g1_100 = st.number_input(
        "Nilai Semester Sebelumnya (0–100)",
        min_value=0, max_value=100, value=75
    )

    nilai_g2_100 = st.number_input(
        "Nilai Semester Terakhir (0–100)",
        min_value=0, max_value=100, value=80
    )

    submit = st.form_submit_button("🔍 Prediksi Kelulusan")

# =========================
# HASIL PREDIKSI
# =========================
if submit:
    studytime = convert_studytime(jam_belajar)
    g1 = convert_score(nilai_g1_100)
    g2 = convert_score(nilai_g2_100)

    input_data = [[studytime, failures, absences, g1, g2]]

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.divider()
    st.subheader("🧠 Hasil Prediksi")

    st.write(f"👤 **Nama**: {nama}")
    st.write(f"🆔 **NIM**: {nim}")

    if prediction[0] == 1:
        st.success("✅ **MAHASISWA DIPREDIKSI LULUS** 🎓")
    else:
        st.error("❌ **MAHASISWA DIPREDIKSI TIDAK LULUS**")

    st.subheader("📊 Probabilitas Prediksi")

    st.dataframe(pd.DataFrame({
        "Status": ["Tidak Lulus", "Lulus"],
        "Probabilitas (%)": [
            round(probability[0][0] * 100, 2),
            round(probability[0][1] * 100, 2)
        ]
    }))

    st.info(
        f"📌 Konversi nilai: "
        f"G1={g1}/20, G2={g2}/20, Studytime={studytime}"
    )
