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

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("student-mat.csv", sep=";")

# =========================
# PREPROCESSING
# =========================
df["pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

X = df[["studytime", "failures", "absences", "G1", "G2"]]
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

# =========================
# HEADER
# =========================
st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.caption("Model: Decision Tree | Dataset: Student Performance (Kaggle)")

st.metric("📈 Akurasi Model", f"{accuracy*100:.2f}%")
st.divider()

# =========================
# FORM INPUT MAHASISWA
# =========================
st.subheader("📝 Form Input Data Mahasiswa")

with st.form("form_mahasiswa"):
    nama = st.text_input("Nama Mahasiswa")
    nim = st.text_input("NIM")

    studytime = st.selectbox(
        "Waktu Belajar per Hari",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "≤ 2 jam",
            2: "2 – 5 jam",
            3: "5 – 10 jam",
            4: "> 10 jam"
        }[x]
    )

    failures = st.number_input(
        "Jumlah Mata Kuliah Gagal / Remedial",
        min_value=0, max_value=10, value=0
    )

    absences = st.number_input(
        "Jumlah Ketidakhadiran",
        min_value=0, max_value=50, value=3
    )

    g1 = st.number_input(
        "Nilai Semester Sebelumnya (G1)",
        min_value=0, max_value=20, value=10
    )

    g2 = st.number_input(
        "Nilai Semester Terakhir (G2)",
        min_value=0, max_value=20, value=10
    )

    submit = st.form_submit_button("🔍 Prediksi Kelulusan")

# =========================
# HASIL PREDIKSI
# =========================
if submit:
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

    prob_df = pd.DataFrame({
        "Status": ["Tidak Lulus", "Lulus"],
        "Probabilitas (%)": [
            round(probability[0][0] * 100, 2),
            round(probability[0][1] * 100, 2)
        ]
    })

    st.dataframe(prob_df, use_container_width=True)

    st.info("📌 Prediksi ini bersifat simulasi untuk keperluan akademik.")
