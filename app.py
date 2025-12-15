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

# =========================
# LOAD DATASET (SUPER AMAN)
# =========================
try:
    df = pd.read_csv("student-mat.csv", sep=";", encoding="utf-8")
except:
    df = pd.read_csv("student-mat.csv", sep=";", encoding="latin1")

# Bersihkan nama kolom
df.columns = df.columns.str.strip()

# =========================
# CEK KOLOM G3
# =========================
if "G3" not in df.columns:
    st.error("❌ Kolom **G3** tidak ditemukan di dataset.")
    st.write("Kolom yang tersedia:")
    st.write(df.columns.tolist())
    st.stop()

# =========================
# PREPROCESSING
# =========================
df["pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

features = ["studytime", "failures", "absences", "G1", "G2"]

# Cek fitur
for col in features:
    if col not in df.columns:
        st.error(f"❌ Kolom **{col}** tidak ditemukan.")
        st.stop()

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
# FORM INPUT MAHASISWA
# =========================
st.subheader("📝 Input Data Mahasiswa")

with st.form("form_mahasiswa"):
    nama = st.text_input("Nama Mahasiswa")
    nim = st.text_input("NIM")

    studytime = st.selectbox(
        "Waktu Belajar per Hari",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "≤ 2 jam",
            2: "2–5 jam",
            3: "5–10 jam",
            4: "> 10 jam"
        }[x]
    )

    failures = st.number_input("Jumlah Mata Kuliah Gagal", 0, 10, 0)
    absences = st.number_input("Jumlah Ketidakhadiran", 0, 50, 3)
    g1 = st.number_input("Nilai G1", 0, 20, 10)
    g2 = st.number_input("Nilai G2", 0, 20, 10)

    submit = st.form_submit_button("🔍 Prediksi")

# =========================
# HASIL PREDIKSI
# =========================
if submit:
    input_data = [[studytime, failures, absences, g1, g2]]
    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)

    st.divider()
    st.subheader("🧠 Hasil Prediksi")

    st.write(f"👤 **Nama**: {nama}")
    st.write(f"🆔 **NIM**: {nim}")

    if pred[0] == 1:
        st.success("✅ **MAHASISWA DIPREDIKSI LULUS** 🎓")
    else:
        st.error("❌ **MAHASISWA DIPREDIKSI TIDAK LULUS**")

    st.subheader("📊 Probabilitas")
    st.dataframe(pd.DataFrame({
        "Status": ["Tidak Lulus", "Lulus"],
        "Probabilitas (%)": [
            round(prob[0][0]*100, 2),
            round(prob[0][1]*100, 2)
        ]
    }))
