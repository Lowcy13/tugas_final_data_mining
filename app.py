import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Kelulusan Mata Kuliah Matematika",
    page_icon="📘",
    layout="centered"
)

st.title("📘 Prediksi Kelulusan Mata Kuliah Matematika")
st.caption("Berdasarkan UTS, UAS, Kehadiran, dan Tugas")

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("student-mat.csv")
df.columns = df.columns.str.strip()

# =========================
# PREPROCESSING
# =========================
df["pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

# Fitur utama (dataset asli)
features = ["absences", "G1", "G2"]
X = df[features]
y = df["pass"]

# =========================
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
st.metric("📈 Akurasi Model", f"{accuracy*100:.2f}%")
st.divider()

# =========================
# FUNGSI KONVERSI
# =========================
def convert_score(nilai):
    """0–100 → 0–20"""
    return round((nilai / 100) * 20, 2)

def tugas_to_bonus(jumlah_tugas):
    """
    Jumlah tugas (0–10) → bonus nilai (0–2)
    """
    return round((jumlah_tugas / 10) * 2, 2)

# =========================
# FORM INPUT MAHASISWA
# =========================
st.subheader("📝 Input Data Mahasiswa")

with st.form("form_mahasiswa"):
    nama = st.text_input("Nama Mahasiswa")
    nim = st.text_input("NIM")

    absences = st.number_input(
        "Jumlah Ketidakhadiran (Sangat Berpengaruh)",
        min_value=0, max_value=50, value=3
    )

    tugas = st.number_input(
        "Jumlah Tugas yang Dikumpulkan",
        min_value=0, max_value=10, value=8,
        help="Semakin banyak tugas dikumpulkan, peluang lulus meningkat"
    )

    nilai_uts = st.number_input(
        "Nilai Ujian Tengah Semester (UTS) – 0–100",
        min_value=0, max_value=100, value=75
    )

    nilai_uas = st.number_input(
        "Nilai Ujian Akhir Semester (UAS) – 0–100",
        min_value=0, max_value=100, value=80
    )

    submit = st.form_submit_button("🔍 Prediksi Kelulusan")

# =========================
# HASIL PREDIKSI
# =========================
if submit:
    g1 = convert_score(nilai_uts)
    g2 = convert_score(nilai_uas)

    # Bonus tugas ditambahkan ke UAS
    bonus = tugas_to_bonus(tugas)
    g2_adjusted = min(g2 + bonus, 20)

    input_data = [[absences, g1, g2_adjusted]]

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.divider()
    st.subheader("🧠 Hasil Prediksi")

    st.write(f"👤 **Nama**: {nama}")
    st.write(f"🆔 **NIM**: {nim}")

    if prediction[0] == 1:
        st.success("✅ **MAHASISWA DIPREDIKSI LULUS MATA KULIAH MATEMATIKA** 🎓")
    else:
        st.error("❌ **MAHASISWA DIPREDIKSI TIDAK LULUS MATA KULIAH MATEMATIKA**")

    st.subheader("📊 Probabilitas Prediksi")

    st.dataframe(pd.DataFrame({
        "Status": ["Tidak Lulus", "Lulus"],
        "Probabilitas (%)": [
            round(probability[0][0] * 100, 2),
            round(probability[0][1] * 100, 2)
        ]
    }))

    st.info(
        f"📌 Detail perhitungan:\n"
        f"- UTS: {g1}/20\n"
        f"- UAS: {g2}/20\n"
        f"- Bonus tugas: +{bonus}\n"
        f"- UAS setelah tugas: {g2_adjusted}/20\n"
        f"- Kehadiran berpengaruh signifikan"
    )
