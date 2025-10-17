import streamlit as st
import pandas as pd
import joblib
import time

# ===============================
# 🔧 Konfigurasi halaman
# ===============================
st.set_page_config(
    page_title="Klasifikasi Tomat",
    page_icon="🍅",
    layout="centered"
)

# ===============================
# 📦 Load model & scaler
# ===============================
model = joblib.load("model_klasifikasi_tomat.joblib")
scaler = joblib.load("scaler_klasifikasi_tomat.joblib")

# ===============================
# 🧠 Judul dan deskripsi
# ===============================
st.markdown(
    "<h1 style='text-align:center;'>🍅 Klasifikasi Tomat 🍅</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Aplikasi <i>machine learning</i> untuk "
    "mengklasifikasikan tomat ke dalam kategori <b>Ekspor</b>, "
    "<b>Lokal Premium</b>, atau <b>Industri</b> berdasarkan fitur kualitas "
    "seperti ukuran, warna, dan tekstur.</p>",
    unsafe_allow_html=True
)

# ===============================
# 🎚️ Input fitur
# ===============================
berat = st.slider("Berat Tomat (gram)", 50, 200, 80)
kekenyalan = st.slider("Tingkat Kekenyalan", 2.0, 10.0, 4.2)
kadar_gula = st.slider("Kadar Gula", 1.0, 10.0, 5.3)
tebal_kulit = st.slider("Tebal Kulit (mm)", 0.1, 1.0, 0.7)

# ===============================
# 🔍 Tombol prediksi
# ===============================
if st.button("🍅 Prediksi Kategori", type="primary", use_container_width=True):
    data_baru = pd.DataFrame(
        [[berat, kekenyalan, kadar_gula, tebal_kulit]],
        columns=["berat", "kekenyalan", "kadar_gula", "tebal_kulit"]
    )
    
    data_baru_scaled = scaler.transform(data_baru)
    prediksi = model.predict(data_baru_scaled)[0]
    presentase = max(model.predict_proba(data_baru_scaled)[0])

    # Efek animasi loading
    with st.spinner("🔍 Menganalisis kualitas tomat..."):
        time.sleep(1.5)

    # Hasil prediksi lebih menarik
    st.success(f"🍅 **Kategori:** {prediksi}")
    st.info(f"Tingkat keyakinan model: **{presentase*100:.2f}%**")
    st.balloons()

st.divider()
st.caption("Dibuat dengan ❤️ dan 🍅 oleh **Raditya Fauzi Pratama**")
