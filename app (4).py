import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# ============================================
# KONFIGURASI APLIKASI
# ============================================
st.set_page_config(
    page_title="Prediksi Saham BBCA - LSTM",
    layout="wide",
    page_icon="📈"
)

MODEL_PATH = "model_bbca.keras"
SCALER_PATH = "scaler_bbca.pkl"
DATA_PATH = "bbca.csv"
TIMESTEP = 60

# ============================================
# LOAD MODEL & SCALER
# ============================================
@st.cache_resource
def load_lstm_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

model = load_lstm_model()
scaler: MinMaxScaler = load_scaler()

# ============================================
# FUNGSI PREDIKSI MASA DEPAN
# ============================================
def predict_future(df, n_future):
    close_prices = df[['Close']].astype(float).values
    scaled_close = scaler.transform(close_prices)

    last_sequence = scaled_close[-TIMESTEP:]
    future_predictions = []

    for _ in range(n_future):
        input_seq = last_sequence.reshape(1, TIMESTEP, 1)
        pred_scaled = model.predict(input_seq, verbose=0)

        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        future_predictions.append(pred_price)

        last_sequence = np.append(last_sequence[1:], pred_scaled, axis=0)

    return future_predictions

# ============================================
# SIDEBAR NAVIGASI
# ============================================
menu = st.sidebar.radio(
    "Navigasi",
    ["Home", "Prediksi Harga"]
)

# ============================================
# HALAMAN HOME
# ============================================
if menu == "Home":
    st.title("Prediksi Harga Saham BBCA (LSTM)")
    st.write("""
Aplikasi ini menggunakan **Long Short-Term Memory (LSTM)**  
dengan **Dropout, L2 Regularization, dan Batch Normalization**  
untuk memprediksi harga saham **BBCA**.

➡️ Pilih **Prediksi Harga** untuk memulai.
    """)

# ============================================
# HALAMAN PREDIKSI
# ============================================
elif menu == "Prediksi Harga":

    st.title("Prediksi Harga Saham BBCA")

    df = pd.read_csv(DATA_PATH)

    if "Close" not in df.columns:
        st.error("Dataset harus memiliki kolom 'Close'")
        st.stop()

    st.sidebar.header("Pengaturan Prediksi")
    n_days = st.sidebar.slider(
        "Prediksi berapa hari ke depan?",
        min_value=1,
        max_value=60,
        value=7
    )

    if st.button("Jalankan Prediksi"):
        preds = predict_future(df, n_days)
        st.success("Prediksi berhasil dibuat!")

        # =============================
        # GRAFIK INTERAKTIF
        # =============================
        st.subheader("Grafik Harga Aktual vs Prediksi")

        actual = df["Close"].values
        future_index = np.arange(len(actual), len(actual) + n_days)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=actual,
            mode='lines',
            name='Harga Aktual',
            line=dict(width=3)
        ))

        fig.add_trace(go.Scatter(
            x=future_index,
            y=preds,
            mode='lines+markers',
            name='Prediksi LSTM',
            line=dict(width=3, dash='dash')
        ))

        fig.update_layout(
            title="Prediksi Harga Saham BBCA (LSTM)",
            xaxis_title="Waktu",
            yaxis_title="Harga (IDR)",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # =============================
        # TABEL PREDIKSI
        # =============================
        st.subheader("Tabel Prediksi Harga")

        pred_df = pd.DataFrame({
            "Hari Ke": np.arange(1, n_days + 1),
            "Prediksi Harga (IDR)": preds
        })

        st.dataframe(pred_df, use_container_width=True)
