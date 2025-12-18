import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from pandas.tseries.offsets import BDay

# ======================
# CONFIG
# ======================
st.set_page_config(
    page_title="Prediksi Harga Saham BBCA (LSTM)",
    layout="wide"
)

st.title("📈 Prediksi Harga Saham BBCA Menggunakan LSTM")

# ======================
# LOAD MODEL & SCALER
# ======================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_lstm_bbca.h5")
    scaler = joblib.load("scaler_bbca.save")
    return model, scaler

model, scaler = load_model()

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("bbca.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")
    return df

data = load_data()
close_price = data[['Close']].values
close_scaled = scaler.transform(close_price)

# ======================
# SIDEBAR
# ======================
st.sidebar.header("⚙️ Pengaturan Model")

timestep = st.sidebar.slider(
    "Window Size (timestep)",
    min_value=10,
    max_value=100,
    value=30
)

pred_days = st.sidebar.number_input(
    "Jumlah Hari Prediksi",
    min_value=1,
    max_value=30,
    value=7
)

# ======================
# CREATE DATASET
# ======================
def create_sequence(data, timestep):
    X = []
    for i in range(len(data) - timestep):
        X.append(data[i:i+timestep])
    return np.array(X)

X = create_sequence(close_scaled, timestep)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ======================
# PREDIKSI MULTI-DAY
# ======================
last_data = close_scaled[-timestep:].reshape(1, timestep, 1)

predictions = []
dates = []

last_date = data['Date'].iloc[-1]

for i in range(pred_days):
    pred_scaled = model.predict(last_data, verbose=0)
    predictions.append(pred_scaled[0][0])

    last_data = np.append(
        last_data[:, 1:, :],
        pred_scaled.reshape(1,1,1),
        axis=1
    )

    dates.append(last_date + BDay(i+1))

predictions = scaler.inverse_transform(
    np.array(predictions).reshape(-1,1)
).flatten()

pred_df = pd.DataFrame({
    "Date": dates,
    "Predicted Close": predictions
})

# ======================
# DISPLAY PREDICTION
# ======================
st.subheader("📊 Hasil Prediksi Harga Saham")

st.dataframe(pred_df, use_container_width=True)

# ======================
# VISUALIZATION
# ======================
st.subheader("📈 Grafik Prediksi")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(data['Date'], data['Close'], label="Harga Aktual")
ax.plot(pred_df['Date'], pred_df['Predicted Close'], label="Prediksi", linestyle="--")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga")
ax.legend()
ax.grid(True)

st.pyplot(fig)
