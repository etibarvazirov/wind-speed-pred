import streamlit as st
import torch
import joblib
import requests
import numpy as np
import pandas as pd
from model import NHiTS

# ===================================================
# CONFIG
# ===================================================
SEQ_LEN = 72
NUM_FEATURES = 16
DEVICE = "cpu"

FEATURES = [
    "wind_speed",
    "temperature",
    "wind_direction",
    "wind_dir_sin", "wind_dir_cos",
    "lag1", "lag3", "lag6", "lag12", "lag24",
    "roll6_mean", "roll12_mean", "roll24_mean",
    "roll6_std", "roll12_std", "roll24_std"
]

# ===================================================
# LOAD SCALER & MODEL
# ===================================================
@st.cache_resource

def load_model():
    # Load scaler (numpy version — SAFE)
    scaler_mean = np.load("scaler_mean.npy")
    scaler_scale = np.load("scaler_scale.npy")

    class SimpleScaler:
        def __init__(self, mean, scale):
            self.mean_ = mean
            self.scale_ = scale
        
        def transform(self, X):
            return (X - self.mean_) / self.scale_

    scaler = SimpleScaler(scaler_mean, scaler_scale)

    # Load model
    # model = NHiTS(seq_len=SEQ_LEN, num_features=len(FEATURES))
    model = NHiTS(seq_len=168, num_features=15)
    model.load_state_dict(torch.load("n_hits_wind_model.pth", map_location="cpu"))
    model.eval()

    return model, scaler


model, scaler = load_model()

# ===================================================
# GET REALTIME ERA5 DATA
# ===================================================
def get_realtime_era5():
    lat, lon = 40.4093, 49.8671  # Baku
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=windspeed_10m,temperature_2m,winddirection_10m"
        "&forecast_days=3"
    )
    r = requests.get(url).json()

    df = pd.DataFrame({
        "wind_speed": r["hourly"]["windspeed_10m"][:72],
        "temperature": r["hourly"]["temperature_2m"][:72],
        "wind_direction": r["hourly"]["winddirection_10m"][:72],
    })
    return df

# ===================================================
# PREPROCESS
# ===================================================
def preprocess(df):
    df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_direction"]))
    df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_direction"]))

    # Create lags
    df["lag1"] = df["wind_speed"].shift(1)
    df["lag3"] = df["wind_speed"].shift(3)
    df["lag6"] = df["wind_speed"].shift(6)
    df["lag12"] = df["wind_speed"].shift(12)
    df["lag24"] = df["wind_speed"].shift(24)

    # Rolling windows
    df["roll6_mean"] = df["wind_speed"].rolling(6).mean()
    df["roll12_mean"] = df["wind_speed"].rolling(12).mean()
    df["roll24_mean"] = df["wind_speed"].rolling(24).mean()

    df["roll6_std"] = df["wind_speed"].rolling(6).std()
    df["roll12_std"] = df["wind_speed"].rolling(12).std()
    df["roll24_std"] = df["wind_speed"].rolling(24).std()

    df = df.dropna().reset_index(drop=True)

    data = df[FEATURES]
    data_scaled = scaler.transform(data)

    return data_scaled.reshape(1, SEQ_LEN, len(FEATURES))

# ===================================================
# PREDICT
# ===================================================
def predict_next_hour():
    df = get_realtime_era5()
    df_scaled = preprocess(df)
    X = torch.tensor(df_scaled).float()

    with torch.no_grad():
        pred = model(X).numpy().squeeze()

    return pred, df

# ===================================================
# STREAMLIT UI
# ===================================================
st.title("🌬️ Azerbaijan Wind Speed Forecasting (N-HiTS Model)")
st.markdown("Real-time wind speed prediction for the next hour using advanced Machine Learning (N-HiTS).")

if st.button("🔮 Predict Next Hour Wind Speed"):
    pred, df = predict_next_hour()
    st.success(f"🌬️ **Next hour predicted wind speed: {pred:.2f} m/s**")

    st.line_chart(df["wind_speed"], use_container_width=True)
    st.caption("Last 72 hours real wind speed from ERA5 API")

st.info("Model: N-HiTS • Features: 16 • Sequence Length: 72 • Data: ERA5 Hourly")



