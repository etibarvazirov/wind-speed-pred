import streamlit as st
import torch
import numpy as np
import pandas as pd
import requests
from model import NHiTS

# ====================================================
# SƏTİRLƏR — KONFİQURASİYA
# ====================================================
SEQ_LEN = 168
NUM_FEATURES = 15
DEVICE = "cpu"

FEATURES = [
    "temperature", "wind_speed", "wind_dir_sin", "wind_dir_cos",
    "lag1", "lag3", "lag6", "lag12", "lag24",
    "roll6_mean", "roll12_mean", "roll24_mean",
    "roll6_std", "roll12_std", "roll24_std"
]

# ====================================================
# MODEL & SCALER YÜKLƏNMƏSİ
# ====================================================
@st.cache_resource
def load_model():
    scaler_mean = np.load("scaler_mean.npy")
    scaler_scale = np.load("scaler_scale.npy")

    class SimpleScaler:
        def __init__(self, mean, scale):
            self.mean_ = mean
            self.scale_ = scale
        
        def transform(self, X):
            return (X - self.mean_) / self.scale_

    scaler = SimpleScaler(scaler_mean, scaler_scale)

    model = NHiTS(seq_len=SEQ_LEN, num_features=NUM_FEATURES)
    model.load_state_dict(torch.load("n_hits_wind_model.pth", map_location="cpu"))
    model.eval()

    return model, scaler

model, scaler = load_model()


# ====================================================
# REAL-TIME ERA5 MƏLUMATLARININ ALINMASI
# ====================================================
def get_era5(hours):
    days = int(np.ceil(hours / 24)) + 1   # rolling üçün əlavə 1 gün
    lat, lon = 40.4093, 49.8671           # Bakı

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=windspeed_10m,temperature_2m,winddirection_10m"
        f"&forecast_days={days}"
    )

    r = requests.get(url).json()

    total_needed = hours + 24  # roll24 üçün əlavə 24 saat
    df = pd.DataFrame({
        "wind_speed": r["hourly"]["windspeed_10m"][:total_needed],
        "temperature": r["hourly"]["temperature_2m"][:total_needed],
        "wind_direction": r["hourly"]["winddirection_10m"][:total_needed],
    })

    return df


# ====================================================
# PREPROCESS FUNKSIYASI
# ====================================================
def preprocess(df, hours):
    df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_direction"]))
    df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_direction"]))

    df["lag1"] = df["wind_speed"].shift(1)
    df["lag3"] = df["wind_speed"].shift(3)
    df["lag6"] = df["wind_speed"].shift(6)
    df["lag12"] = df["wind_speed"].shift(12)
    df["lag24"] = df["wind_speed"].shift(24)

    df["roll6_mean"]  = df["wind_speed"].rolling(6).mean()
    df["roll12_mean"] = df["wind_speed"].rolling(12).mean()
    df["roll24_mean"] = df["wind_speed"].rolling(24).mean()

    df["roll6_std"]  = df["wind_speed"].rolling(6).std()
    df["roll12_std"] = df["wind_speed"].rolling(12).std()
    df["roll24_std"] = df["wind_speed"].rolling(24).std()

    df = df.dropna().reset_index(drop=True)

    segment = df[FEATURES].iloc[-SEQ_LEN:]  # son 168 saat
    X = scaler.transform(segment.to_numpy())
    return X.reshape(1, SEQ_LEN, NUM_FEATURES), df


# ====================================================
# PROQNOZ FUNKSIYASI
# ====================================================
def forecast(hours):
    df = get_era5(hours)
    X, processed_df = preprocess(df, hours)

    preds = []
    inp = torch.tensor(X).float()

    for _ in range(hours):
        with torch.no_grad():
            p = model(inp).item()
        preds.append(p)

        # Yeni proqnozu inputa daxil et
        next_row = inp.numpy()[0, -1, :].copy()
        next_row[FEATURES.index("wind_speed")] = p

        new_input = np.vstack([inp.numpy()[0, 1:, :], next_row])
        inp = torch.tensor(new_input.reshape(1, SEQ_LEN, NUM_FEATURES)).float()

    return preds, processed_df


# ====================================================
# STREAMLIT İSTİFADƏÇİ İNTERFEYSİ
# ====================================================
st.title("🌬️ **Azerbaycanda Real-Time Külək Sürəti Proqnozu (N-HiTS Modeli)**")
st.markdown("""
Bu tətbiq ERA5 real-time meteoroloji məlumatları əsasında  
**N-HiTS (Neural Hierarchical Interpolation for Time-Series)** modeli ilə  
gələcək saatlar üçün külək sürətini proqnoz edir.
""")

# ---------------- Sidebar ----------------
st.sidebar.header("🔧 Parametrlər")
hours = st.sidebar.slider("Neçə saatlıq proqnoz verilsin?", 1, 24, 6)


# ---------------- Buttons ----------------
if st.button("🔮 Proqnoz et"):
    with st.spinner("Hesablanır..."):
        preds, df_processed = forecast(hours)

    st.success(f"🌬️ Növbəti **{hours} saat** üçün külək proqnozu hazırdır!")

    st.line_chart(preds, use_container_width=True)
    st.caption("Modelin proqnoz etdiyi külək sürəti (m/s)")


# ---------------- Charts Section ----------------
st.header("📊 Qrafik Analizləri")

with st.expander("📌 **Xüsusiyyətlərin Əhəmiyyəti (Integrated Gradients)**"):
    st.image("feature_importance.png", use_column_width=True)
    st.write("Bu qrafik modelin ən çox hansı dəyişənlərə həssas olduğunu göstərir.")

with st.expander("📌 **ERA5 + N-HiTS Külək Proqnozu Qrafiki**"):
    st.image("wind_forecast_plot.png", use_column_width=True)
    st.write("N-HiTS modeli küləyin real və proqnoz edilmiş dəyişimini göstərir.")


st.info("Bu sistem tədris və tədqiqat məqsədləri üçün hazırlanmışdır.")
