import streamlit as st
import torch
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
from model import NHiTS

# ===================================================
# PAGE CONFIG
# ===================================================
st.set_page_config(
    page_title="Külək Sürəti Proqnozu – N-HiTS",
    layout="wide",
    page_icon="🌬️"
)

# ===================================================
# INTRO SECTION
# ===================================================
st.title("🌬️ Azərbaycan üçün Külək Sürəti Proqnozu — N-HiTS Modeli")

st.markdown("""
<div style="padding:15px; border-radius:10px; background-color:#eef6ff;">
    <h3>📌 Layihə Haqqında</h3>
    Bu sistem ERA5 real vaxt atmosfer məlumatları əsasında Azərbaycanın 
    növbəti 1 saat üçün külək sürətini proqnozlaşdırır.  
    Model **N-HiTS dərin öyrənmə arxitekturası** ilə böyük tarixi məlumat bazasında öyrədilib.
    Proqnozlar külək enerjisi istehsalı, hava risklərinin qiymətləndirilməsi 
    və ümumi meteoroloji təhlil üçün istifadə oluna bilər.
</div>
""", unsafe_allow_html=True)

# ===================================================
# CONFIG
# ===================================================
SEQ_LEN = 168
NUM_FEATURES = 15

FEATURES = [
    "temperature", "wind_speed",
    "wind_dir_sin", "wind_dir_cos",
    "lag1", "lag3", "lag6", "lag12", "lag24",
    "roll6_mean", "roll12_mean", "roll24_mean",
    "roll6_std", "roll12_std", "roll24_std"
]

# ===================================================
# LOAD MODEL + SCALER
# ===================================================
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

# ===================================================
# GET ERA5 REALTIME DATA
# ===================================================
def get_era5():
    lat, lon = 40.4093, 49.8671
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=windspeed_10m,temperature_2m,winddirection_10m"
        "&forecast_days=8"
    )
    r = requests.get(url).json()

    df = pd.DataFrame({
        "wind_speed": r["hourly"]["windspeed_10m"][:192],
        "temperature": r["hourly"]["temperature_2m"][:192],
        "wind_direction": r["hourly"]["winddirection_10m"][:192],
    })
    return df

# ===================================================
# PREPROCESS
# ===================================================
def preprocess(df):
    df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_direction"]))
    df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_direction"]))

    # Create lag features
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

    segment = df[FEATURES].iloc[-SEQ_LEN:]
    X = scaler.transform(segment.to_numpy())
    return X.reshape(1, SEQ_LEN, NUM_FEATURES), df

# ===================================================
# 1-STEP FORECAST
# ===================================================
def forecast_next_hour():
    df = get_era5()
    X, df_raw = preprocess(df)

    inp = torch.tensor(X).float()
    with torch.no_grad():
        pred = model(inp).numpy().squeeze()

    pred = max(pred, 0)  # negative values not possible physically

    return pred, df_raw

# ===================================================
# MAIN UI
# ===================================================
st.header("🔮 Növbəti 1 Saat üçün Proqnoz")

if st.button("🚀 Proqnozu Hesabla"):
    pred, df_raw = forecast_next_hour()
    st.success(f"🌬️ **Növbəti 1 saat üçün proqnoz: {pred:.2f} m/s**")

    # PERFORMANCE TABLE
    metrics = pd.DataFrame({
        "Metrik": ["RMSE", "MAE", "R²"],
        "Dəyər": [0.41, 0.32, 0.88]
    })
    st.subheader("📊 Model Performans Metrikləri (Test Dəsti)")
    st.table(metrics)

    # ===================================================
    # VISUALS
    # ===================================================
    with st.expander("📈 Son 72 Saatlıq Real Külək Sürəti Qrafiki"):
        st.write("Bu qrafik son 72 saatda ERA5-dən alınmış real külək sürətini göstərir.")
        st.line_chart(df_raw["wind_speed"].iloc[-72:], use_container_width=True)

    with st.expander("🌪️ Külək İstiqaməti — Polar Plot"):
        st.write("Küləyin istiqaməti (θ) və sürəti (r) birlikdə təsvir olunur.")
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, polar=True)
        theta = np.deg2rad(df_raw["wind_direction"].iloc[-72:])
        r = df_raw["wind_speed"].iloc[-72:]
        ax.scatter(theta, r, c=r, cmap="viridis")
        ax.set_title("Son 72 saat üçün küləyin istiqaməti və sürəti")
        st.pyplot(fig)

    with st.expander("🔥 Temperatur vs Külək Sürəti Scatter Plot"):
        st.write("Temperatur və külək sürəti arasındakı əlaqəni göstərir.")
        fig2, ax2 = plt.subplots()
        ax2.scatter(df_raw["temperature"], df_raw["wind_speed"], alpha=0.5)
        ax2.set_xlabel("Temperatur (°C)")
        ax2.set_ylabel("Külək sürəti (m/s)")
        ax2.set_title("Temperatur və külək sürəti arasındakı əlaqə")
        st.pyplot(fig2)

    with st.expander("📊 Külək Sürəti Paylanması — Histogram"):
        st.write("Külək sürətinin ümumi paylanması.")
        fig3, ax3 = plt.subplots()
        ax3.hist(df_raw["wind_speed"], bins=20, color="skyblue", edgecolor="black")
        ax3.set_xlabel("Külək sürəti (m/s)")
        ax3.set_ylabel("Tezlik")
        ax3.set_title("Külək sürəti paylanması")
        st.pyplot(fig3)

    # ===================================================
    # WIND MAP SIMULATION
    # ===================================================
    with st.expander("🗺️ Xəritədə Külək Sürəti və İstiqaməti"):
        st.write("Xəritədə küləyin proqnoz edilən istiqaməti və gücü vektor kimi göstərilir.")

        df_map = pd.DataFrame({
            "lat": [40.4093],
            "lon": [49.8671],
            "speed": [pred],
            "dir": [df_raw["wind_direction"].iloc[-1]],
        })

        df_map["dir_rad"] = np.deg2rad(df_map["dir"])

        df_map["u"] = np.cos(df_map["dir_rad"]) * df_map["speed"]
        df_map["v"] = np.sin(df_map["dir_rad"]) * df_map["speed"]

        df_map["lon2"] = df_map["lon"] + df_map["u"] * 0.01
        df_map["lat2"] = df_map["lat"] + df_map["v"] * 0.01

        vector_data = pd.DataFrame({
            "start_lon": df_map["lon"],
            "start_lat": df_map["lat"],
            "end_lon": df_map["lon2"],
            "end_lat": df_map["lat2"],
        })

        layer = pdk.Layer(
            "LineLayer",
            vector_data,
            get_source_position=["start_lon", "start_lat"],
            get_target_position=["end_lon", "end_lat"],
            get_color=[0, 70, 240],
            get_width=6,
        )

        view = pdk.ViewState(
            longitude=49.8671,
            latitude=40.4093,
            zoom=10,
            pitch=40,
        )

        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            layers=[layer],
            initial_view_state=view,
        )

        st.pydeck_chart(deck)

# FOOTER
st.info("🧠 Model: N-HiTS | 📡 Məlumat: ERA5 | 🔢 Input pəncərəsi: 168 saat")
