import streamlit as st
import torch
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
from model import NHiTS
import os

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Külək Sürəti Proqnozu – N-HiTS",
    layout="wide",
    page_icon="🌬️"
)

# ============================================
# INTRO TEXT — Better Styled
# ============================================
st.title("🌬️ Azərbaycan üçün Külək Sürəti Proqnozu — N-HiTS Modeli")

st.markdown("""
<div style="padding:18px; border-radius:12px; background-color:#e9f3ff;">
    <h3 style="color:#004f8b;">📌 Layihənin Məqsədi</h3>
    Bu tətbiqin əsas məqsədi ERA5 real vaxt meteoroloji məlumatları əsasında 
    <b>növbəti 1 saat üçün külək sürətini proqnozlaşdırmaqdır.</b> 
    Sistem külək enerjisi istehsalında planlama, enerji şəbəkəsinin idarə edilməsi və 
    meteoroloji monitorinq kimi sahələrdə istifadə oluna bilər.

    <h3 style="color:#004f8b;">🧠 Model Necə İşləyir?</h3>
    Model N-HiTS dərin öyrənmə arxitekturasından istifadə edir və
    <b>168 saatlıq tarixi məlumat pəncərəsi</b> əsasında temperatur, külək istiqaməti,
    lag-lar, rollinq statistikaları kimi xüsusiyyətləri analiz edir.
    Daha sonra növbəti 1 saat üçün külək sürətini hesablayır.

    <h3 style="color:#004f8b;">🌍 Niyə Vacibdir?</h3>
    <ul>
        <li>⚡ Külək enerjisi istehsalının dəqiq planlaması</li>
        <li>🛡️ Enerji şəbəkəsində risklərin azaldılması</li>
        <li>📡 Real vaxt monitorinqi və analitika</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ============================================
# CONFIG
# ============================================
SEQ_LEN = 168
NUM_FEATURES = 14

FEATURES = [
    "temperature",
    "wind_dir_sin", "wind_dir_cos",
    "lag1", "lag3", "lag6", "lag12", "lag24",
    "roll6_mean", "roll12_mean", "roll24_mean",
    "roll6_std", "roll12_std", "roll24_std",
]

# ============================================
# LOAD MODEL + SCALER
# ============================================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(__file__)

    mean_path = os.path.join(BASE_DIR, "scaler_mean.npy")
    scale_path = os.path.join(BASE_DIR, "scaler_scale.npy")
    model_path = os.path.join(BASE_DIR, "n_hits_wind_model.pth")

    scaler_mean = np.load(mean_path)
    scaler_scale = np.load(scale_path)

    class SimpleScaler:
        def __init__(self, mean, scale):
            self.mean_ = mean
            self.scale_ = scale

        def transform(self, X):
            return (X - self.mean_) / self.scale_

    scaler = SimpleScaler(scaler_mean, scaler_scale)

    model = NHiTS(seq_len=SEQ_LEN, num_features=NUM_FEATURES)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, scaler

model, scaler = load_model()

# ============================================
# GET ERA5 REALTIME DATA
# ============================================
def get_era5():
    lat, lon = 40.4093, 49.8671
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
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

# ============================================
# PREPROCESS
# ============================================
def preprocess(df):
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

    df["roll6_std"]   = df["wind_speed"].rolling(6).std()
    df["roll12_std"]  = df["wind_speed"].rolling(12).std()
    df["roll24_std"]  = df["wind_speed"].rolling(24).std()

    df = df.dropna().reset_index(drop=True)

    segment = df[FEATURES].iloc[-SEQ_LEN:]
    X = scaler.transform(segment.to_numpy())

    return X.reshape(1, SEQ_LEN, NUM_FEATURES), df

# ============================================
# 1-STEP FORECAST
# ============================================
def forecast_next_hour():
    df = get_era5()
    X, df_processed = preprocess(df)

    inp = torch.tensor(X).float()
    with torch.no_grad():
        pred = model(inp).numpy().squeeze()

    pred = max(pred, 0)
    return pred, df

# ============================================
# MAIN UI
# ============================================
st.header("🔮 Növbəti 1 Saat üçün Külək Proqnozu")

if st.button("🚀 Proqnozu Hesabla"):
    pred, df_raw = forecast_next_hour()

    st.success(f"🌬️ **Proqnozlaşdırılan külək sürəti: {pred:.2f} m/s**")

    # ============================
    # PERFORMANCE TABLE
    # ============================
    metrics = pd.DataFrame({
        "Metrik": ["RMSE", "MAE", "R²"],
        "Dəyər": [22.587060150321918, 3.6778681608650263, 0.6715118127712671]
    })
    st.subheader("📊 Model Performans Metrikləri")
    st.table(metrics)

    # ============================
    # WIND MAP SIMULATION
    # ============================
    with st.expander("🗺️ Xəritədə Külək Sürəti Simulyasiyası"):
        st.markdown("Bu vizual proqnozlaşdırılan külək istiqamətini və gücünü xəritə üzərində ox şəklində göstərir.")

        df_map = pd.DataFrame({
            "lat": [40.4093],
            "lon": [49.8671],
            "speed": [pred],
            "dir": [df_raw["wind_direction"].iloc[-1]],
        })
        df_map["dir_rad"] = np.deg2rad(df_map["dir"])
        df_map["u"] = np.cos(df_map["dir_rad"]) * df_map["speed"]
        df_map["v"] = np.sin(df_map["dir_rad"]) * df_map["speed"]

        layer = pdk.Layer(
            "ArrowLayer",
            df_map,
            get_position=["lon", "lat"],
            get_direction=["u", "v"],
            get_color=[0, 100, 255],
            width_scale=8,
            get_length=1200,
        )

        view = pdk.ViewState(latitude=40.4093, longitude=49.8671, zoom=10)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))

    # ============================
    # VISUALIZATIONS
    # ============================
    with st.expander("📈 Son 72 Saatlıq Külək Sürəti (Real Məlumat)"):
        st.markdown("Bu qrafik son 72 saat ərzində ERA5 məlumatlarından alınan real külək sürətini göstərir.")
        st.line_chart(df_raw["wind_speed"].iloc[-72:])

    with st.expander("🌪️ Külək İstiqaməti — Polar Plot"):
        st.markdown("Bu polar qrafik son 72 saat üçün külək istiqaməti və sürətinin paylanmasını göstərir.")
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, polar=True)
        theta = np.deg2rad(df_raw["wind_direction"].iloc[-72:])
        r = df_raw["wind_speed"].iloc[-72:]
        ax.scatter(theta, r, c=r, cmap="viridis")
        ax.set_title("Külək İstiqaməti və Sürəti")
        st.pyplot(fig)

    with st.expander("🔥 Temperatur və Külək Sürəti — Scatter Plot"):
        st.markdown("Bu qrafik temperatur və külək sürəti arasındakı əlaqəni göstərir.")
        fig2, ax2 = plt.subplots(figsize=(5,4))
        ax2.scatter(df_raw["temperature"], df_raw["wind_speed"], alpha=0.5)
        ax2.set_xlabel("Temperatur (°C)")
        ax2.set_ylabel("Külək sürəti (m/s)")
        st.pyplot(fig2)

    with st.expander("📊 Külək Sürəti Paylanması — Histogram"):
        st.markdown("Bu histogram son məlumatlarda külək sürətinin paylanmasını göstərir.")
        fig3, ax3 = plt.subplots(figsize=(5,4))
        ax3.hist(df_raw["wind_speed"], bins=20, color="skyblue", edgecolor="black")
        ax3.set_xlabel("Külək sürəti (m/s)")
        ax3.set_ylabel("Tezlik")
        st.pyplot(fig3)

    # IG images
    with st.expander("🧠 Feature Importance (Integrated Gradients)"):
        st.image("feature_importance.png")

    with st.expander("🌬️ Modelin Nümunə Proqnozu"):
        st.image("wind_forecast_plot.png")

st.info("🧠 Model: N-HiTS | 📡 ERA5 Real-time Data | 🔢 Giriş pəncərəsi: 168 saat")
