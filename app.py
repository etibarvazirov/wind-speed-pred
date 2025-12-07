import streamlit as st
import torch
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import NHiTS

# ============================================
# PAGE CONFIG + GLOBAL CSS (fix big charts)
# ============================================
st.set_page_config(
    page_title="Külək Sürəti Proqnozu – N-HiTS",
    layout="wide",
    page_icon="🌬️"
)

st.markdown("""
<style>
img, .stPlotlyChart, .stImage > img {
    max-width: 550px !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# INTRO SECTION
# ============================================

# Title Box
st.markdown("""
<div style="
    background-color:#e6f2ff;
    padding:18px;
    border-radius:12px;
    border:1px solid #bcd9ff;
">
    <h2 style="margin:0; padding:0;">🌬️ Külək Sürəti Proqnozu — N-HiTS Modeli</h2>
    <p style="margin-top:6px; font-size:16px;">
        ERA5 real vaxt məlumatları ilə dəqiq və stabil külək proqnozu
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")  # small spacing

# Description Box
st.markdown("""
<div style="
    background-color:#f7fbff;
    padding:16px;
    border-radius:10px;
    border-left: 5px solid #4da3ff;
    font-size:15px;
    line-height:1.5;
">
Bu tətbiq son <b>168 saatlıq ERA5 atmosfer məlumatlarından</b> istifadə edərək Azərbaycanın 
növbəti 1 saat üçün <b>külək sürətini proqnozlaşdırır</b>.

Model <b>N-HiTS</b> dərin öyrənmə arxitekturası ilə öyrədilmişdir və 
proqnozlarda yüksək stabillik və ardıcıllıq təmin edir.
</div>
""", unsafe_allow_html=True)


with st.expander("ℹ️ Modeldə istifadə olunan xüsusiyyətlər haqqında qısa izah"):
    st.markdown("""
- **temperature:** hava temperaturu  
- **wind_dir_sin / wind_dir_cos:** külək istiqamətinin trigonometrik kodlanması  
- **lag1…lag24:** əvvəlki saatlardan gecikmə xüsusiyyətləri  
- **roll_mean / roll_std:** küləyin son saatlardakı orta qiymətləri və dəyişkənliyi  

Bu xüsusiyyətlər birlikdə modelə külək dinamikasını öyrənməyə kömək edir.
""")

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

# ============================================
# GET ERA5
# ============================================
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
# ONE-STEP FORECAST
# ============================================
def forecast_next_hour():
    df = get_era5()
    X, _ = preprocess(df)
    inp = torch.tensor(X).float()

    with torch.no_grad():
        pred = model(inp).numpy().squeeze()

    return max(pred, 0), df

# ============================================
# MAIN UI
# ============================================
st.markdown("## 🔮 1 Saatlıq Külək Proqnozu")

if st.button("🚀 Proqnozu Hesabla"):
    pred, df_raw = forecast_next_hour()

    st.success(f"🌬️ **Proqnozlaşdırılan külək sürəti: {pred:.2f} m/s**")

    metrics = pd.DataFrame({
        "Metrik": ["RMSE", "MAE", "R²"],
        "Dəyər": [22.587060150321918, 3.6778681608650263, 0.6715118127712671]
    })
    st.subheader("📊 Model Performansı")
    st.table(metrics)

    # ============================
    # VISUALS
    # ============================

    with st.expander("📈 Son 72 Saatlıq Külək Sürətinin Dəyişimi"):
        st.write("Küləyin son 3 gündə necə dəyişdiyini göstərir. Piklər küləyin gücləndiyi saatlardır.")
        st.line_chart(df_raw["wind_speed"].iloc[-72:])

    with st.expander("🌪️ Külək İstiqaməti — Polar Plot"):
        st.write("Nöqtələrin bucağı istiqaməti, məsafəsi isə küləyin gücünü göstərir.")
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, polar=True)
        theta = np.deg2rad(df_raw["wind_direction"].iloc[-72:])
        r = df_raw["wind_speed"].iloc[-72:]
        ax.scatter(theta, r, c=r, cmap="viridis", s=10)
        st.pyplot(fig, use_container_width=False)

    with st.expander("🔥 Temperatur və Külək Sürətinin Əlaqəsi"):
        fig2, ax2 = plt.subplots(figsize=(5,4))
        ax2.scatter(df_raw["temperature"], df_raw["wind_speed"], alpha=0.5)
        ax2.set_xlabel("Temperatur (°C)")
        ax2.set_ylabel("Külək (m/s)")
        st.write("Temperaturun yüksəlməsi külək sürətini hər zaman artırmır — əlaqə zəifdir.")
        st.pyplot(fig2, use_container_width=False)

    with st.expander("📊 Külək Sürəti Paylanması"):
        fig3, ax3 = plt.subplots(figsize=(5,4))
        ax3.hist(df_raw["wind_speed"], bins=20, color="skyblue")
        ax3.set_xlabel("Külək (m/s)")
        ax3.set_ylabel("Tezlik")
        st.write("Əksər saatlarda külək sürəti orta dəyərlər ətrafında toplanıb.")
        st.pyplot(fig3, use_container_width=False)

    with st.expander("🧠 Feature Importance (IG)"):
        st.write("Modelin qərar verməsinə ən çox təsir edən xüsusiyyətlər.")
        st.image("feature_importance.png")

    with st.expander("🌬️ Modelin Nümunə Proqnozu"):
        st.image("wind_forecast_plot.png", width=550)

st.info("🧠 Model: N-HiTS | 📡 ERA5 məlumatları | 🔢 Giriş pəncərəsi: 168 saat")


