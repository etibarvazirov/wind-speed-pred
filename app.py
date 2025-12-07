import streamlit as st
import torch
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import NHiTS

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Külək Sürəti Proqnozu – N-HiTS",
    layout="wide",
    page_icon="🌬️"
)

# ============================================
# INTRO SECTION (HTML YOXDUR — NORMAL STREAMLIT)
# ============================================
st.title("🌬️ Azərbaycan üçün Külək Sürəti Proqnozu — N-HiTS Modeli")

st.markdown("""
### 📌 Layihə Haqqında

Bu tətbiq **ERA5 real vaxt atmosfer məlumatları** əsasında Azərbaycanın növbəti 1 saat üçün  
**külək sürəti proqnozunu** təqdim edir. Model **N-HiTS dərin öyrənmə arxitekturası** ilə tren olunub  
və son 168 saatlıq məlumatı giriş kimi istifadə edir.

Proqnozlar külək enerjisi planlamasında, aerodinamik hesablarda və təhlükəsizlik qiymətləndirməsində faydalıdır.
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
# GET ERA5 REALTIME DATA
# ============================================
def get_era5():
    lat, lon = 40.4093, 49.8671  # Baku
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
# 1-STEP FORECAST
# ============================================
def forecast_next_hour():
    df = get_era5()
    X, df_processed = preprocess(df)

    inp = torch.tensor(X).float()
    with torch.no_grad():
        pred = model(inp).numpy().squeeze()

    pred = max(pred, 0)  # no negative winds
    return pred, df

# ============================================
# MAIN UI
# ============================================
st.header("🔮 Növbəti 1 Saat üçün Proqnoz")

if st.button("🚀 Proqnozu Hesabla"):
    pred, df_raw = forecast_next_hour()

    st.success(f"🌬️ **Proqnozlaşdırılan külək sürəti: {pred:.2f} m/s**")

    # PERFORMANCE TABLE
    metrics = pd.DataFrame({
        "Metrik": ["RMSE", "MAE", "R²"],
        "Dəyər": [22.587060150321918, 3.6778681608650263, 0.6715118127712671]
    })
    st.subheader("📊 Model Performans Metrikləri")
    st.table(metrics)

    # ============================
    # VISUALS — WITH SHORT EXPLANATION
    # ============================

    with st.expander("📈 Son 72 Saatlıq Real Külək Sürəti"):
        st.write("Bu qrafik ERA5 məlumatlarına əsasən son 72 saatda külək sürətinin dəyişməsini göstərir.")
        st.line_chart(df_raw["wind_speed"].iloc[-72:])

    with st.expander("🌪️ Külək İstiqaməti — Polar Plot (Wind Rose)"):
        st.write("Nöqtələr küləyin istiqamətini və gücünü göstərir. Daha uzaqdakı nöqtələr daha güclü küləyə uyğundur.")
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, polar=True)
        theta = np.deg2rad(df_raw["wind_direction"].iloc[-72:])
        r = df_raw["wind_speed"].iloc[-72:]
        ax.scatter(theta, r, c=r, cmap="viridis", s=20)
        st.pyplot(fig)

    with st.expander("🔥 Temperatur və Külək Sürətinin Əlaqəsi"):
        st.write("Bu scatter plot temperaturun artması və külək sürəti arasında hər hansı əlaqəni müşahidə etməyə imkan verir.")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.scatter(df_raw["temperature"], df_raw["wind_speed"], alpha=0.5)
        ax2.set_xlabel("Temperatur (°C)")
        ax2.set_ylabel("Külək sürəti (m/s)")
        st.pyplot(fig2)

    with st.expander("📊 Külək Sürəti Paylanması — Histogram"):
        st.write("Bu histogram son məlumatlarda külək sürətinin hansı aralıqlarda daha çox baş verdiyini göstərir.")
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.hist(df_raw["wind_speed"], bins=20, color="skyblue", edgecolor="black")
        ax3.set_xlabel("Külək sürəti (m/s)")
        ax3.set_ylabel("Tezlik")
        st.pyplot(fig3)

    with st.expander("🧠 Feature Importance (Integrated Gradients)"):
        st.write("Modelin qərar verməsində hansı xüsusiyyətlərin daha böyük rol oynadığını göstərir.")
        st.image("feature_importance.png", use_container_width=True)

    with st.expander("🌬️ Modelin Nümunə Proqnozu"):
        st.write("Bu şəkildə modelin müəyyən tarix üçün verdiyi nümunə proqnoz göstərilir.")
        st.image("wind_forecast_plot.png", use_container_width=True)

st.info("🧠 Model: N-HiTS | 📡 Məlumat: ERA5 | 🔢 Son 168 saat input kimi istifadə olunur")
