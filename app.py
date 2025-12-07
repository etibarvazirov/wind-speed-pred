import streamlit as st
import torch
import requests
import numpy as np
import pandas as pd
from model import NHiTS

# ===================================================
# CONFIG
# ===================================================
SEQ_LEN = 168
NUM_FEATURES = 15
DEVICE = "cpu"

FEATURES = [
    "temperature", "wind_speed", "wind_dir_sin", "wind_dir_cos",
    "lag1", "lag3", "lag6", "lag12", "lag24",
    "roll6_mean", "roll12_mean", "roll24_mean",
    "roll6_std", "roll12_std", "roll24_std"
]

# ===================================================
# LOAD MODEL & SCALER
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
# GET ERA5
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

# ===================================================
# FORECAST
# ===================================================
def forecast(hours):
    df = get_era5()
    X, processed_df = preprocess(df)

    preds = []
    inp = torch.tensor(X).float()

    for _ in range(hours):
        with torch.no_grad():
            pred = model(inp).numpy().squeeze()

        preds.append(pred)

        new_row = processed_df[FEATURES].iloc[-1].copy()
        new_row["wind_speed"] = pred
        processed_df.loc[len(processed_df)] = new_row

        next_segment = processed_df[FEATURES].iloc[-SEQ_LEN:]
        X_next = scaler.transform(next_segment.to_numpy()).reshape(1, SEQ_LEN, NUM_FEATURES)
        inp = torch.tensor(X_next).float()

    return preds


# ===================================================
# STREAMLIT UI
# ===================================================
st.title("🌬️ Azərbaycan üçün Külək Sürəti Proqnozu — N-HiTS Modeli")
st.markdown("""
Bu sistem ERA5 real vaxt məlumatlarına əsaslanaraq **növbəti saatların külək sürətini**
N-HiTS kimi müasir dərin öyrənmə modeli vasitəsilə proqnozlaşdırır.
""")

hours = st.slider("⏳ Neçə saatlıq proqnoz istəyirsiniz?", 1, 24, 6)

if st.button("🔮 Proqnozu Hesabla"):
    preds = forecast(hours)

    st.success(f"📌 **Növbəti {hours} saat üçün son proqnoz:** {preds[-1]:.2f} m/s")

    # ---------------------------
    # RESULTS TABLE
    # ---------------------------
    df_pred = pd.DataFrame({
        "Saat": list(range(1, hours + 1)),
        "Proqnoz (m/s)": preds
    })
    st.subheader("📋 Proqnoz Cədvəli")
    st.dataframe(df_pred, use_container_width=True)

    # ---------------------------
    # ACCORDION – PLOTS
    # ---------------------------
    with st.expander("📈 Proqnoz Qrafiki"):
        st.line_chart(preds, use_container_width=True)
        st.caption("Model tərəfindən ardıcıl saatlıq proqnozların qrafiki")

    with st.expander("🧠 Feature Importance (Integrated Gradients)"):
        st.image("feature_importance.png", use_container_width=True)
        st.caption("Modelin hansı xüsusiyyətlərə daha çox önəm verdiyini göstərir.")

    with st.expander("🌬️ Model Nümunə Proqnozu"):
        st.image("wind_forecast_plot.png", use_container_width=True)
        st.caption("Modelin əvvəlki test nəticələrindən nümunə proqnoz.")


st.info("🧠 Model: N-HiTS | 📡 Məlumat: ERA5 | 🔢 168 saatlıq giriş pəncərəsi")
