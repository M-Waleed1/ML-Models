import streamlit as st
import joblib
import pandas as pd
import os

# -----------------------------
# Load Model
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'pipeline.pkl')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Spam Detector", page_icon="📱")

st.title("📱 Spam / Ham Detector")
st.write("Enter a message to check if it's Spam or Ham")

# -----------------------------
# Single Prediction
# -----------------------------
message = st.text_area("Message")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Enter a message first")
    else:
        pred = model.predict([message])[0]

        if pred == "spam":
            st.error("❌ Spam Message")
        else:
            st.success("✅ Ham Message")

# -----------------------------
# Batch Prediction
# -----------------------------
st.markdown("---")
st.subheader("Batch Prediction (CSV)")

file = st.file_uploader("Upload CSV with 'text' column")

if file is not None:
    df = pd.read_csv(file)

    if "text" in df.columns:
        df["prediction"] = model.predict(df["text"])

        st.write(df)

        csv = df.to_csv(index=False)
        st.download_button("Download Results", csv, "results.csv")
    else:
        st.error("CSV must contain 'text' column")
