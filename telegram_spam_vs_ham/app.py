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

# --------------------------------------------
# Sidebar
# --------------------------------------------

st.sidebar.title("📌 Overview")
st.sidebar.write("""
Spam Detection System using:

TF-IDF Vectorization
LinearSVC / Machine Learning Models

Detect whether a Telegram message is Spam or Ham.
""")

st.sidebar.title("📊 Data Section")
st.sidebar.markdown(
    "🔗 [Kaggle Dataset](https://www.kaggle.com/datasets/mexwell/telegram-spam-or-ham)"
)

st.sidebar.title("🔗 Contact")
st.sidebar.markdown("""
- 👨‍💻 LinkedIn: [Mohammed Waleed](https://www.linkedin.com/in/mohammed-waleed-533931375/)
- 💻 GitHub: [ML Projects](https://github.com/M-Waleed1/ML-Models)
""")



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

# --------------------------------------------
# Footer
# --------------------------------------------

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit + Scikit-learn")
