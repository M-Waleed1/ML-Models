# ============================================
# Streamlit App - Fake News Classification
# ============================================

import streamlit as st
import joblib
import os

# --------------------------------------------
# Load trained model pipeline
# --------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'pipeline.pkl')

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# --------------------------------------------
# Page Config
# --------------------------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# --------------------------------------------
# Sidebar
# --------------------------------------------

st.sidebar.title("📌 Overview")
st.sidebar.write("""
Fake News Detection System using:
- TF-IDF Vectorization
- LinearSVC / ML Models

Detect whether a news article is Fake or Real.
""")

st.sidebar.title("📊 Data Section")
st.sidebar.markdown(
    "🔗 [Kaggle Dataset](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection/data)"
)

st.sidebar.title("🔗 Contact")
st.sidebar.markdown("""
- 👨‍💻 LinkedIn: [Mohammed Waleed](https://www.linkedin.com/in/mohammed-waleed-533931375/)
- 💻 GitHub: [ML Projects](https://github.com/M-Waleed1/ML-Models)
""")

# --------------------------------------------
# Main UI
# --------------------------------------------

st.title("📰 Fake News Detector")
st.write("Enter a news text below:")

user_input = st.text_area("News Text", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter text.")
    else:
        prediction = model.predict([user_input])[0]

        st.subheader("Result")

        if prediction == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")

        # Confidence (if available)
        if hasattr(model.named_steps['clf'], "predict_proba"):
            proba = model.predict_proba([user_input])[0]
            st.write("Confidence:")
            st.write({
                "Fake": float(proba[0]),
                "Real": float(proba[1])
            })

# --------------------------------------------
# Footer
# --------------------------------------------

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit + Scikit-learn")
