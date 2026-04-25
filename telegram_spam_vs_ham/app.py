# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Spam/Ham Message Detector",
    page_icon="📱",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px;
    }
    .prediction-spam {
        background-color: #ff6b6b;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-ham {
        background-color: #51cf66;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('pipeline.pkl')
        return model
    except FileNotFoundError:
        st.error("Model not found! Please train the model first by running train_model.py")
        return None

# Function to predict single message
def predict_message(model, message):
    prediction = model.predict([message])[0]
    probability = None
    
    # Get probability if model supports it
    if hasattr(model.named_steps['clf'], 'predict_proba'):
        proba = model.named_steps['clf'].predict_proba(
            model.named_steps['Tfidf'].transform([message])
        )
        probability = proba[0]
    
    return prediction, probability

# Function to evaluate on test data
def evaluate_model(model, xtest, ytest):
    y_pred = model.predict(xtest)
    accuracy = accuracy_score(ytest, y_pred)
    report = classification_report(ytest, y_pred, output_dict=True)
    cm = confusion_matrix(ytest, y_pred)
    return accuracy, report, cm

# Main app
def main():
    st.title("📱 Telegram Spam/Ham Message Detector")
    st.markdown("### Detect whether a message is Spam or Ham (legitimate)")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app uses Machine Learning to classify messages as:
        - **Spam**: Unwanted, promotional, or malicious messages
        - **Ham**: Legitimate, wanted messages
        
        **Models used:**
        - Logistic Regression
        - Support Vector Machine (SVM)
        - Naive Bayes
        
        **Features:**
        - TF-IDF Vectorization
        - N-gram range (1,2)
        - 5000 max features
        """)
        
        st.header("Sample Messages")
        st.markdown("""
        **Spam examples:**
        - "CONGRATULATIONS! You've won $1,000,000! Click here to claim your prize"
        - "FREE iPhone giveaway! Limited time offer!"
        
        **Ham examples:**
        - "Hey, how are you doing today?"
        - "Don't forget about our meeting at 3pm"
        """)
    
    # Load model
    model = load_model()
    
    if model is not None:
        # Create tabs
        tab1, tab2 = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction"])
        
        # Tab 1: Single Prediction
        with tab1:
            st.header("Test a Single Message")
            
            # Text input
            message = st.text_area(
                "Enter your message here:",
                height=150,
                placeholder="Type or paste a message to check if it's spam or ham..."
            )
            
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                predict_button = st.button("🔍 Predict", use_container_width=True)
            
            if predict_button and message:
                with st.spinner("Analyzing message..."):
                    prediction, probability = predict_message(model, message)
                    
                    # Display prediction
                    if prediction == "spam":
                        st.markdown("""
                        <div class='prediction-spam'>
                            <h2>⚠️ SPAM DETECTED</h2>
                            <p>This message appears to be spam!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class='prediction-ham'>
                            <h2>✅ HAM (Legitimate)</h2>
                            <p>This message appears to be legitimate!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display probability if available
                    if probability is not None:
                        st.subheader("Prediction Confidence")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Spam Probability", f"{probability[1]*100:.2f}%")
                        with col2:
                            st.metric("Ham Probability", f"{probability[0]*100:.2f}%")
                        
                        # Progress bar
                        st.progress(float(probability[1]))
            
            elif predict_button and not message:
                st.warning("Please enter a message to analyze.")
        
        # Tab 2: Batch Prediction (Upload CSV)
        with tab2:
            st.header("Batch Prediction from CSV File")
            st.markdown("Upload a CSV file with a 'text' column containing messages to classify.")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    if 'text' in df.columns:
                        with st.spinner("Processing messages..."):
                            # Make predictions
                            predictions = model.predict(df['text'])
                            df['prediction'] = predictions
                            
                            # Show results
                            st.success(f"Processed {len(df)} messages successfully!")
                            
                            # Display statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Messages", len(df))
                            with col2:
                                spam_count = (predictions == "spam").sum()
                                st.metric("Spam Detected", spam_count)
                            with col3:
                                ham_count = (predictions == "ham").sum()
                                st.metric("Ham Detected", ham_count)
                            
                            # Show results dataframe
                            st.subheader("Results")
                            st.dataframe(df, use_container_width=True)
                            
                            # Download button
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="spam_detection_results.csv",
                                mime="text/csv"
                            )
                    else:
                        st.error("CSV file must contain a 'text' column")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
if __name__ == "__main__":
    main()