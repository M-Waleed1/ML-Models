import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with metric color options
st.markdown("""
<style>
    /* Main button styling */
    .stButton > button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #ff6b6b;
        transform: translateY(-2px);
    }
    
    /* Risk level boxes */
    .high-risk {
        background: linear-gradient(135deg, #ffcccc 0%, #ff9999 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .medium-risk {
        background: linear-gradient(135deg, #ffffcc 0%, #ffe699 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffa500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .low-risk {
        background: linear-gradient(135deg, #ccffcc 0%, #99ff99 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00ff00;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metric card styling - NEW CUSTOM COLORS */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        color: white;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card h3 {
        font-size: 32px;
        margin: 0;
        font-weight: bold;
    }
    
    .metric-card p {
        margin: 10px 0 0 0;
        font-size: 14px;
        opacity: 0.9;
    }
    
    /* Custom metric color variants */
    .metric-blue {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
    }
    .metric-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-orange {
        background: linear-gradient(135deg, #f12711 0%, #f5af19 100%);
    }
    .metric-purple {
        background: linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%);
    }
    .metric-pink {
        background: linear-gradient(135deg, #ee9ca7 0%, #ffdde1 100%);
    }
    .metric-dark {
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
    }
    
    /* Prediction result cards */
    .prediction-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-top: 4px solid #ff4b4b;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('Final_Pipeline.pkl')
        return model
    except:
        st.error("❌ Model file 'Final_Pipeline.pkl' not found!")
        return None

model = load_model()

# Feature engineering function
def engineer_features(df):
    """Apply feature engineering exactly as in training"""
    df = df.copy()
    
    # Tenure group
    df['tenure_group'] = pd.cut(df['tenure'], 
                                bins=[0, 12, 24, 48, 72, 100], 
                                labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yrs'])
    
    # Average monthly spend per tenure
    df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # High-value customer flag
    df['high_value'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
    
    # Senior citizen with no partner
    df['senior_no_partner'] = ((df['SeniorCitizen'] == 1) & (df['Partner'] == 'No')).astype(int)
    
    # Multiple services flag
    services = ['PhoneService', 'InternetService', 'StreamingTV', 'StreamingMovies']
    df['num_services'] = df[services].apply(lambda x: (x != 'No').sum(), axis=1)
    
    # Risk score
    df['risk_score'] = (
        (df['Contract'] == 'Month-to-month').astype(int) * 3 +
        (df['PaperlessBilling'] == 'Yes').astype(int) * 2 +
        (df['PaymentMethod'] == 'Electronic check').astype(int) * 2
    )
    
    return df

# Prediction function
def predict_churn(customer_data):
    """Make prediction for single customer"""
    df = pd.DataFrame([customer_data])
    df = engineer_features(df)
    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return prediction, probability

# Batch prediction function
def predict_batch(df):
    """Make predictions for multiple customers"""
    df_engineered = engineer_features(df)
    predictions = model.predict(df_engineered)
    probabilities = model.predict_proba(df_engineered)[:, 1]
    
    return predictions, probabilities

# Function to display metric with custom color
def display_metric(label, value, color_style="default", help_text=None):
    """Display metric with different color schemes"""
    color_classes = {
        "default": "metric-card",
        "blue": "metric-card metric-blue",
        "green": "metric-card metric-green", 
        "orange": "metric-card metric-orange",
        "purple": "metric-card metric-purple",
        "pink": "metric-card metric-pink",
        "dark": "metric-card metric-dark"
    }
    
    css_class = color_classes.get(color_style, color_classes["default"])
    
    if help_text:
        st.markdown(f"""
        <div class="{css_class}" title="{help_text}">
            <p style="font-size: 14px; margin:0">{label}</p>
            <h3 style="margin:5px 0">{value}</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="{css_class}">
            <p style="font-size: 14px; margin:0">{label}</p>
            <h3 style="margin:5px 0">{value}</h3>
        </div>
        """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("🎯 Navigation")
    
    page = st.radio("", [
        "🏠 Home",
        "🔍 Single Prediction",
        "📁 Batch Prediction",
        "📊 Model Insights",
        "📈 Analytics Dashboard"
    ])
    
    st.markdown("---")
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    display_metric("Accuracy", "80.4%", 'blue', "Model accuracy on test data")
    display_metric("ROC-AUC", "0.8465", 'blue', "Area under ROC curve")
    display_metric("Model Type", "Logistic Regression", 'blue', "Classification algorithm")
    
    st.markdown("---")
    st.markdown("### 💡 Tip")
    st.info("High-risk customers (probability >70%) need immediate retention actions!")

# Home Page
if page == "🏠 Home":
    st.title("🏠 Customer Churn Prediction System")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_metric("Model Accuracy", "80.4%", 'blue', "Accuracy on test set")
    
    with col2:
        display_metric("ROC-AUC Score", "84.6%", 'blue', "Area under ROC curve")
    
    with col3:
        display_metric("Features Used", "30+", 'blue', "Total features in model")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 What is Churn Prediction?")
        st.write("""
        Customer churn prediction helps identify customers who are likely to stop using 
        your service. Our AI model analyzes customer behavior, contract details, and 
        usage patterns to predict churn risk with 80.4% accuracy.
        """)
        
        st.subheader("🔑 Key Features")
        st.markdown("""
        - ✅ **Real-time predictions** - Get instant churn risk assessment
        - ✅ **Batch processing** - Upload CSV for bulk predictions  
        - ✅ **Risk analysis** - Understand why customers might churn
        - ✅ **Actionable insights** - Get retention recommendations
        """)
    
    with col2:
        st.subheader("📊 How It Works")
        st.markdown("""
        1. **Enter customer details** in Single Prediction tab
        2. **AI analyzes** 30+ features and patterns
        3. **Get instant risk score** (Low/Medium/High)
        4. **Receive recommendations** for retention
        """)
        
        st.subheader("🎯 Risk Levels")
        st.markdown("""
        - 🔴 **High Risk (>70%)** - Immediate action required
        - 🟡 **Medium Risk (40-70%)** - Monitor closely
        - 🟢 **Low Risk (<40%)** - Regular maintenance
        """)
    
    st.markdown("---")
    st.info("👈 **Get Started:** Use the sidebar to navigate to Single Prediction or Batch Prediction")

# Single Prediction Page
elif page == "🔍 Single Prediction":
    st.title("🔍 Customer Churn Predictor")
    st.markdown("Enter customer details to get churn prediction")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Customer Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12, help="How long customer has been with company")
        
        st.subheader("💰 Account Information")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", "Bank transfer (automatic)", 
                                      "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=70.0, step=5.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=tenure * monthly_charges)
    
    with col2:
        st.subheader("📞 Service Information")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    st.markdown("---")
    
    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        if model is None:
            st.error("Model not loaded!")
        else:
            # Prepare data
            customer_data = {
                "gender": gender,
                "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }
            
            with st.spinner("Analyzing customer data..."):
                prediction, probability = predict_churn(customer_data)
                
                # Display results with custom colors
                st.subheader("📊 Prediction Result")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    prediction_text = "Will Churn" if prediction == 1 else "Will Not Churn"
                    color = "orange" if prediction == 1 else "green"
                    display_metric("Prediction", prediction_text, color)
                
                with col2:
                    display_metric("Churn Probability", f"{probability*100:.1f}%", "blue")
                
                with col3:
                    risk_text = "HIGH RISK" if probability >= 0.7 else "MEDIUM RISK" if probability >= 0.4 else "LOW RISK"
                    risk_color = "orange" if probability >= 0.7 else "pink" if probability >= 0.4 else "green"
                    display_metric("Risk Level", risk_text, risk_color)
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    title = {'text': "Churn Risk Score"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': probability * 100
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendation box
                if probability >= 0.7:
                    risk_class = "high-risk"
                    recommendation = "🚨 Immediate action required! Offer discount or loyalty program."
                elif probability >= 0.4:
                    risk_class = "medium-risk"
                    recommendation = "⚠️ Monitor closely. Send engagement offers and retention campaigns."
                else:
                    risk_class = "low-risk"
                    recommendation = "✅ Customer is loyal. Continue quality service and offer referral programs."
                
                st.markdown(f"""
                <div class="{risk_class}">
                    <h4>💡 Recommendation</h4>
                    <p>{recommendation}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Key risk factors
                if probability >= 0.4:
                    st.subheader("⚠️ Key Risk Factors")
                    risk_factors = []
                    
                    if contract == "Month-to-month":
                        risk_factors.append("• Month-to-month contract (3x higher churn risk)")
                    if payment_method == "Electronic check":
                        risk_factors.append("• Electronic check payment (2x higher churn risk)")
                    if online_security == "No":
                        risk_factors.append("• No online security service")
                    if paperless_billing == "Yes":
                        risk_factors.append("• Paperless billing enabled")
                    if tenure < 12:
                        risk_factors.append("• New customer (less than 1 year tenure)")
                    
                    for factor in risk_factors:
                        st.warning(factor)

# Fixed Batch Prediction Page
elif page == "📁 Batch Prediction":
    st.title("📁 Batch Prediction")
    st.markdown("Upload CSV file to predict churn for multiple customers")
    st.markdown("---")
    
    st.info("""
    **CSV Format Requirements:**
    - Must include all the same columns as single prediction
    - Download sample template below
    """)
    
    # Create sample template with ALL required columns
    sample_data = {
        'gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'Yes'],
        'tenure': [12, 24],
        'PhoneService': ['Yes', 'Yes'],
        'MultipleLines': ['No', 'Yes'],
        'InternetService': ['Fiber optic', 'DSL'],
        'OnlineSecurity': ['No', 'Yes'],
        'OnlineBackup': ['No', 'No'],
        'DeviceProtection': ['No', 'Yes'],
        'TechSupport': ['No', 'No'],
        'StreamingTV': ['Yes', 'No'],
        'StreamingMovies': ['No', 'No'],
        'Contract': ['Month-to-month', 'One year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Bank transfer (automatic)'],
        'MonthlyCharges': [75.0, 65.0],
        'TotalCharges': [900.0, 1560.0]
    }
    sample_df = pd.DataFrame(sample_data)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_df.to_csv(index=False),
            file_name="sample_customers.csv",
            mime="text/csv"
        )
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Check for required columns
        required_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                           'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                           'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                           'MonthlyCharges', 'TotalCharges']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info("Please use the sample CSV template as reference")
        else:
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(df.head(10))
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Total Records:** {len(df)}")
            with col2:
                st.write(f"**Columns:** {len(df.columns)}")
            
            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                with st.spinner(f"Processing {len(df)} customers..."):
                    try:
                        # Make a copy before feature engineering
                        df_processed = df.copy()
                        
                        # Apply the EXACT same feature engineering as training
                        # Tenure group
                        df_processed['tenure_group'] = pd.cut(df_processed['tenure'], 
                                                            bins=[0, 12, 24, 48, 72, 100], 
                                                            labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yrs'])
                        
                        # Average monthly spend per tenure
                        df_processed['avg_monthly_spend'] = df_processed['TotalCharges'] / (df_processed['tenure'] + 1)
                        
                        # High-value customer flag
                        df_processed['high_value'] = (df_processed['MonthlyCharges'] > df_processed['MonthlyCharges'].median()).astype(int)
                        
                        # Senior citizen with no partner
                        df_processed['senior_no_partner'] = ((df_processed['SeniorCitizen'] == 1) & (df_processed['Partner'] == 'No')).astype(int)
                        
                        # Multiple services flag
                        services = ['PhoneService', 'InternetService', 'StreamingTV', 'StreamingMovies']
                        df_processed['num_services'] = df_processed[services].apply(lambda x: (x != 'No').sum(), axis=1)
                        
                        # Risk score
                        df_processed['risk_score'] = (
                            (df_processed['Contract'] == 'Month-to-month').astype(int) * 3 +
                            (df_processed['PaperlessBilling'] == 'Yes').astype(int) * 2 +
                            (df_processed['PaymentMethod'] == 'Electronic check').astype(int) * 2
                        )
                        
                        # Drop customerID if exists (like in training)
                        if 'customerID' in df_processed.columns:
                            df_processed = df_processed.drop('customerID', axis=1)
                        
                        # Make predictions
                        predictions = model.predict(df_processed)
                        probabilities = model.predict_proba(df_processed)[:, 1]
                        
                        # Add predictions to dataframe
                        df_results = df.copy()
                        df_results['Churn_Prediction'] = ['Will Churn' if p == 1 else 'Will Not Churn' for p in predictions]
                        df_results['Churn_Probability'] = probabilities
                        df_results['Risk_Level'] = df_results['Churn_Probability'].apply(
                            lambda x: 'High' if x >= 0.7 else 'Medium' if x >= 0.4 else 'Low'
                        )
                        
                        # Summary stats with custom colors
                        st.subheader("📊 Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            display_metric("Total Customers", len(df_results), 'blue')
                        with col2:
                            churn_count = (df_results['Churn_Prediction'] == 'Will Churn').sum()
                            display_metric("Expected Churn", churn_count, "orange")
                        with col3:
                            high_risk = (df_results['Risk_Level'] == 'High').sum()
                            display_metric("High Risk", high_risk, "orange")
                        with col4:
                            avg_risk = f"{df_results['Churn_Probability'].mean()*100:.1f}%"
                            display_metric("Avg Risk Score", avg_risk, "blue")
                        
                        # Risk distribution chart
                        risk_counts = df_results['Risk_Level'].value_counts()
                        fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                                    title="Risk Level Distribution",
                                    color=risk_counts.index,
                                    color_discrete_map={'High':'red', 'Medium':'yellow', 'Low':'green'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display results
                        st.subheader("📋 Detailed Results")
                        display_cols = ['gender', 'tenure', 'Contract', 'MonthlyCharges', 
                                       'Churn_Prediction', 'Churn_Probability', 'Risk_Level']
                        # Only show columns that exist
                        display_cols = [col for col in display_cols if col in df_results.columns]
                        st.dataframe(df_results[display_cols].head(20))
                        
                        # Download results
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # High risk customers list
                        high_risk = df_results[df_results['Risk_Level'] == 'High']
                        if len(high_risk) > 0:
                            st.subheader("⚠️ High Risk Customers (Immediate Action Required)")
                            st.warning(f"{len(high_risk)} customers need immediate retention attention!")
                            st.dataframe(high_risk[['tenure', 'Contract', 'MonthlyCharges', 'Churn_Probability']].head(10))
                    
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.info("Make sure your CSV has the same format as the sample template")

# Model Insights Page
elif page == "📊 Model Insights":
    st.title("📊 Model Insights & Interpretation")
    st.markdown("---")
    
    st.subheader("🎯 Top Factors Driving Customer Churn")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔴 High Risk Factors
        
        **Contract Type**
        - Month-to-month contracts have **3x higher** churn risk
        - Long-term contracts (1-2 years) are much more stable
        
        **Payment Method**
        - Electronic check users are **2x more likely** to churn
        - Automatic payment methods show higher loyalty
        
        **Service Adoption**
        - No online security/backup increases churn risk
        - Fiber optic customers churn more than DSL users
        
        **Billing Preferences**
        - Paperless billing correlates with higher churn
        - Higher monthly charges increase churn probability
        """)
    
    with col2:
        st.markdown("""
        ### 🟢 Loyalty Factors
        
        **Customer Tenure**
        - Longer tenure = lower churn risk
        - Most churn happens in first 12 months
        
        **Service Bundling**
        - Multiple services reduce churn by 40%
        - Streaming services increase stickiness
        
        **Customer Support**
        - Having tech support reduces churn
        - Device protection adds loyalty
        
        **Demographics**
        - Seniors with partners are more loyal
        - Customers with dependents churn less
        """)
    
    st.markdown("---")
    
    st.subheader("💡 Actionable Recommendations by Risk Level")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="high-risk">
            <h4>🔴 High Risk (>70%)</h4>
            <ul>
                <li>Call customer immediately</li>
                <li>Offer contract upgrade discount</li>
                <li>Provide loyalty points</li>
                <li>Suggest service bundles</li>
                <li>Assign retention specialist</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="medium-risk">
            <h4>🟡 Medium Risk (40-70%)</h4>
            <ul>
                <li>Send personalized email offers</li>
                <li>Recommend service upgrades</li>
                <li>Share loyalty program benefits</li>
                <li>Offer free trial of premium services</li>
                <li>Schedule satisfaction survey</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="low-risk">
            <h4>🟢 Low Risk (<40%)</h4>
            <ul>
                <li>Regular satisfaction check-ins</li>
                <li>Referral program promotion</li>
                <li>Thank you loyalty rewards</li>
                <li>Feature new services</li>
                <li>Maintain quality service</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📈 Business Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        display_metric("Potential Savings", "$500K+", "green", "per 1000 customers")
        st.caption("Based on retaining high-risk customers")
    
    with col2:
        display_metric("ROI", "300%", "blue", "average return on investment")
        st.caption("Return on retention investment")

# Analytics Dashboard Page
elif page == "📈 Analytics Dashboard":
    st.title("📈 Analytics Dashboard")
    st.markdown("Upload customer data for detailed analytics")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload CSV for analysis", type=['csv'], key="analytics")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        with st.spinner("Analyzing data..."):
            predictions, probabilities = predict_batch(df)
            
            df_analysis = df.copy()
            df_analysis['Churn_Probability'] = probabilities
            df_analysis['Risk_Level'] = df_analysis['Churn_Probability'].apply(
                lambda x: 'High' if x >= 0.7 else 'Medium' if x >= 0.4 else 'Low'
            )
            
            # Dashboard metrics with custom colors
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                display_metric("Total Customers", len(df_analysis), 'blue')
            with col2:
                high_risk_count = (df_analysis['Risk_Level'] == 'High').sum()
                display_metric("High Risk Customers", high_risk_count, "orange")
            with col3:
                churn_rate = f"{df_analysis['Churn_Probability'].mean()*100:.1f}%"
                display_metric("Expected Churn Rate", churn_rate, "pink")
            with col4:
                revenue_at_risk = f"${df_analysis['MonthlyCharges'].sum() * df_analysis['Churn_Probability'].mean():,.0f}"
                display_metric("Revenue at Risk", revenue_at_risk, "orange")
            
            # Risk by contract type
            st.subheader("📊 Churn Risk by Contract Type")
            contract_risk = df_analysis.groupby('Contract')['Churn_Probability'].mean().sort_values(ascending=False)
            fig = px.bar(x=contract_risk.values, y=contract_risk.index, 
                        orientation='h', color=contract_risk.values,
                        color_continuous_scale='RdYlGn_r',
                        title="Average Churn Probability by Contract Type")
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk by tenure
            st.subheader("📈 Churn Risk by Tenure")
            df_analysis['tenure_group'] = pd.cut(df_analysis['tenure'], 
                                                bins=[0, 6, 12, 24, 48, 72], 
                                                labels=['0-6mo', '6-12mo', '1-2yr', '2-4yr', '4-6yr'])
            tenure_risk = df_analysis.groupby('tenure_group')['Churn_Probability'].mean()
            fig = px.line(x=tenure_risk.index, y=tenure_risk.values,
                         markers=True, title="Churn Risk Trend by Tenure")
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk distribution
            st.subheader("Risk Level Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                risk_counts = df_analysis['Risk_Level'].value_counts()
                fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                            title="Customer Risk Distribution",
                            color_discrete_map={'High':'red', 'Medium':'orange', 'Low':'green'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(df_analysis, x='Churn_Probability', 
                                  nbins=20, title="Probability Distribution",
                                  color_discrete_sequence=['blue'])
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🚀 Powered by Machine Learning | Model Accuracy: 80.4% | Last Updated: 2024</p>
    <p>💡 For high-risk customers (>70% probability), immediate retention action is recommended</p>
</div>
""", unsafe_allow_html=True)