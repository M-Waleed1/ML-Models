import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib

st.set_page_config('Price Prediction')
st.title('Motorcycle Price Prediction')

pages = ['Info', 'Test']

st.sidebar.title('Navigation')
page = st.sidebar.radio('Navigation', pages)
if page == 'Info':
    st.markdown("""
    ### 🏍️ Motorcycle Price Prediction System
    
    Welcome! This application is a machine learning–powered tool designed to estimate the market price of motorcycles based on their key features.
    
    Using a trained model, the system analyzes attributes such as brand, manufacturing year, mileage, ownership status, and more to deliver fast and reliable price predictions.
    
    ---
    
    ### 🔍 What you can do
    - Enter motorcycle details  
    - Get instant price predictions  
    - Explore how different features influence pricing  
    
    ---
    
    ### 💡 Why use this app?
    Whether you're buying or selling a motorcycle, this tool helps you make smarter decisions with data-driven insights—so you can price competitively or avoid overpaying.
    
    ---
    """)
            
if page == 'Test':
    st.title('🔮 Motorcycle Price Prediction')

    # Load model
    try:
        model = joblib.load('best_pipeline.pkl') 
        st.success("Loaded pretrained model ✅")
    except:
        st.error("❌ No trained model found. Please train model first.")
        st.stop()

    # ===== Extract categories from trained model =====
    try:
        ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
        name_categories = ohe.categories_[0]  # assuming 'name' is first categorical column
    except:
        name_categories = ["Unknown"]

    # ===== USER INPUT =====
    name = st.selectbox("Motorcycle Name", name_categories)
    year = st.number_input("Year", min_value=1990, max_value=2025, step=1)
    seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer'])
    owner = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner'])
    km_driven = st.number_input("KM Driven", min_value=0)
    ex_showroom_price = st.number_input("Ex-showroom Price", min_value=0.0)

    # Create DataFrame
    input_df = pd.DataFrame({
        'name': [name],
        'year': [year],
        'seller_type': [seller_type],
        'owner': [owner],
        'km_driven': [km_driven],
        'ex_showroom_price': [ex_showroom_price]
    })

    st.write("Input Data:")
    st.dataframe(input_df)

    # Prediction
    if st.button("Predict Price 💰"):
        try:
            prediction = model.predict(input_df)[0]
            st.metric(label="💰 Estimated Price", value=f"{prediction:.0f}")
            st.balloons()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
