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

st.set_page_config('Price Prediction')
st.title('Motorcycle Price Prediction')

pages = ['Info', 'Data', 'Model Training', 'Test']

st.sidebar.title('Navigation')
page = st.sidebar.radio('Navigation', pages)
if page == 'Info':
    st.markdown("""
    Welcome to the Motorcycle Price Prediction System — a data-driven tool designed to        estimate the market value of motorcycles based on their features.
    
    This application leverages a trained machine learning model to analyze key attributes     such as brand, engine capacity, mileage, year of manufacture, and more, providing you     with an accurate price prediction in seconds.
    
    🔍 What you can do:
    Input motorcycle specifications 
    Get instant price predictions
    Understand how different features impact pricing
    
    💡 Why use this app?
     Whether you're a buyer looking for a fair deal or a seller aiming to price your           motorcycle competitively, this tool helps you make informed decisions backed by data.
    """)

if page == 'Data':
    st.title('Data Uploader')
    st.markdown("""In this Site you will Upload the data and preview it""")
    st.markdown("""> **In case in Excel File ensure that the Columns name in first row**""")
    
    ext = st.selectbox('File type', ['CSV', 'xlsx', 'xls'])
    file = st.file_uploader('File Upload', type=['csv', 'xlsx', 'xls'])

    if file is not None:
        if ext == 'CSV':
            st.session_state['df'] = pd.read_csv(file)
            st.write(st.session_state['df'].head())
        if ext == 'xlsx' or ext == 'xls':
            st.session_state['df'] = pd.read_excel(file)
            st.write(st.session_state['df'].head())

        dup = st.session_state['df'].duplicated().sum()
        if dup == 0:
            st.success('No Duplicates')
        else:
            st.write(f'Numbe rof duplicates is: {dup}')
            if st.button('Drop Dupliactes') is True:
                st.session_state['df'] = st.session_state['df'].drop_duplicates()
                dupl = st.session_state['df'].duplicated().sum()
                st.write(f'Number of Duplicates is:  {dupl}')
                st.write('Now No Duplicates')


if page == 'Model Training':
    st.title('Model Training')

    if 'df' not in st.session_state:
        st.warning("Please upload data first")
    
    elif st.button('🚀 Train Model'):
        with st.spinner('Training model...'):
            X = st.session_state['df'].drop('selling_price', axis=1)
            y = st.session_state['df']['selling_price']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            num_cols = X.select_dtypes(include=np.number).columns
            cat_cols = X.select_dtypes(exclude=np.number).columns

            num_trans = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler(quantile_range=(20, 80)))
            ])

            cat_trans = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer([
                ('num', num_trans, num_cols),
                ('cat', cat_trans, cat_cols)
            ])

            models = {
                'Linear': LinearRegression(),
                'RandomForest': RandomForestRegressor(random_state=42),
                'XGBoost': XGBRegressor(random_state=42),
                'CatBoost': CatBoostRegressor(verbose=0, random_state=42)
            }

            scores = {}
            best_model = None
            best_score = -np.inf
            best_model_name = ""

            for name, model in models.items():
                pipe = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])

                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                scores[name] = {
                    'R2': r2,
                    'MAE': mae,
                    'MSE': mse
                }

                if r2 > best_score:
                    best_score = r2
                    st.session_state['best_model'] = pipe
                    best_model_name = name

            results_df = pd.DataFrame(scores).T
            st.subheader("📊 Model Performance")
            st.dataframe(results_df)

            st.success(f"🏆 Best Model: {best_model_name}")
            st.write(f"Best R2 Score: {best_score:.4f}")

if page == 'Test':
    st.title('🔮 Motorcycle Price Prediction')

    if 'df' not in st.session_state:
        st.warning("Please upload and train model first")
    
    elif 'best_model' not in st.session_state:
        st.warning("Please train the model first")
    
    else:
        df = st.session_state['df']
        model = st.session_state['best_model']

        st.subheader("Enter Motorcycle Details")

        # Inputs
        name = st.selectbox("Motorcycle Name", df['name'].unique())
        year = st.number_input("Year", min_value=1990, max_value=2025, value=2018)
        seller_type = st.selectbox("Seller Type", df['seller_type'].unique())
        owner = st.selectbox("Owner Type", df['owner'].unique())
        km_driven = st.number_input("KM Driven", min_value=0, value=10000)
        ex_showroom_price = st.number_input("Ex-Showroom Price", min_value=0, value=50000)

        # Create input dataframe
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
            prediction = model.predict(input_df)[0]
            st.success(f"Estimated Selling Price: {prediction:.2f}")
    
