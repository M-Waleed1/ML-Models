import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title='ML Classification App', layout='wide')
st.title('Classification ML Model')

# Sidebar page selection
page = st.sidebar.radio('Go to', ['Upload', 'Model'])

# Initialize session state
if "df" not in st.session_state:
    st.session_state["df"] = None

# --- Upload Page ---
if page == 'Upload':
    st.header('Upload Page')
    st.markdown("""
    ### In this page you will do:
    1. Upload Data  
    2. Choose Target
    """)
    ext = st.selectbox('File Type', ['csv', 'xlsx', 'xls'])
    file = st.file_uploader('Upload File', type=['csv', 'xlsx', 'xls'])
    
    if file is not None:
        try:
            df = pd.read_csv(file) if ext == 'csv' else pd.read_excel(file)
            st.session_state['df'] = df.copy()
            target = st.selectbox('Choose the target Column', df.columns)
            st.session_state['target'] = target
            st.success('File Uploaded Successfully!')
            st.subheader('Preview')
            st.dataframe(df.head())
        except Exception as e:
            st.error(f'Error reading file: {e}')
    else:
        st.info('Please upload a file.')

# --- Model Page ---
if page == 'Model':
    st.header('Model Page')
    st.markdown("""
    ### This page will:
    1. Remove duplicates
    2. Fill missing values
    3. Encode & scale data
    4. Train multiple models
    5. Show best model & download pipeline
    """)

    if st.session_state['df'] is None:
        st.warning('Please upload a file from "Upload Page"')
        st.stop()

    df = st.session_state['df'].copy()

    st.subheader('Data Preview')
    st.dataframe(df.head())

    # Duplicates
    st.subheader('Duplicates')
    dup = df.duplicated().sum()
    if dup == 0:
        st.success('No Duplicates')
    else:
        st.write(f'Number of Duplicates: {dup}')
        if st.button('Remove Duplicates'):
            df = df.drop_duplicates()
            st.session_state['df'] = df
            st.success('Duplicates removed')
            st.dataframe(df.head())

    # Missing values
    st.subheader('Missing Values')
    null = df.isnull().sum()[df.isnull().sum() > 0]
    if null.empty:
        st.success('No missing values')
        st.dataframe(df.head())
    else:
        st.write('Columns with missing values')
        st.dataframe(null.reset_index().rename(columns={'index':'Column', 0:'Missing values'}))

    # Train pipeline and models
    if st.button('Pipeline & Train Models'):
        X = df.drop(columns=[st.session_state['target']])
        y = df[st.session_state['target']]

        # Split dataset (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Columns
        num_cols = X.select_dtypes(include=np.number).columns
        cat_cols = X.select_dtypes(exclude=np.number).columns

        # Preprocessing
        num_trans = Pipeline([
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', RobustScaler(quantile_range=(0.2,0.8)))
        ])
        cat_trans = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer([
            ('num', num_trans, num_cols),
            ('cat', cat_trans, cat_cols)
        ])

        # Models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
        }

        scores = {}
        for name, model in models.items():
            pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            scores[name] = acc
            st.write(f"{name} Accuracy: {acc:.3f}")

        # Best model
        best_model_name = max(scores, key=scores.get)
        best_model = models[best_model_name]
        st.success(f"Best Model: {best_model_name} (Accuracy: {scores[best_model_name]:.3f})")

        # Final pipeline
        final_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', best_model)])
        final_pipeline.fit(X, y)

        # Confusion matrix
        y_pred = final_pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix (Best Model)")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        # Save pipeline and download
        joblib.dump(final_pipeline, 'final_pipeline.pkl')
        with open('final_pipeline.pkl', 'rb') as f:
            st.download_button('Download Trained Pipeline', f, file_name='final_pipeline.pkl')
            
