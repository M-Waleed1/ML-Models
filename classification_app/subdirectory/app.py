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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title='Customer Churn Prediction', layout='wide')
st.title('📊 Customer Churn Prediction App')

# Sidebar page selection
page = st.sidebar.radio('Go to', ['Data Upload', 'Predict Churn'])

# Initialize session state
if "df" not in st.session_state:
    st.session_state["df"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "preprocessor" not in st.session_state:
    st.session_state["preprocessor"] = None

# --- Data Upload Page ---
if page == 'Data Upload':
    st.header('📁 Upload Customer Data')
    st.markdown("""
    ### Steps:
    1. Upload your customer dataset
    2. Select the target column (churn indicator)
    3. Train the model to predict customer churn
    """)
    
    ext = st.selectbox('File Type', ['csv', 'xlsx', 'xls'])
    file = st.file_uploader('Upload File', type=['csv', 'xlsx', 'xls'])
    
    if file is not None:
        try:
            df = pd.read_csv(file) if ext == 'csv' else pd.read_excel(file)
            st.session_state['df'] = df.copy()
            
            st.subheader('📋 Data Preview')
            st.dataframe(df.head())
            
            st.subheader('📊 Data Information')
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Rows:** {df.shape[0]}")
                st.write(f"**Columns:** {df.shape[1]}")
            with col2:
                st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
                st.write(f"**Duplicates:** {df.duplicated().sum()}")
            
            target = st.selectbox('🎯 Select the Churn Target Column', df.columns)
            st.session_state['target'] = target
            
            if st.button('🚀 Train Churn Prediction Model'):
                with st.spinner('Training model...'):
                    X = df.drop(columns=[target])
                    y = df[target]
                    
                    # Split dataset
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Identify column types
                    num_cols = X.select_dtypes(include=np.number).columns
                    cat_cols = X.select_dtypes(exclude=np.number).columns
                    
                    # Preprocessing pipelines
                    num_trans = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')), 
                        ('scaler', RobustScaler(quantile_range=(0.2, 0.8)))
                    ])
                    
                    cat_trans = Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
                    ])
                    
                    preprocessor = ColumnTransformer([
                        ('num', num_trans, num_cols),
                        ('cat', cat_trans, cat_cols)
                    ])
                    
                    # Models for churn prediction
                    models = {
                        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
                        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                        'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
                    }
                    
                    scores = {}
                    best_model = None
                    best_score = 0
                    best_model_name = ""
                    
                    st.subheader("📈 Model Performance")
                    
                    for name, model in models.items():
                        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
                        pipe.fit(X_train, y_train)
                        y_pred = pipe.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        scores[name] = acc
                        
                        if acc > best_score:
                            best_score = acc
                            best_model = pipe
                            best_model_name = name
                        
                        # Display accuracy with color coding
                        if acc >= 0.8:
                            st.success(f"✅ {name}: {acc:.3f}")
                        elif acc >= 0.7:
                            st.info(f"📊 {name}: {acc:.3f}")
                        else:
                            st.warning(f"⚠️ {name}: {acc:.3f}")
                    
                    # Save best model
                    st.session_state['model'] = best_model
                    st.session_state['preprocessor'] = preprocessor
                    
                    st.success(f"🏆 Best Model: {best_model_name} with {best_score:.3f} accuracy")
                    
                    # Confusion Matrix
                    y_pred_best = best_model.predict(X_test)
                    cm = confusion_matrix(y_test, y_pred_best)
                    
                    st.subheader("📊 Confusion Matrix (Best Model)")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax, 
                                xticklabels=['No Churn', 'Churn'],
                                yticklabels=['No Churn', 'Churn'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Churn Prediction Confusion Matrix')
                    st.pyplot(fig)
                    
                    # Classification Report
                    st.subheader("📋 Detailed Classification Report")
                    report = classification_report(y_test, y_pred_best, 
                                                  target_names=['No Churn', 'Churn'],
                                                  output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(3))
                    
        except Exception as e:
            st.error(f'Error: {e}')
    else:
        st.info('📌 Please upload a customer dataset to begin.')

# --- Predict Churn Page ---
if page == 'Predict Churn':
    st.header('🔮 Predict Customer Churn')
    st.markdown("""
    ### Enter customer information to predict if they will churn
    Fill in the details below to get a churn prediction
    """)
    
    if st.session_state['df'] is None or st.session_state['model'] is None:
        st.warning('⚠️ Please upload data and train the model first from the "Data Upload" page')
        st.stop()
    
    # Get feature names from training data
    df = st.session_state['df'].drop(columns=[st.session_state['target']])
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns
    
    st.subheader("📝 Customer Information")
    
    # Create input fields for numeric features
    numeric_inputs = {}
    col1, col2 = st.columns(2)
    
    for i, col in enumerate(num_cols):
        with col1 if i % 2 == 0 else col2:
            numeric_inputs[col] = st.number_input(
                f"{col}",
                value=float(df[col].median()),
                help=f"Range: {df[col].min():.2f} to {df[col].max():.2f}"
            )
    
    # Create input fields for categorical features
    categorical_inputs = {}
    for col in cat_cols:
        unique_vals = df[col].dropna().unique().tolist()
        categorical_inputs[col] = st.selectbox(
            f"{col}",
            options=unique_vals,
            help=f"Choose from {len(unique_vals)} categories"
        )
    
    # Predict button
    if st.button('🎯 Predict Churn Risk', type='primary'):
        # Create input dataframe
        input_data = {}
        input_data.update(numeric_inputs)
        input_data.update(categorical_inputs)
        
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = st.session_state['model'].predict(input_df)[0]
        prediction_proba = st.session_state['model'].predict_proba(input_df)[0]
        
        # Display results
        st.subheader("📊 Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ **HIGH CHURN RISK**")
                st.markdown("This customer is likely to churn")
            else:
                st.success("✅ **LOW CHURN RISK**")
                st.markdown("This customer is likely to stay")
        
        with col2:
            st.metric("Churn Probability", f"{prediction_proba[1]:.1%}")
        
        with col3:
            st.metric("Retention Probability", f"{prediction_proba[0]:.1%}")
        
        # Add visual gauge
        st.subheader("Risk Meter")
        risk_level = prediction_proba[1]
        
        fig, ax = plt.subplots(figsize=(10, 2))
        colors = ['green', 'yellow', 'red']
        
        # Create gradient bar
        for i in range(10):
            color = 'green' if i < 3 else 'yellow' if i < 7 else 'red'
            ax.barh(0, 0.1, left=i*0.1, color=color, edgecolor='white', height=0.5)
        
        # Add marker
        marker_pos = risk_level
        ax.scatter(marker_pos, 0, color='black', s=100, zorder=5, marker='v')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_yticks([])
        ax.set_xlabel('Churn Probability')
        ax.set_title('Churn Risk Assessment')
        
        st.pyplot(fig)
        
        # Recommendations based on risk level
        st.subheader("💡 Recommendations")
        if prediction_proba[1] > 0.7:
            st.error("""
            **Immediate Action Required:**
            - Offer retention discount or special promotion
            - Schedule customer satisfaction call
            - Investigate recent service issues
            """)
        elif prediction_proba[1] > 0.4:
            st.warning("""
            **Proactive Retention Needed:**
            - Send personalized engagement email
            - Offer loyalty program benefits
            - Monitor usage patterns closely
            """)
        else:
            st.success("""
            **Maintain Current Strategy:**
            - Continue regular engagement
            - Send satisfaction surveys
            - Keep nurturing the relationship
            """)
