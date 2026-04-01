import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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
if "best_model_name" not in st.session_state:
    st.session_state["best_model_name"] = None

# --- Data Upload Page ---
if page == 'Data Upload':
    st.header('📁 Upload Customer Data')
    st.markdown("""
    ### Steps:
    1. Upload your customer dataset
    2. Select the target column (churn indicator)
    3. Train multiple models to predict customer churn
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
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
                st.metric("Columns", df.shape[1])
            with col2:
                st.metric("Missing Values", df.isnull().sum().sum())
                st.metric("Duplicates", df.duplicated().sum())
            with col3:
                st.metric("Data Types", len(df.dtypes.unique()))
            
            target = st.selectbox('🎯 Select the Churn Target Column', df.columns)
            st.session_state['target'] = target
            
            # Show target distribution
            st.subheader("🎯 Target Distribution")
            target_counts = df[target].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(target_counts)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', 
                       colors=['#2ecc71', '#e74c3c'])
                ax.set_title('Churn Distribution')
                st.pyplot(fig)
            
            if st.button('🚀 Train All Models', type='primary'):
                with st.spinner('Training models... This may take a moment...'):
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
                    
                    # Enhanced Models for churn prediction
                    models = {
                        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
                        'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5, weights='distance'),
                        'Support Vector Machine (SVM)': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
                        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
                        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
                        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                        'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
                    }
                    
                    scores = {}
                    best_model = None
                    best_score = 0
                    best_model_name = ""
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["📈 Model Performance", "📊 Comparison Chart", "🎯 Best Model Details"])
                    
                    with tab1:
                        st.subheader("📈 Model Performance Comparison")
                        
                        # Create columns for model cards
                        cols = st.columns(2)
                        model_idx = 0
                        
                        for name, model in models.items():
                            with cols[model_idx % 2]:
                                with st.spinner(f'Training {name}...'):
                                    # Update progress
                                    status_text.text(f'Training {name}...')
                                    progress = (model_idx + 1) / len(models)
                                    progress_bar.progress(progress)
                                    
                                    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
                                    pipe.fit(X_train, y_train)
                                    y_pred = pipe.predict(X_test)
                                    acc = accuracy_score(y_test, y_pred)
                                    scores[name] = acc
                                    
                                    # Create model card
                                    with st.container():
                                        st.markdown(f"### {name}")
                                        if acc >= 0.85:
                                            st.success(f"🎯 Accuracy: {acc:.3f}")
                                            st.markdown("⭐⭐⭐⭐⭐ Excellent")
                                        elif acc >= 0.75:
                                            st.info(f"📊 Accuracy: {acc:.3f}")
                                            st.markdown("⭐⭐⭐⭐ Good")
                                        elif acc >= 0.65:
                                            st.warning(f"⚠️ Accuracy: {acc:.3f}")
                                            st.markdown("⭐⭐⭐ Fair")
                                        else:
                                            st.error(f"❌ Accuracy: {acc:.3f}")
                                            st.markdown("⭐⭐ Needs Improvement")
                                        
                                        # Add brief explanation
                                        if name == 'K-Nearest Neighbors (KNN)':
                                            st.caption("Based on similar customer patterns")
                                        elif name == 'Support Vector Machine (SVM)':
                                            st.caption("Finds optimal boundary between churners and non-churners")
                                        elif name == 'Logistic Regression':
                                            st.caption("Simple and interpretable baseline model")
                                        elif 'Forest' in name:
                                            st.caption("Ensemble of decision trees for robust predictions")
                                        else:
                                            st.caption("Advanced boosting algorithm")
                                    
                                    if acc > best_score:
                                        best_score = acc
                                        best_model = pipe
                                        best_model_name = name
                                    
                                    model_idx += 1
                        
                        progress_bar.progress(1.0)
                        status_text.text('Training complete!')
                        
                        st.session_state['model'] = best_model
                        st.session_state['preprocessor'] = preprocessor
                        st.session_state['best_model_name'] = best_model_name
                        
                        st.success(f"🏆 **Best Model: {best_model_name}** with {best_score:.3f} accuracy")
                    
                    with tab2:
                        st.subheader("📊 Model Performance Comparison Chart")
                        
                        # Create comparison dataframe
                        comparison_df = pd.DataFrame(list(scores.items()), columns=['Model', 'Accuracy'])
                        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.barh(comparison_df['Model'], comparison_df['Accuracy'], 
                                      color=['#2ecc71' if i == 0 else '#3498db' for i in range(len(comparison_df))])
                        ax.set_xlabel('Accuracy Score')
                        ax.set_title('Model Performance Comparison')
                        ax.set_xlim(0, 1)
                        
                        # Add value labels
                        for i, (bar, val) in enumerate(zip(bars, comparison_df['Accuracy'])):
                            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                                   f'{val:.3f}', va='center')
                        
                        st.pyplot(fig)
                        
                        # Show top 3 models
                        st.subheader("🏆 Top 3 Models")
                        top3 = comparison_df.head(3)
                        for idx, row in top3.iterrows():
                            if idx == 0:
                                st.success(f"🥇 **1st Place: {row['Model']}** - {row['Accuracy']:.3f}")
                            elif idx == 1:
                                st.info(f"🥈 **2nd Place: {row['Model']}** - {row['Accuracy']:.3f}")
                            else:
                                st.info(f"🥉 **3rd Place: {row['Model']}** - {row['Accuracy']:.3f}")
                    
                    with tab3:
                        st.subheader(f"🎯 Best Model: {best_model_name}")
                        
                        # Train best model on full data
                        final_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', best_model.named_steps['classifier'])])
                        final_pipeline.fit(X_train, y_train)
                        y_pred_best = final_pipeline.predict(X_test)
                        
                        # Confusion Matrix
                        cm = confusion_matrix(y_test, y_pred_best)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Confusion Matrix")
                            fig, ax = plt.subplots(figsize=(6, 5))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax, 
                                       xticklabels=['No Churn', 'Churn'],
                                       yticklabels=['No Churn', 'Churn'])
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            ax.set_title(f'{best_model_name} - Confusion Matrix')
                            st.pyplot(fig)
                        
                        with col2:
                            st.markdown("#### Classification Report")
                            report = classification_report(y_test, y_pred_best, 
                                                          target_names=['No Churn', 'Churn'],
                                                          output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df.round(3))
                        
                        # Feature Importance (if applicable)
                        if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
                            st.markdown("#### Feature Importance")
                            # Get feature names after preprocessing
                            preprocessor.fit(X_train)
                            feature_names = []
                            for name, trans, cols in preprocessor.transformers_:
                                if name == 'num':
                                    feature_names.extend(cols)
                                elif name == 'cat':
                                    encoder = trans.named_steps['encoder']
                                    feature_names.extend(encoder.get_feature_names_out(cols))
                            
                            importances = best_model.named_steps['classifier'].feature_importances_
                            importance_df = pd.DataFrame({
                                'Feature': feature_names[:len(importances)],
                                'Importance': importances
                            }).sort_values('Importance', ascending=False).head(10)
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.barh(importance_df['Feature'], importance_df['Importance'], color='#3498db')
                            ax.set_xlabel('Importance')
                            ax.set_title('Top 10 Most Important Features')
                            st.pyplot(fig)
                        
                        # Model Recommendations
                        st.markdown("#### 💡 Recommendations Based on Best Model")
                        if best_model_name == 'Random Forest':
                            st.info("""
                            **Random Forest Insights:**
                            - Excellent for handling complex interactions
                            - Provides feature importance for business decisions
                            - Robust against overfitting with 200 trees
                            """)
                        elif best_model_name == 'XGBoost':
                            st.info("""
                            **XGBoost Insights:**
                            - Best for imbalanced datasets
                            - Handles missing values automatically
                            - Provides early stopping to prevent overfitting
                            """)
                        elif best_model_name == 'CatBoost':
                            st.info("""
                            **CatBoost Insights:**
                            - Excellent for categorical features
                            - No need for manual encoding
                            - Handles categorical variables automatically
                            """)
                        elif best_model_name == 'K-Nearest Neighbors (KNN)':
                            st.info("""
                            **KNN Insights:**
                            - Good for finding similar customer segments
                            - Works well with normalized data
                            - Useful for customer segmentation analysis
                            """)
                        elif best_model_name == 'Support Vector Machine (SVM)':
                            st.info("""
                            **SVM Insights:**
                            - Excellent for finding complex boundaries
                            - Works well with high-dimensional data
                            - Robust against outliers
                            """)
                        else:
                            st.info("""
                            **Model Insights:**
                            - This model provides good baseline performance
                            - Consider ensemble methods for better accuracy
                            - Review feature importance for business insights
                            """)
                        
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
        
        # Create gradient bar
        for i in range(10):
            if i < 3:
                color = '#2ecc71'  # Green
            elif i < 7:
                color = '#f39c12'  # Yellow/Orange
            else:
                color = '#e74c3c'  # Red
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
            **🚨 Immediate Action Required:**
            - Offer retention discount or special promotion
            - Schedule customer satisfaction call
            - Investigate recent service issues
            - Provide personalized support
            - Consider VIP treatment program
            """)
        elif prediction_proba[1] > 0.4:
            st.warning("""
            **⚠️ Proactive Retention Needed:**
            - Send personalized engagement email
            - Offer loyalty program benefits
            - Monitor usage patterns closely
            - Schedule check-in call next month
            - Send satisfaction survey
            """)
        else:
            st.success("""
            **✅ Maintain Current Strategy:**
            - Continue regular engagement
            - Send satisfaction surveys quarterly
            - Keep nurturing the relationship
            - Consider upsell opportunities
            - Reward loyalty with small perks
            """)
