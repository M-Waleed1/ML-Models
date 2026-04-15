import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title='Heart Disease Classification')

st.title('Heart Disease Classification')

pages = ['Info', 'Prediction']
page = st.sidebar.radio('Navigation', pages)

# =========================
# INFO PAGE
# =========================
if page == 'Info':
    st.markdown("""
    ## Welcome 👋  
    This is a Machine Learning web app to predict heart disease.

    ### How to use:
    - Go to **Prediction**
    - Enter your data
    - Click Predict

    ### Libraries used:
    - Pandas
    - NumPy
    - Scikit-learn

    ### Model Performance:
    ~85% Accuracy

    ---
    👨‍💻 Mohammed Waleed  
    [GitHub](https://github.com/M-Waleed1/ML-Models)  
    [LinkedIn](https://www.linkedin.com/in/mohammed-waleed-533931375/)
    """)

# =========================
# PREDICTION PAGE
# =========================
if page == 'Prediction':
    st.header('Enter Patient Data')

    # تحميل الموديل مرة واحدة (optimization)
    @st.cache_resource
    def load_model():
        return joblib.load('Best_model_pipeline.pkl')

    model = load_model()

    # Inputs
    age = st.number_input('Age', 1, 120, 25)
    sex = st.radio('Sex', ['Male', 'Female'])
    chestpain = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
    resting_bp = st.number_input('Resting Blood Pressure', 50, 250, 120)
    cholesterol = st.number_input('Cholesterol (mg/dL)', 50, 400, 180)
    fasting_bs = st.radio('Fasting Blood Sugar > 120 mg/dL', [0, 1])
    resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
    max_hr = st.number_input('Max Heart Rate', 60, 220, 150)
    exercise_angina = st.radio('Exercise Angina', ['Yes', 'No'])
    oldpeak = st.number_input('Oldpeak', 0.0, 10.0, 1.0, step=0.1)
    st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

    if st.button("Predict"):

        # DataFrame
        data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'ChestPainType': [chestpain],
            'RestingBP': [resting_bp],
            'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs],
            'RestingECG': [resting_ecg],
            'MaxHR': [max_hr],
            'ExerciseAngina': [exercise_angina],
            'Oldpeak': [oldpeak],
            'ST_Slope': [st_slope]
        })

        # Prediction
        prediction = model.predict(data)[0]

        # لو عندك predict_proba
        try:
            prob = model.predict_proba(data)[0][1]
        except:
            prob = None

        # Display Result
        st.subheader("Result:")

        if prediction == 1:
            st.error(f"⚠️ High Risk of Heart Disease")
        else:
            st.success(f"✅ Low Risk (Healthy)")

        if prob:
            st.write(f"Confidence: {prob:.2f}")

        # Show input data
        with st.expander("Show Input Data"):
            st.dataframe(data)