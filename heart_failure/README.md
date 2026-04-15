# ❤️ Heart Disease Classification App

A Machine Learning web application built with **Streamlit** to predict the likelihood of heart disease based on patient medical data.

---

## 🚀 Overview

This project provides an interactive interface where users can input medical attributes and receive a prediction indicating whether a patient is at **high risk** or **low risk** of heart disease.

The model is trained using classical machine learning techniques and deployed as a simple, user-friendly web app.

---

## 🧠 Features

* 📊 User-friendly interface with Streamlit
* 🧾 Input form for patient health data
* 🤖 Pre-trained ML model for prediction
* 📈 Displays prediction result (High Risk / Low Risk)
* 🔍 Shows prediction confidence (if available)
* 📦 Clean pipeline (preprocessing + model)

---

## 🗂️ Project Structure

```
├── app.py                         # Streamlit application
├── Best_model_pipeline.pkl        # Trained ML pipeline (model + preprocessing)
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/M-Waleed1/ML-Models.git
cd ML-Models
```

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 🧪 Input Features

The model uses the following medical attributes:

| Feature        | Description                                 |
| -------------- | ------------------------------------------- |
| Age            | Patient age                                 |
| Sex            | Male / Female                               |
| ChestPainType  | Type of chest pain (ATA, NAP, ASY, TA)      |
| RestingBP      | Resting blood pressure                      |
| Cholesterol    | Serum cholesterol (mg/dL)                   |
| FastingBS      | Fasting blood sugar (>120 mg/dL: 1, else 0) |
| RestingECG     | ECG results (Normal, ST, LVH)               |
| MaxHR          | Maximum heart rate                          |
| ExerciseAngina | Exercise-induced angina (Yes/No)            |
| Oldpeak        | ST depression value                         |
| ST_Slope       | Slope of ST segment (Up, Flat, Down)        |

---

## 🤖 Model Details

* Algorithm: (e.g., Random Forest / Logistic Regression / etc.)

* Preprocessing:
  
  * Handling categorical variables
  * Scaling numerical features

* Pipeline: Built using `scikit-learn Pipeline`

---

## 📊 Performance

* Accuracy: ~85%

* Evaluation Metrics:
  
  * Precision
  * Recall
  * F1-score

---

## 📸 App Preview

*(You can add screenshots here later)*

---

## 🌐 Deployment

You can deploy this app using:

* Streamlit Community Cloud
* Render
* Hugging Face Spaces

---

## 👨‍💻 Author

**Mohammed Waleed**

* 🎯 Data Analyst / ML Enthusiast
* 🔗 [GitHub](https://github.com/M-Waleed1/ML-Models)
* 💼 [LinkedIn](https://www.linkedin.com/in/mohammed-waleed-533931375/)

---

## 📌 Notes

* Ensure `Best_model_pipeline.pkl` exists in the root directory
* The model must be trained using the same feature names used in the app
* If prediction fails, verify preprocessing consistency

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!

---
