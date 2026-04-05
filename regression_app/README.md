# 🏍️ Motorcycle Price Prediction App

A lightweight machine learning web app built with **Streamlit** that predicts motorcycle prices using a pre-trained model.

---

## 🚀 Project Overview

This application allows users to estimate the selling price of a motorcycle by entering key specifications.

Unlike traditional ML apps, this version is designed to be **simple, fast, and ready-to-use**, relying on a **pre-trained pipeline** without requiring data upload or model training.

---

## 🔮 Features

* 🎯 Predict motorcycle prices instantly
* 📦 Uses a pre-trained ML pipeline (`best_pipeline.pkl`)
* 🧠 Handles preprocessing automatically (imputation, scaling, encoding)
* 🎨 Clean and simple user interface
* ⚡ Fast predictions with no setup required

---

## 📊 Input Features

The model uses the following inputs:

* Motorcycle Name
* Year
* Seller Type
* Owner
* Kilometers Driven
* Ex-showroom Price

---

## 🧠 Machine Learning Details

The model is trained using a full pipeline that includes:

* **Numerical Processing**

  * Median imputation
  * Robust scaling

* **Categorical Processing**

  * Most frequent imputation
  * One-hot encoding (handles unseen values)

* **Models Tested**

  * Linear Regression
  * Random Forest
  * XGBoost
  * CatBoost

The best-performing model was selected based on evaluation metrics and saved for deployment.

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Pandas / NumPy
* Scikit-learn
* XGBoost / CatBoost
* Joblib

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Required File

Make sure the trained model file exists:

```bash
best_pipeline.pkl
```

---

## 💡 Use Case

This tool helps:

* Buyers estimate a fair price
* Sellers price their motorcycles competitively

---

## 👨‍💻 Author

**Mohammed Waleed**
Aspiring Data Analyst

---

⭐ If you find this project useful, feel free to star the repository!
