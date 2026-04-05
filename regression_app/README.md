# 🏍️ Motorcycle Price Prediction App

A machine learning-powered web application built with **Streamlit** to predict the selling price of motorcycles based on their specifications.

---

## 🚀 Project Overview

This application allows users to:

* Upload their own motorcycle dataset
* Clean and preprocess the data
* Train multiple machine learning models
* Compare model performance using evaluation metrics
* Predict motorcycle prices based on input features

The goal is to provide a **data-driven tool** for buyers and sellers to estimate fair motorcycle prices.

---

## 📊 Features

* 📁 Upload CSV or Excel datasets
* 🧹 Automatic data cleaning (duplicates handling)
* ⚙️ Preprocessing pipeline:

  * Missing value imputation
  * Feature scaling
  * One-hot encoding
* 🤖 Multiple models:

  * Linear Regression
  * Random Forest
  * XGBoost
  * CatBoost
* 📈 Model evaluation:

  * R² Score
  * MAE (Mean Absolute Error)
  * MSE (Mean Squared Error)
* 🏆 Automatic best model selection
* 🔮 Real-time price prediction

---

## 🗂️ Dataset Structure

The dataset should include the following columns:

* `name` → Motorcycle model
* `selling_price` → Target variable
* `year` → Manufacturing year
* `seller_type` → Dealer / Individual
* `owner` → Ownership type
* `km_driven` → Distance driven
* `ex_showroom_price` → Original price

---

## 🛠️ Tech Stack

* Python 🐍
* Streamlit
* Pandas & NumPy
* Scikit-learn
* XGBoost
* CatBoost
* Matplotlib & Seaborn

---

## ▶️ How to Run the App

### 1. Clone the repository

```bash
git clone https://github.com/M-Waleed1/motorcycle-price-prediction.git
cd motorcycle-price-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📌 Usage

1. Go to **Data Page** → Upload your dataset
2. Navigate to **Model Training** → Train models
3. Check performance metrics
4. Go to **Test Page** → Enter motorcycle details
5. Get predicted selling price 💰

---

## 📷 App Screenshots (Optional)

*Add screenshots here after deployment*

---

## 🌐 Deployment

You can deploy this app easily using:

* Streamlit Cloud
* Heroku
* Render

---

## 📈 Future Improvements

* Add feature importance visualization
* Hyperparameter tuning
* Save & load trained models
* Add user authentication
* Improve UI/UX

---

## 👨‍💻 Author

**Mohammed Waleed**
Data Analyst | Aspiring Data Scientist

* 📧 [mohammed.waleed1121@gmail.com](mailto:mohammed.waleed1121@gmail.com)
* 🔗 https://www.linkedin.com/in/mohammed-waleed-533931375/

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
