# 📰 Fake News Detection System

A Machine Learning-powered web application that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP). Built with **Python**, **scikit-learn**, and **Streamlit**, this project demonstrates an end-to-end workflow from data preprocessing to model deployment.

---

## 🚀 Overview

This project leverages **TF-IDF Vectorization** and multiple classification algorithms to detect fake news. The best-performing model is selected and deployed in an interactive **Streamlit web app**, allowing users to input news text and receive instant predictions.

---

## 🧠 Machine Learning Pipeline

The system follows a structured ML workflow:

### 1. Data Collection

- Dataset sourced from Kaggle:
  
  - Fake news articles
  
  - Real news articles

### 2. Data Preprocessing

- Removed duplicates and null values

- Dropped unnecessary columns (e.g., `date`)

- Normalized text (lowercasing, trimming spaces)

- Cleaned categorical values (`subject`)

- Feature engineering:
  
  - Combined `title`, `text`, and `subject` into one feature
  
  - Added `text_length`

### 3. Text Vectorization

- **TF-IDF Vectorizer**
  
  - `ngram_range=(1,2)`
  
  - `max_features=5000`
  
  - Optimized `min_df` and `max_df`

### 4. Models Used

- Logistic Regression

- Support Vector Machine (SVM)

- Multinomial Naive Bayes

- Dummy Classifier (baseline)

### 5. Model Evaluation

- Accuracy Score

- Cross-validation

- Classification Report

- Confusion Matrix

### 6. Model Selection

- Best-performing model selected based on accuracy

- Saved using `joblib` as `pipeline.pkl`

---

## 🖥️ Streamlit Application

An interactive UI that allows users to:

- ✍️ Enter custom news text

- 🔍 Predict whether it is Fake or Real

- 📊 View prediction confidence 

- 🔗 Access dataset and project links

### UI Sections:

- **Main Page** → Prediction interface

- **Sidebar**:
  
  - Overview
  
  - Dataset link
  
  - Contact (LinkedIn & GitHub)

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/M-Waleed1/ML-Models.git
cd ML-Models
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

## 📊 Example Usage

Input:

```
Breaking: Government announces new economic reforms...
```

Output:

```
✅ Real News
or
❌ Fake News
```

---

## 📈 Future Improvements

- 🔹 Hyperparameter tuning (GridSearch / RandomizedSearch)

- 🔹 Deep Learning models (LSTM / BERT)

- 🔹 Batch prediction via CSV upload

- 🔹 REST API deployment باستخدام FastAPI

- 🔹 Model explainability (SHAP / LIME)

---

## 🔗 Links

- 📊 Dataset:  
  https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection/data

- 💼 LinkedIn:  
  [Mohammed Waleed](https://www.linkedin.com/in/mohammed-waleed-533931375/)

- 💻 GitHub:  
  [GitHub - M-Waleed1/ML-Models: Machine Learning Models · GitHub](https://github.com/M-Waleed1/ML-Models)

---

## ❤️ Acknowledgments

- Kaggle for providing the dataset

- scikit-learn for ML tools

- Streamlit for deployment framework

---

## 📌 Author

**Mohammed Waleed**  
Aspiring Data Analyst | Machine Learning Enthusiast

---

## ⭐ If you found this project useful

Give it a star on GitHub and share it 🚀
