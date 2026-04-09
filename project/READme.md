# Salary Prediction System

A complete end-to-end machine learning system for predicting salaries based on job-related features. The system includes a trained ML model, a FastAPI backend for serving predictions, and a Streamlit frontend for user interaction.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model Training](#model-training)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Input Features](#input-features)
- [Feature Engineering](#feature-engineering)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)

## 🎯 Overview

This project predicts salaries for various tech and business roles using machine learning. Users input their job details through an intuitive Streamlit interface, which sends the data to a FastAPI backend. The backend uses a trained CatBoost regression model to generate salary predictions.

## ✨ Features

- **Interactive UI**: User-friendly Streamlit interface for input collection
- **RESTful API**: FastAPI backend for prediction serving
- **Multiple ML Models**: Trained and compared 6 different regression models
- **Feature Engineering**: Automatic creation of derived features for better predictions
- **Real-time Predictions**: Instant salary estimates based on user inputs

## 📦 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🔧 Installation

1. **Clone or download the project files**

2. **Install required packages**:

```bash
pip install streamlit fastapi uvicorn joblib pandas numpy scikit-learn xgboost catboost requests matplotlib seaborn
```

### 🧠 Model Training
The training script performs the following steps:

Data Loading: Reads salary prediction dataset

Feature Engineering:

exp_skills = experience_years × skills_count

exp_level = junior/mid/senior based on experience

cert_per_year = certifications / (experience_years + 1)

skills_cert = skills_count × certifications

---

Model Comparison: Trains and evaluates:

Linear Regression

Lasso Regression

Ridge Regression

Random Forest

XGBoost

CatBoost (selected as final model)

Save Model: Exports the best model as model.pkl

```bash
# Run training (after preparing data)
python train_model.py
```

### 🚀 Running the Application
Step 1: Start the FastAPI Backend

```bash
uvicorn fast_api:app --reload --port 8000
```

The API will be available at: http://127.0.0.1:8000

Step 2: Start the Streamlit Frontend
Open a new terminal and run:

```bash
streamlit run app_streamlit_input.py
```

The web app will open automatically at: http://localhost:8501

### 📡 API Endpoints
GET /
Check if the API is running.

Response:

json
{
  "Message": "API is Working Now"
}
POST /predict
Get salary prediction based on input features.

Request Body:

json
{
  "job_title": "Data Scientist",
  "experience_years": 5,
  "education_level": "Master",
  "skills_count": 8,
  "industry": "Technology",
  "company_size": "Large",
  "location": "USA",
  "remote_work": "Yes",
  "certifications": 3,
  "exp_skills": 40,
  "exp_level": "mid",
  "cert_per_year": 0.5,
  "skills_cert": 24
}
Response:

json
{
  "Prediction": 125000.00
}

### 📊 Input Features
Feature	Type	Options/Range
Job Title	Categorical	AI Engineer, Data Analyst, Frontend Developer, Business Analyst, Product Manager, Backend Developer, Machine Learning Engineer, DevOps Engineer, Software Engineer, Cybersecurity Analyst, Data Scientist, Cloud Engineer

Experience Years	Numeric	0-50 years

Education Level	Categorical	High School, Bachelor, Master, PhD

Skills Count	Numeric	0-50

Industry	Categorical	Healthcare, Telecom, Media, Retail, Manufacturing, Education, Finance, Technology, Consulting, Government

Company Size	Categorical	Small, Medium, Large

Location	Categorical	India, Australia, Singapore, Canada, Sweden, USA, Netherlands, Remote, Germany, UK

Remote Work	Categorical	Yes, No

Certifications	Numeric	0-50

### 🔬 Feature Engineering
The system automatically creates three derived features:

exp_skills: Interaction between experience and skills

Formula: experience_years × skills_count

exp_level: Experience level categorization

junior: < 2 years

mid: 2-5 years

senior: > 5 years

cert_per_year: Certifications per year of experience

Formula: certifications / (experience_years + 1)

skills_cert: Interaction between skills and certifications

Formula: skills_count × certifications

📁 Project Structure

project/
│
├── app_streamlit_input.py    # Streamlit frontend application
├── fast_api.py               # FastAPI backend service
├── train_model.py            # Model training script
├── model.pkl                 # Trained CatBoost model
│
├── Data/
│   └── job_salary_prediction_dataset.csv  # Training dataset
│
└── README.md                 # This file

### 🛠️ Technologies Used
Component	Technology
Frontend	Streamlit
Backend	FastAPI, Uvicorn
ML Models	Scikit-learn, XGBoost, CatBoost
Data Processing	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Model Serialization	Joblib

### 📈 Model Performance (CatBoost)
The final CatBoost model achieves:

R² Score: (check your training output)

MAE: (check your training output)

RMSE: (check your training output)

### 🤝 Contributing
Feel free to fork this project and enhance it with:

Additional features

More sophisticated models

Docker containerization

Cloud deployment (AWS/GCP/Azure)

### 📧 Support
For issues or questions, please open an issue in the repository.

Happy Predicting! 🎯
