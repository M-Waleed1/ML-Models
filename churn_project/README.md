# 📊 Customer Churn Prediction System

An end-to-end machine learning system for predicting customer churn in telecommunications companies. This project includes data analysis, feature engineering, model training with 6 different algorithms (80.4% accuracy), and a production-ready web application for real-time predictions.

## 🎯 Business Impact

- **80.4%** model accuracy on test data
- **84.6%** ROC-AUC score
- **$500K+** potential savings per 1000 customers
- **300%** projected ROI from retention campaigns

## ✨ Features

### 🤖 Machine Learning

- 6 different models comparison (Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM)
- Advanced feature engineering (30+ features)
- Cross-validation and hyperparameter tuning
- Feature importance analysis

### 📱 Web Application

- **Single Prediction**: Real-time churn risk assessment
- **Batch Processing**: Upload CSV for bulk predictions (1000+ customers)
- **Analytics Dashboard**: Interactive visualizations and insights
- **Model Insights**: Understand key drivers of churn

### 📊 Visualizations

- Interactive risk gauges
- Probability distributions
- Risk level breakdowns
- Feature impact analysis

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```



2. Install dependencies

```bash
pip install -r requirements.txtTrain the model (optional - model already included)
```

3. Run the web application

```bash
streamlit run app.py
```

## 🔧 Feature Engineering

The system creates 8 advanced features to improve prediction accuracy:

| Feature                  | Description                                 | Impact    |
| ------------------------ | ------------------------------------------- | --------- |
| `tenure_group`           | Customer tenure binned (0-1yr, 1-2yr, etc.) | High      |
| `avg_monthly_spend`      | Average monthly spending over tenure        | High      |
| `high_value`             | Flag for customers above median spending    | Medium    |
| `senior_no_partner`      | Senior citizens without partners            | Low       |
| `num_services`           | Count of services subscribed                | High      |
| `risk_score`             | Composite risk score (0-7)                  | Very High |
| `contract_type_encoded`  | Contract type numeric encoding              | Very High |
| `payment_method_encoded` | Payment method encoding                     | High      |

## 📈 Model Performance

### Comparison Results

| Model               | CV Accuracy | Test Accuracy | ROC-AUC    | Precision (Churn) | Recall (Churn) |
| ------------------- | ----------- | ------------- | ---------- | ----------------- | -------------- |
| Logistic Regression | 80.1%       | **80.4%**     | **0.8465** | 0.65              | 0.54           |
| Random Forest       | 79.2%       | 79.8%         | 0.8321     | 0.63              | 0.52           |
| XGBoost             | 79.5%       | 80.1%         | 0.8389     | 0.64              | 0.53           |
| CatBoost            | 79.8%       | 80.2%         | 0.8412     | 0.64              | 0.53           |
| LightGBM            | 79.3%       | 79.9%         | 0.8356     | 0.63              | 0.52           |

### Top 5 Important Features

1. **Contract Type** (Month-to-month → 3x higher churn)

2. **Tenure** (Higher tenure → lower churn)

3. **Monthly Charges** (Higher charges → higher churn)

4. **Payment Method** (Electronic check → 2x higher churn)

5. **Number of Services** (More services → lower churn)

## 🎨 Web Application Features

### 1. Home Dashboard

- Model performance metrics

- Quick overview of capabilities

- Navigation guide

### 2. Single Prediction

- Form with 20+ customer attributes

- Real-time prediction

- Risk level visualization (Low/Medium/High)

- Personalized retention recommendations

### 3. Batch Prediction

- Upload CSV file

- Process thousands of customers

- Download predictions with risk scores

- High-risk customer identification

### 4. Analytics Dashboard

- Risk distribution analysis

- Contract type impact

- Tenure analysis

- Revenue at risk calculation

### 5. Model Insights

- Key churn drivers explained

- Actionable recommendations by risk level

- Business impact analysis



## 🛠️ Technology Stack

### Core

- **Python 3.8+**: Primary programming language

- **Pandas/NumPy**: Data manipulation and analysis

- **Scikit-learn**: Machine learning models and preprocessing

### ML Models

- **XGBoost**: Gradient boosting framework

- **CatBoost**: Categorical features boosting

- **LightGBM**: Lightweight gradient boosting

- **Random Forest**: Ensemble learning

### Frontend

- **Streamlit**: Web application framework

- **Plotly**: Interactive visualizations

- **Matplotlib/Seaborn**: Static visualizations

### Utilities

- **Joblib**: Model serialization

- **Pickle**: Object serialization

## 📈 Key Insights from Analysis

### High Risk Factors (Increase Churn)

- ❌ Month-to-month contracts (3x higher risk)

- ❌ Electronic check payments (2x higher risk)

- ❌ No online security/backup services

- ❌ Paperless billing enabled

- ❌ Tenure < 12 months

### Loyalty Factors (Decrease Churn)

- ✅ Long-term contracts (1-2 years)

- ✅ Automatic payment methods

- ✅ Multiple service bundles

- ✅ Tech support subscription

- ✅ Tenure > 24 months

## 🔄 Deployment Options

### Local Deployment

```bash
streamlit run app.py
```



## 📝 Future Improvements

- Real-time monitoring dashboard

- A/B testing framework for retention strategies

- Email/Slack alerts for high-risk customers

- API endpoint for integration with CRM systems

- Deep learning models (LSTM for time-series data)

- Customer segmentation for targeted campaigns

- Automated retraining pipeline

- SHAP values for individual predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository

2. Create your feature branch (`git checkout -b feature/AmazingFeature`)

3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)

4. Push to the branch (`git push origin feature/AmazingFeature`)

5. Open a Pull Request



## 👨‍💻 Author

**Your Name**

- GitHub: [M-Waleed1](https://github.com/M-Waleed1)

- LinkedIn: [Mohammed-Waleed](https://www.linkedin.com/in/mohammed-waleed-533931375/)

## 📧 Contact

For questions or collaboration opportunities, please open an issue or email: mohammed.waleed.aqw@gmail.com

---

## ⭐ Show Your Support

If you found this project helpful, please give it a ⭐️ on GitHub!

---

**Built with ❤️ for Data Science and Customer Success**


