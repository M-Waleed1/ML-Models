### 📱 Telegram Spam Detection System

An end-to-end Machine Learning system that classifies Telegram messages as Spam or Ham (legitimate) using Natural Language Processing and supervised learning models. The project includes a full ML pipeline, model comparison, and an interactive Streamlit web app.

### 🚀 Project Overview

This system leverages NLP techniques and classical machine learning algorithms to automatically detect spam messages with high accuracy. It is trained on labeled Telegram message data and evaluates multiple models to select the best-performing one.

### ✨ Key Features
**🔍 Real-time message classification (Spam / Ham)**
**📊 Model comparison (Logistic Regression, SVM, Naive Bayes)**
**🧠 Automatic best model selection**
**⚡ TF-IDF feature extraction with n-grams (1–2)**
**📈 Performance evaluation (Accuracy, F1-score, Confusion Matrix)**
**🌐 Interactive Streamlit web interface**
**📁 Batch prediction via CSV upload**
**💾 Model persistence using Joblib**
**🧠 Machine Learning Pipeline**

#### 1. Data Preprocessing
Lowercasing text
Removing duplicates
Handling missing values
Tokenization & Lemmatization (spaCy-based)

#### 2. Feature Engineering
TF-IDF Vectorization
N-gram range: (1,2)
Max features tuning for performance

#### 3. Model Training
The following models are trained and compared:

Logistic Regression
Support Vector Machine (LinearSVC)
Naive Bayes (BernoulliNB)

### 📊 Model Performance
Model	Accuracy	Notes
SVM (LinearSVC)	~95%	Best performing model
Logistic Regression	~92–93%	Strong baseline
Naive Bayes	~83–85%	Weak for TF-IDF continuous features

### 🏆 Best Model
Model: Linear Support Vector Classifier
Accuracy: ~95%
Strength: Excellent performance on high-dimensional sparse text data

### 🌐 Web Application

#### Built using:
Streamlit

#### Features:
✍️ Single message prediction
📂 Bulk CSV prediction
📊 Visual result feedback
⚡ Instant inference

### ⚙️ Installation

1. Clone the repository
```bash
git clone https://github.com/your-repo/spam-detector.git
cd spam-detector
```

2. Install dependencies
`pip install -r requirements.txt`

3. Download spaCy model
`python -m spacy download en_core_web_sm`

### 🚀 How to Run

#### Step 1: Train the model
python model.py

#### Step 2: Launch web app
streamlit run app.py

### 🧪 Example Predictions

#### 🟢 Ham Messages
“Hey, are we still meeting tomorrow?”
“Don’t forget the meeting at 3 PM”
“Thanks for your help!”

#### 🔴 Spam Messages
“Congratulations! You won $1,000, click here”
“Your bank account is locked, verify now”
“Free iPhone giveaway, limited offer!”

### 📁 Dataset Requirements
CSV format
Columns:
text → message content
text_type → spam / ham
Minimum 1000 samples recommended
Balanced classes preferred

### 🛠️ Tech Stack
Python
Pandas
NumPy
scikit-learn
spaCy
Streamlit
Matplotlib
Seaborn

### 📌 What I Learned
Building full ML pipelines end-to-end
Text preprocessing for NLP tasks
Model comparison and evaluation strategies
Deploying ML models using Streamlit
Real-world feature engineering with TF-IDF

### 🚀 Future Improvements
Add BERT / Transformer-based models
Improve recall for spam detection
Add confidence score visualization
Deploy on cloud (Render / HuggingFace Spaces)
Add authentication for API usage

### 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss improvements or features.

### 📧 Contact

[LinkedIn](https://www.linkedin.com/in/mohammed-waleed-533931375/)