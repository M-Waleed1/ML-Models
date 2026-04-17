# Import Libraries

# Basics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
# Metrics
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
# Save Model
import joblib


df = pd.read_csv('Data/customer .csv')
df.head()

df.shape

df.columns

df.info()

df.describe()

# Cleaning

dup = df.duplicated().sum()
print(f'Duplicates: {dup}')

null = df.isnull().sum()
print(f'Null:\n{null[null > 0]}')

# Fill Null Values
df.fillna(df['SeniorCitizen'].mode()[0], inplace=True)
df.fillna(df['tenure'].median(), inplace=True)

print(f'Null:\n{df.isnull().sum()}')

df['Churn'] = df['Churn'].replace(to_replace='Yes', value=1)
df['Churn'] = df['Churn'].replace(to_replace='No', value=0)

# EDA

citizen_churn = df.groupby(['Churn', 'SeniorCitizen']).size()
citizen_churn.plot(kind='bar')

print(df['Churn'].value_counts(normalize=True))

# 1. Churn rate by categorical features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
categoricals = ['Contract', 'PaymentMethod', 'InternetService', 
                'SeniorCitizen', 'Partner', 'PaperlessBilling']

for i, col in enumerate(categoricals):
    pd.crosstab(df[col], df['Churn'], normalize='index').plot(
        kind='bar', ax=axes[i//3, i%3], stacked=True
    )
    axes[i//3, i%3].set_title(f'Churn by {col}')

plt.tight_layout()
plt.show()

# 2. Numerical features distribution by Churn
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(num_cols):
    sns.boxplot(x='Churn', y=col, data=df, ax=axes[i])
    axes[i].set_title(f'{col} by Churn')
plt.show()

# 3. Correlation heatmap (numeric only)
numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlations')
plt.show()

# Tenure vs Monthly Charges colored by Churn
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Churn', alpha=0.6)
plt.title('Tenure vs Monthly Charges (Churn highlighted)')
plt.show()

# Churn rate by tenure group
churn_by_tenure = df.groupby('tenure')['Churn'].value_counts(normalize=True).unstack()
churn_by_tenure[1].plot(kind='bar', figsize=(10, 5))
plt.title('Churn Rate by Tenure Group')
plt.ylabel('Churn Rate')
plt.show()

# Feature Engineering

df['tenure_group'] = pd.cut(df['tenure'], 
                            bins=[0, 12, 24, 48, 72], 
                            labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr'])

# Average monthly spend per tenure
df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)

# High-value customer flag
df['high_value'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)

# Senior citizen with no partner
df['senior_no_partner'] = ((df['SeniorCitizen'] == 1) & (df['Partner'] == 'No')).astype(int)

# Multiple services flag
services = ['PhoneService', 'InternetService', 'StreamingTV', 'StreamingMovies']
df['num_services'] = df[services].apply(lambda x: (x != 'No').sum(), axis=1)

# Risk score
df['risk_score'] = (
    (df['Contract'] == 'Month-to-month').astype(int) * 3 +
    (df['PaperlessBilling'] == 'Yes').astype(int) * 2 +
    (df['PaymentMethod'] == 'Electronic check').astype(int) * 2
)

del df['customerID']

# Split Data

x = df.drop('Churn', axis=1)
y = df['Churn']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

le = LabelEncoder()
ytrain_encoded = le.fit_transform(ytrain)
ytest_encoded = le.transform(ytest)

# Preprocessor

num_cols = x.select_dtypes(include=np.number).columns
cat_cols = x.select_dtypes(exclude=np.number).columns

num_trans = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='median')),
    ('Scaler', RobustScaler(quantile_range=(20,80)))
])

cat_trans = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='most_frequent')),
    ('Encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('Num', num_trans, num_cols),
    ('Cat', cat_trans, cat_cols)
])

# Modeling

# Encode target variable
le = LabelEncoder()
ytrain_encoded = le.fit_transform(ytrain)
ytest_encoded = le.transform(ytest)

# Models (now compatible with encoded labels)
models = {
    'Dummy': DummyClassifier(strategy='most_frequent'),
    'Logistic': LogisticRegression(random_state=42, max_iter=1000),
    'RF': RandomForestClassifier(random_state=42, n_estimators=100),
    'XGB': XGBClassifier(random_state=42, eval_metric='logloss'),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
    'LightGBM': LGBMClassifier(random_state=42, n_estimators=100, verbose=-1)
}

results = []
best_score = 0
best_model = None
best_name = ''

for name, model in models.items():
    print(f"Training: {name}")
    
    # Create pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Cross Validation (using encoded labels)
    try:
        cv = cross_val_score(model_pipeline, xtrain, ytrain_encoded, cv=5, scoring='accuracy')
        print(f"Cross-validation scores: {cv}")
        print(f"CV Mean Accuracy: {cv.mean():.4f} (+/- {cv.std():.4f})")
        
        # Train Model
        model_pipeline.fit(xtrain, ytrain_encoded)
        
        # Predict Values
        ypred = model_pipeline.predict(xtest)
        ypred_prob = model_pipeline.predict_proba(xtest)[:, 1] if hasattr(model_pipeline, 'predict_proba') else None
        
        # Evaluation (using encoded labels)
        acc = accuracy_score(ytest_encoded, ypred)
        report = classification_report(ytest_encoded, ypred, 
                                      target_names=le.classes_,  # Use original names for display
                                      output_dict=True)
        
        print(f"\nTest Set Performance:")
        print(f"Accuracy: {acc:.4f}")
        
        if ypred_prob is not None:
            roc_auc = roc_auc_score(ytest_encoded, ypred_prob)
            print(f"ROC-AUC: {roc_auc:.4f}")
        else:
            roc_auc = None
            print("ROC-AUC: Not supported")
        
        # Confusion Matrix
        cm = confusion_matrix(ytest_encoded, ypred)
        
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=le.classes_,
                    yticklabels=le.classes_)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()
        
        # Save Results
        results.append({
            'Model': name,
            'CV_Mean': cv.mean(),
            'CV_Std': cv.std(),
            'Test_Accuracy': acc,
            'ROC_AUC': roc_auc,
            f'Precision_{le.classes_[0]}': report[le.classes_[0]]['precision'],
            f'Recall_{le.classes_[0]}': report[le.classes_[0]]['recall'],
            f'Precision_{le.classes_[1]}': report[le.classes_[1]]['precision'],
            f'Recall_{le.classes_[1]}': report[le.classes_[1]]['recall']
        })
        
        # Define Best Model
        if acc > best_score:
            best_score = acc
            best_model = model_pipeline
            best_name = name
            
    except Exception as e:
        print(f"Error with {name}: {str(e)}")
        continue

# Create results dataframe
results_df = pd.DataFrame(results)
print("FINAL RESULTS SUMMARY")
print(results_df.to_string())
print(f"Best Model: {best_name} with Accuracy: {best_score:.4f}")

# Feature Importance

# Get the trained logistic regression model from your pipeline
logistic_model = best_model.named_steps['classifier']

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get coefficients (importance weights)
coefficients = logistic_model.coef_[0]

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
})

# Sort by absolute coefficient (most important first)
importance_df = importance_df.sort_values('abs_coefficient', ascending=False)

print("TOP 20 MOST IMPORTANT FEATURES FOR CHURN PREDICTION")
print(importance_df.head(20).to_string(index=False))

# Top 15 features
top_features = importance_df.head(15)

plt.figure(figsize=(10, 8))
colors = ['red' if x < 0 else 'green' for x in top_features['coefficient']]
bars = plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Coefficient Value (Impact on Churn)')
plt.title('Top 15 Features Impacting Churn Prediction')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.legend([plt.Rectangle((0,0),1,1, facecolor='green'),
            plt.Rectangle((0,0),1,1, facecolor='red')],
           ['Reduces Churn', 'Increases Churn'],
           loc='lower right')
plt.tight_layout()
plt.show()

joblib.dump(best_model, 'Final_Pipeline.pkl')