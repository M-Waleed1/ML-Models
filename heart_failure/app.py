# Import Libraries

# Basics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
# Metrics
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
# Save Model
import joblib

# Load Data

df = pd.read_csv('heart.csv')

# Understanding Data

df.head()

dup = df.duplicated().sum()
print(f'Duplicates: {dup}')

null = df.isnull().sum()
null

df.describe()

df.info()

df.shape

df.columns

### From the data:
# 1. No Duplicates
# 2. No Null values
# 3. Need Scaling
# 4. Need Encoding

# EDA

plt.subplot(1, 2, 1)
df['HeartDisease'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Heart Disease Distribution')
plt.xlabel('HeartDisease (0= No, 1= Yes )')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
df['HeartDisease'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Percentage')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

sns.boxplot(x='HeartDisease', y='Age', data=df, ax=axes[0,0])
axes[0,0].set_title('Heart Disease VS Age')

pd.crosstab(df['Sex'], df['HeartDisease']).plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Heart Disease VS Sex')
axes[0,1].set_xlabel('Sex')

pd.crosstab(df['FastingBS'], df['HeartDisease']).plot(kind='bar', ax=axes[0,2])
axes[0,2].set_title('Heart Disease VS FastingBS')

sns.boxplot(x='HeartDisease', y='MaxHR', data=df, ax=axes[1,0])
axes[1,0].set_title('Heart Disease VS MaxHR')

sns.boxplot(x='HeartDisease', y='Oldpeak', data=df, ax=axes[1,1])
axes[1,1].set_title('Oldpeak VS HeartDisease')

sns.boxplot(x='HeartDisease', y='Cholesterol', data=df, ax=axes[1,2])
axes[1,2].set_title('Heart Disease VS Cholesterol')

plt.tight_layout()
plt.show()

df_encoded = df.copy()
categorical_cols = df.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

plt.figure(figsize=(12, 8))
correlation = df_encoded.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

corr_target = correlation['HeartDisease'].sort_values(ascending=False)
print("Most Important Features")
print(corr_target)

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df['AgeGroup'] = pd.cut(
            df['Age'],
            bins=[0, 40, 50, 60, 70, 120],
            labels=['<40', '40-50', '50-60', '60-70', '70+']
        )

        df['Age_Ratio'] = df['Age'] / 100

        df['HR_Deficit'] = (220 - df['Age']) - df['MaxHR']
        df['HR_Percentage'] = df['MaxHR'] / (220 - df['Age'])

        df['HR_Risk'] = (df['MaxHR'] < (220 - df['Age'])).astype(int)

        df['HR_Category'] = pd.cut(
            df['MaxHR'],
            bins=[0, 100, 120, 140, 200],
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        df['PulsePressure'] = df['RestingBP'] - 80
        df['BP_Risk'] = (df['RestingBP'] > 140).astype(int)

        df['Cholesterol_Cleaned'] = df['Cholesterol'].replace(
            0,
            df['Cholesterol'].median()
        )

        df['Chol_Age_Ratio'] = df['Cholesterol_Cleaned'] / df['Age']

        df['Chol_Category'] = pd.cut(
            df['Cholesterol_Cleaned'],
            bins=[0, 200, 240, 1000],
            labels=['Normal', 'Borderline', 'High']
        )

        df['FastingBS_Age'] = df['FastingBS'] * df['Age']

        df['High_Oldpeak'] = (df['Oldpeak'] > 1.5).astype(int)

        df['Oldpeak_Category'] = pd.cut(
            df['Oldpeak'],
            bins=[-1, 0, 1, 2, 10],
            labels=['None', 'Mild', 'Moderate', 'Severe']
        )

        risk_factors = (
            (df['Age'] > 55).astype(int) +
            (df['RestingBP'] > 140).astype(int) +
            (df['Cholesterol_Cleaned'] > 240).astype(int) +
            (df['FastingBS'] == 1).astype(int) +
            (df['MaxHR'] < 100).astype(int) +
            (df['Oldpeak'] > 1).astype(int) +
            (df['ExerciseAngina'] == 1).astype(int)
        )

        df['RiskScore'] = risk_factors
        df['HighRisk'] = (df['RiskScore'] >= 3).astype(int)

        df['Pain_Exercise_Risk'] = (
            df['ChestPainType'].isin(['ASY', 'NAP']) &
            (df['ExerciseAngina'] == 1)
        ).astype(int)

        df['ST_Slope_Risk'] = df['ST_Slope'].isin(['Down', 'Flat']).astype(int)

        df['RestingECG_Risk'] = df['RestingECG'].map({
            'Normal': 0,
            'ST': 1,
            'LVH': 1
        })

        return df

feature_eng = FeatureEngineering()

x = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=42, test_size=0.2)

num_cols = x.select_dtypes(include=np.number).columns
cat_cols = x.select_dtypes(exclude=np.number).columns

num_trans = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='median')),
    ('Scaler',RobustScaler(quantile_range=(20,80)))
])
cat_trans = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='most_frequent')),
    ('Encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('Num', num_trans, num_cols),
    ('Cat', cat_trans, cat_cols)
])

# 

models = {
    'Dummy': DummyClassifier(strategy='most_frequent'),
    'Logistic': LogisticRegression(random_state=42, max_iter=1000),
    'RF': RandomForestClassifier(random_state=42, n_estimators=100),
    'XGB': XGBClassifier(random_state=42, eval_metric='logloss'),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
    'SVM': SVC(random_state=42, probability=True)  # probability=True عشان predict_proba
}

results = []
best_score = 0
best_model = None
best_name = ''

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print('='*50)
    
    pipe = Pipeline(steps=[
        ('feature_engineering', feature_eng),
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    cv_scores = cross_val_score(pipe, xtrain, ytrain, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"CV Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    pipe.fit(xtrain, ytrain)
    
    ypred = pipe.predict(xtest)
    ypred_prob = pipe.predict_proba(xtest)[:, 1] if hasattr(pipe, 'predict_proba') else None
    
    acc = accuracy_score(ytest, ypred)
    report = classification_report(ytest, ypred, output_dict=True)
    
    print(f"Test Set Performance:")
    print(f"Accuracy: {acc:.4f}")
    
    if ypred_prob is not None:
        roc_auc = roc_auc_score(ytest, ypred_prob)
        print(f"ROC-AUC: {roc_auc:.4f}")
    else:
        roc_auc = None
        print("ROC-AUC: Not supported")
    
    cm = confusion_matrix(ytest, ypred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()
    
    results.append({
        'Model': name,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std(),
        'Test_Accuracy': acc,
        'ROC_AUC': roc_auc,
        'Precision_0': report['0']['precision'],
        'Recall_0': report['0']['recall'],
        'Precision_1': report['1']['precision'],
        'Recall_1': report['1']['recall']
    })
    
    if acc > best_score:
        best_score = acc
        best_model = pipe
        best_name = name

print(f"🏆 BEST MODEL: {best_name} with Test Accuracy: {best_score:.4f}")

results_df = pd.DataFrame(results)
print("All Models Comparison:")
print(results_df.to_string(index=False))

plt.figure(figsize=(12, 6))
models_names = results_df['Model']
test_acc = results_df['Test_Accuracy']
cv_mean = results_df['CV_Mean']
cv_std = results_df['CV_Std']

x = np.arange(len(models_names))
width = 0.35

plt.bar(x - width/2, test_acc, width, label='Test Accuracy', color='skyblue')
plt.bar(x + width/2, cv_mean, width, label='CV Mean', color='lightcoral', yerr=cv_std, capsize=5)

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.xticks(x, models_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Fetaure Importance

best_catboost_model = best_model.named_steps['classifier']

feature_importance = best_catboost_model.get_feature_importance()
feature_names = preprocessor.get_feature_names_out()

feature_names_clean = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

importance_df = pd.DataFrame({
    'feature': feature_names_clean,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("FEATURE IMPORTANCE - CatBoost Model")

print(importance_df.head(15))

plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, 15))
plt.barh(importance_df['feature'][:15], importance_df['importance'][:15], color=colors)
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 15 Most Important Features - CatBoost', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# Tuning

X_train_transformed = preprocessor.fit_transform(xtrain)
X_test_transformed = preprocessor.transform(xtest)

random_search = RandomizedSearchCV(
    Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', CatBoostClassifier(random_state=42, verbose=0))
    ]),
    param_distributions={
        'classifier__depth': [4, 6, 8, 10],
        'classifier__learning_rate': [0.03, 0.05, 0.1, 0.15],
        'classifier__iterations': [100, 200, 300],
        'classifier__l2_leaf_reg': [1, 3, 5, 7],
        'classifier__border_count': [128, 254]
    },
    n_iter=30,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(xtrain, ytrain)  

joblib.dump(random_search.best_estimator_, 'Best_model_pipeline.pkl')
print("Saved complete pipeline with preprocessing!")