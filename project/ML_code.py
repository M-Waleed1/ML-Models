# Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv('Data/job_salary_prediction_dataset.csv')
df.head()

df.shape

df.columns

df.info()

df.describe()

# Clean

dup = df.duplicated().sum()
print(f'Duplicates: {dup}')

null = df.isnull().sum()
print(f'Null:\n{null[null > 0]}')

sns.histplot(df['salary'], kde=True)
plt.title('Salary Distribution')

# Feature Engineering

df['exp_skills'] = df['experience_years'] * df['skills_count']

def exp_level(x):
    if x < 2: return 'junior'
    elif x < 5: return 'mid'
    else: return 'senior'

df['exp_level'] = df['experience_years'].apply(exp_level)

df['cert_per_year'] = df['certifications'] / (df['experience_years'] + 1)

df['skills_cert'] = df['skills_count'] * df['certifications']

x = df.drop('salary', axis=1)
y = df['salary']

xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size= 0.3,
                                                random_state=24)

num_cols = x.select_dtypes(include=np.number).columns
cat_cols = x.select_dtypes(exclude=np.number).columns

num_trans = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='median')),
    ('Scaler', RobustScaler(quantile_range=(20, 80)))
])

cat_trans = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='most_frequent')),
    ('Encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('Num', num_trans, num_cols),
    ('Cat', cat_trans, cat_cols)
])

# linearRegression
linear = Pipeline(steps=[
    ('Preprocessor', preprocessor),
    ('Linear', LinearRegression())
])

linear.fit(xtrain, ytrain)
y_pred = linear.predict(xtest)

res = ytest - y_pred
print(f'R2: {r2_score(ytest, y_pred):.4f}')
print(f'MAE: {mean_absolute_error(ytest, y_pred):.4f}')
print(f'MSE: {mean_squared_error(ytest, y_pred):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(ytest, y_pred)):.4f}')

sns.histplot(res, kde=True)
plt.title('linear Resisduals')
plt.tight_layout()
plt.show()

# Lasso
lasso = Pipeline(steps=[
    ('Preprocessor', preprocessor),
    ('Lasso', Lasso())
])

lasso.fit(xtrain, ytrain)
y_pred = lasso.predict(xtest)

res = ytest - y_pred
print(f'R2: {r2_score(ytest, y_pred):.4f}')
print(f'MAE: {mean_absolute_error(ytest, y_pred):.4f}')
print(f'MSE: {mean_squared_error(ytest, y_pred):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(ytest, y_pred)):.4f}')

sns.histplot(res, kde=True)
plt.title('linear Resisduals')
plt.tight_layout()
plt.show()

# Ridge
ridge = Pipeline(steps=[
    ('Preprocessor', preprocessor),
    ('Ridge', Ridge())
])

ridge.fit(xtrain, ytrain)
y_pred = ridge.predict(xtest)

res = ytest - y_pred
print(f'R2: {r2_score(ytest, y_pred):.4f}')
print(f'MAE: {mean_absolute_error(ytest, y_pred):.4f}')
print(f'MSE: {mean_squared_error(ytest, y_pred):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(ytest, y_pred)):.4f}')

sns.histplot(res, kde=True)
plt.title('linear Resisduals')
plt.tight_layout()
plt.show()

# Random Forest
rf = Pipeline(steps=[
    ('Preprocessor', preprocessor),
    ('RF', RandomForestRegressor(random_state=42))
])

rf.fit(xtrain, ytrain)
y_pred = rf.predict(xtest)

res = ytest - y_pred
print(f'R2: {r2_score(ytest, y_pred):.4f}')
print(f'MAE: {mean_absolute_error(ytest, y_pred):.4f}')
print(f'MSE: {mean_squared_error(ytest, y_pred):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(ytest, y_pred)):.4f}')

sns.histplot(res, kde=True)
plt.title('linear Resisduals')
plt.tight_layout()
plt.show()

# XGBoost
xg = Pipeline(steps=[
    ('Preprocessor', preprocessor),
    ('XG', XGBRegressor(random_state=42))
])

xg.fit(xtrain, ytrain)
y_pred = xg.predict(xtest)

res = ytest - y_pred
print(f'R2: {r2_score(ytest, y_pred):.4f}')
print(f'MAE: {mean_absolute_error(ytest, y_pred):.4f}')
print(f'MSE: {mean_squared_error(ytest, y_pred):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(ytest, y_pred)):.4f}')

sns.histplot(res, kde=True)
plt.title('linear Resisduals')
plt.tight_layout()
plt.show()

# CatBoost
cat = Pipeline(steps=[
    ('Preprocessor', preprocessor),
    ('CAT', CatBoostRegressor(random_state=42, iterations=100))
])

cat.fit(xtrain, ytrain)
y_pred = cat.predict(xtest)

res = ytest - y_pred
print(f'R2: {r2_score(ytest, y_pred):.4f}')
print(f'MAE: {mean_absolute_error(ytest, y_pred):.4f}')
print(f'MSE: {mean_squared_error(ytest, y_pred):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(ytest, y_pred)):.4f}')

sns.histplot(res, kde=True)
plt.title('linear Resisduals')
plt.tight_layout()
plt.show()

import joblib

joblib.dump(cat, 'model.pkl')