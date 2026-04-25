import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset.csv')
df.head()

(df['text_type'].value_counts()/len(df['text_type']))*100

print(f'Duplicates: {df.duplicated().sum()}')

df = df.drop_duplicates()

print(df.isnull().sum())

print(df.shape)

x = df['text']
y = df['text_type']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=42, test_size=0.2, stratify=y)

nlp = spacy.load('en_core_web_sm')

# def token(text):
#     doc = nlp(text)
#     tokens = []
#     for token in doc:
#         if not token.is_stop and not token.is_punct:
#             tokens.append(token.lemma_)
#     return tokens

# xtrain = xtrain.apply(token)
# xtest = xtest.apply(token)

tfidf = TfidfVectorizer(
    lowercase=True,
    max_features=5000,
    ngram_range=(1, 2)
)

models = {
    'Logistic': LogisticRegression(),
    'SVM': LinearSVC(),
    'NB': MultinomialNB()
}

best_score = 0
best_model_name = ''
best_model = None

for name, model in models.items():
    model = Pipeline([
        ('Tfidf', tfidf),
        ('clf', model)
    ])
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    scores = cross_val_score(model, xtrain, ytrain, scoring='accuracy')

    acc = accuracy_score(ytest, y_pred)
    cr = classification_report(ytest, y_pred)
    cm = confusion_matrix(ytest, y_pred)
    print(f'{name} model Training')
    print(f'Score: {scores}')
    print(f'Mean Accuracy: {scores.mean()}')
    print(f'Accuracy: {acc:0.4f}')
    print(f'Report:\n{cr}')
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predict')
    plt.tight_layout()
    plt.show()
    print('='*60)

    if acc > best_score:
        best_score = acc
        best_model_name = name
        best_model = model

print(f'Best Model is {best_model_name} with score {best_score:0.4f}')

import joblib 

joblib.dump(best_model, 'pipeline.pkl')