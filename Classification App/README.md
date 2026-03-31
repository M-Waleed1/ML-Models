ML Classification App

This is an interactive Streamlit web app for building and evaluating classification machine learning models with your own dataset.

It includes preprocessing, multiple classifiers, model selection, confusion matrix visualization, and a downloadable trained pipeline.

------------------------------------------------------------

FEATURES:

1. Upload Your Dataset – CSV, XLS, or XLSX formats.
2. Choose Target Column – specify which column to predict.
3. Data Cleaning – remove duplicates, check and impute missing values.
4. Preprocessing Pipeline – handles numeric scaling and categorical encoding.
5. Model Training – trains multiple classifiers:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - CatBoost
6. Best Model Selection – automatically chooses the best model by accuracy.
7. Visualization – confusion matrix heatmap for model evaluation.
8. Download Pipeline – export the trained pipeline (final_pipeline.pkl) for future use.

------------------------------------------------------------

INSTALLATION:

1. Clone this repository:
   git clone <your-repo-url>
   cd <repo-folder>

2. Install requirements:
   pip install -r requirements.txt

------------------------------------------------------------

RUN LOCALLY:

Run the Streamlit app with:
   streamlit run app.py

Follow the instructions on the Upload and Model pages.