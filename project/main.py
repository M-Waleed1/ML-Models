from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('model.pkl')  # الكاتبوت/موديلك

class SalaryInput(BaseModel):
    job_title: str
    experience_years: int
    education_level: str
    skills_count: int
    industry: str
    company_size: str
    location: str
    remote_work: str
    certifications: int
    exp_skills: int
    exp_level: str
    cert_per_year: float
    skills_cert: int

@app.post('/predict')
async def prediction(data: SalaryInput):
    # تحويل ال input لقيم DataFrame
    df = pd.DataFrame([data.dict()])

    # predict باستخدام pipeline جاهز
    pred = model.predict(df)

    return {'prediction': float(pred[0])}