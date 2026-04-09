# app_streamlit_input.py
import streamlit as st
import requests

st.title("Salary Prediction (Single Input)")
job_titles = ['AI Engineer','Data Analyst','Frontend Developer','Business Analyst',
              'Product Manager','Backend Developer','Machine Learning Engineer',
              'DevOps Engineer','Software Engineer','Cybersecurity Analyst',
              'Data Scientist','Cloud Engineer']

locations = ['India','Australia','Singapore','Canada','Sweden','USA',
             'Netherlands','Remote','Germany','UK']

industries = ['Healthcare','Telecom','Media','Retail','Manufacturing','Education',                         'Finance','Technology','Consulting','Government']
# 1️⃣ User Inputs
job_title = st.selectbox("Job Title", job_titles)
experience_years = st.number_input("Years of Experience", min_value=0, max_value=50, value=1)
education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
skills_count = st.number_input("Skills Count", min_value=0, max_value=50, value=1)
industry = st.selectbox("Industry", industries)
company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
location = st.selectbox("Location", locations)
remote_work = st.selectbox("Remote Work", ["Yes", "No"])
certifications = st.number_input("Certifications Count", min_value=0, max_value=50, value=0)

# 2️⃣ Feature Engineering
exp_skills = experience_years * skills_count

if experience_years < 2:
    exp_level = 'junior'
elif experience_years < 5:
    exp_level = 'mid'
else:
    exp_level = 'senior'

cert_per_year = certifications / (experience_years + 1)
skills_cert = skills_count * certifications

# 3️⃣ Prepare data for API
data = {
    "job_title": job_title,
    "experience_years": experience_years,
    "education_level": education_level,
    "skills_count": skills_count,
    "industry": industry,
    "company_size": company_size,
    "location": location,
    "remote_work": remote_work,
    "certifications": certifications,
    "exp_skills": exp_skills,
    "exp_level": exp_level,
    "cert_per_year": cert_per_year,
    "skills_cert": skills_cert
}

# 4️⃣ Send to FastAPI
if st.button("Predict Salary"):
    api_url = "http://127.0.0.1:8000/predict"  # endpoint in FastAPI
    response = requests.post(api_url, json=data)
    if response.status_code == 200:
        st.success(f"Predicted Salary: ${response.json()['prediction']:.2f}")
    else:
        st.error("Error calling API")