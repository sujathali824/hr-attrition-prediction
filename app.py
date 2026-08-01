import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and column order
model = joblib.load('attrition_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Employee Attrition Predictor")
st.write("Enter employee details to predict attrition risk.")

# --- Input fields ---
age = st.slider("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
job_satisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
overtime = st.selectbox("OverTime", ["Yes", "No"])
years_at_company = st.slider("Years at Company", 0, 40, 5)
distance_from_home = st.slider("Distance From Home (km)", 0, 30, 5)

# --- Build input row matching training columns ---
input_dict = dict.fromkeys(model_columns, 0)  # start all columns at 0
input_dict['Age'] = age
input_dict['MonthlyIncome'] = monthly_income
input_dict['JobSatisfaction'] = job_satisfaction
input_dict['YearsAtCompany'] = years_at_company
input_dict['DistanceFromHome'] = distance_from_home
if overtime == "Yes":
    input_dict['OverTime_Yes'] = 1

input_df = pd.DataFrame([input_dict])[model_columns]  # ensure correct column order

# --- Predict ---
if st.button("Predict Attrition Risk"):
    input_scaled = scaler.transform(input_df)
    proba = model.predict_proba(input_scaled)[0][1]
    threshold = 0.3
    prediction = "Yes - High Risk" if proba >= threshold else "No - Low Risk"

    st.subheader(f"Prediction: {prediction}")
    st.write(f"Attrition Probability: {proba:.2%}")