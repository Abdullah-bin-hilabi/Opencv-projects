import numpy as np
import pandas
import joblib
import streamlit as st

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Diabetes Prediction App")

pregnancies = st.number_input("Number of Pregnancies",min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure Value", min_value=0, max_value=200, value=80)
skin_thickness = st.number_input("Skin Thickness Value", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=100)
bmi = st.number_input("BMI Value", min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function Value", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age of the Person", min_value=0, max_value=120, value=30)

if st.button("Predict"):
    input_data = np.array([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age
]])

    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    
    probability = model.predict_proba(input_data)[0][1]
    st.write(f"Diabetes Probability: {probability:.2f}")

    if probability >= 0.5:
        st.error("🚨 Likely Diabetic")
    else:
        st.success("✅ Likely Non-Diabetic")
