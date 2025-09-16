import streamlit as st
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('car_price_model.pkl')

st.title("🚗 Car Price Prediction App")

# User Inputs
symboling = st.number_input("Symboling (Risk Factor)", min_value=-3, max_value=3, value=0)
fueltype_gas = st.selectbox("Fuel Type", [0, 1])  # 0: Diesel, 1: Gas
aspiration_turbo = st.selectbox("Aspiration Turbo?", [0, 1])  # 0: No, 1: Yes
wheelbase = st.number_input("Wheelbase (inches)", min_value=80.0, max_value=120.0, value=95.0)
curbweight = st.number_input("Curb Weight (lbs)", min_value=1500, max_value=5000, value=2500)
enginesize = st.number_input("Engine Size (cc)", min_value=60, max_value=500, value=150)
horsepower = st.number_input("Horsepower", min_value=40, max_value=400, value=100)
citympg = st.number_input("City MPG", min_value=10, max_value=60, value=25)
highwaympg = st.number_input("Highway MPG", min_value=10, max_value=60, value=30)
boreratio = st.number_input("Bore Ratio", min_value=2.0, max_value=4.0, value=3.0)
stroke = st.number_input("Stroke", min_value=2.0, max_value=5.0, value=3.5)
compressionratio = st.number_input("Compression Ratio", min_value=7.0, max_value=12.0, value=9.0)
peakrpm = st.number_input("Peak RPM", min_value=3000, max_value=7000, value=5000)

if st.button("Predict Price"):
    # Prepare input array in correct order
    input_features = np.array([[symboling, fueltype_gas, aspiration_turbo, wheelbase, curbweight,
                                enginesize, horsepower, citympg, highwaympg, boreratio,
                                stroke, compressionratio, peakrpm]])
    
    # Predict
    price_prediction = model.predict(input_features)[0]
    
    st.success(f"💰 Estimated Car Price: ${price_prediction:.2f}")
