import streamlit as st
import pickle
import pandas as pd

with open('svr_diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Diabetes Progression Prediction — SVR (Linear Kernel)")
st.write("Enter patient measurements to predict disease progression score.")

age = st.number_input("Age (standardized)", value=0.0, format="%.4f")
sex = st.number_input("Sex (standardized)", value=0.0, format="%.4f")
bmi = st.number_input("BMI (standardized)", value=0.0, format="%.4f")
bp = st.number_input("Blood Pressure (standardized)", value=0.0, format="%.4f")
s1 = st.number_input("S1 (standardized)", value=0.0, format="%.4f")
s2 = st.number_input("S2 (standardized)", value=0.0, format="%.4f")
s3 = st.number_input("S3 (standardized)", value=0.0, format="%.4f")
s4 = st.number_input("S4 (standardized)", value=0.0, format="%.4f")
s5 = st.number_input("S5 (standardized)", value=0.0, format="%.4f")
s6 = st.number_input("S6 (standardized)", value=0.0, format="%.4f")

if st.button("Predict Progression Score"):
    input_df = pd.DataFrame({
        'age': [age], 'sex': [sex], 'bmi': [bmi], 'bp': [bp],
        's1': [s1], 's2': [s2], 's3': [s3], 's4': [s4], 's5': [s5], 's6': [s6]
    })
    
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Progression Score: {prediction[0]:.2f}")