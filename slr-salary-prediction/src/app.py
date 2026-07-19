import streamlit as st
import pickle
import numpy as np

#Load the trained model

with open('../src/salary_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Salary Prediction")
st.write("Predict Salary Based on Years of Experience")


years_experience = st.number_input(
    "Years of Exp",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.5
)


if st.button("Predict Salary"):
    input_data = np.array([[years_experience]])
    prediction = model.predict(input_data)
    st.success(f"Predicited Salary :: £ {prediction[0]:,.2f}")