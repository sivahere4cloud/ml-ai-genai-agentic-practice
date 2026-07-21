import streamlit as st
import pickle
import numpy as np

# Load model and polynomial transformer
with open('poly_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('poly_transformer.pkl', 'rb') as file:
    poly = pickle.load(file)

st.title("Position Salary Prediction — Polynomial Regression")
st.write("Enter a position level (1-10) to predict the expected salary.")

level = st.number_input(
    "Position Level",
    min_value=1.0,
    max_value=10.0,
    value=5.0,
    step=0.5
)

if st.button("Predict Salary"):
    input_data = np.array([[level]])
    
    # Apply the SAME polynomial transformation used during training
    input_poly = poly.transform(input_data)
    
    prediction = model.predict(input_poly)
    st.success(f"Predicted Salary: £{prediction[0]:,.2f}")