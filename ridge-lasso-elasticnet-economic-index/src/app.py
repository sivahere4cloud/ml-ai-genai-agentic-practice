import streamlit as st
import pickle
import pandas as pd

with open('regularized_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Economic Index Prediction — Ridge Regression (CV-Tuned)")
st.write("Enter interest rate and unemployment rate to predict the index price.")

interest_rate = st.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1
)

unemployment_rate = st.number_input(
    "Unemployment Rate (%)",
    min_value=0.0,
    max_value=15.0,
    value=5.5,
    step=0.1
)

if st.button("Predict Index Price"):
    input_df = pd.DataFrame({
        'interest_rate': [interest_rate],
        'unemployment_rate': [unemployment_rate]
    })
    
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Index Price: {prediction[0]:,.2f}")