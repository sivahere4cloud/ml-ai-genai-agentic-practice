import streamlit as st
import pickle
import pandas as pd

with open('../src/homeprice_model.pkl', 'rb') as file:
    model = pickle.load(file)


st.title("Home Price Prediction — Simple Linear Regression")
st.write("Enter area (in square feet) to predict the estimated home price.")

area = st.number_input(
    "Area (sq ft)",
    min_value=500.0,
    max_value=10000.0,
    value=2600.0,
    step=100.0
)

if st.button("Predict Price"):
    input_data = pd.DataFrame({'area': [area]})
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: £{prediction[0]:,.2f}")