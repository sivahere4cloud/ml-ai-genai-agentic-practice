import streamlit as st
import pickle
import pandas as pd

with open('svc_iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

st.title("Iris Species Prediction — SVC (Linear Kernel)")
st.write("Enter flower measurements to predict the Iris species.")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

if st.button("Predict Species"):
    input_df = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })
    
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"Predicted Species: {species_map[prediction]}")