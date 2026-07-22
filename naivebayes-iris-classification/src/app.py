import streamlit as st
import pickle
import pandas as pd

with open('gnb_iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

st.title("Iris Species Prediction — Gaussian Naive Bayes")
st.write("Enter flower measurements to predict the Iris species.")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

if st.button("Predict Species"):
    input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                             columns=feature_names)
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Species: {species_map[prediction]}")