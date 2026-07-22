import streamlit as st
import pickle
import pandas as pd

with open('knn_classifier_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Breast Cancer Prediction — KNN Classifier")
st.write("Enter cell measurements to predict malignant vs benign.")
st.info("This demo uses a subset of key features for simplicity — full model uses all 30 measurements.")

# Using a few representative features for a manageable UI
mean_radius = st.number_input("Mean Radius", min_value=0.0, value=14.0)
mean_texture = st.number_input("Mean Texture", min_value=0.0, value=19.0)
mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, value=90.0)
mean_area = st.number_input("Mean Area", min_value=0.0, value=650.0)

if st.button("Predict"):
    # Build full feature row using dataset means for non-input features (demo simplification)
    input_row = X.mean().to_frame().T
    input_row['mean radius'] = mean_radius
    input_row['mean texture'] = mean_texture
    input_row['mean perimeter'] = mean_perimeter
    input_row['mean area'] = mean_area

    input_scaled = scaler.transform(input_row)
    prediction = model.predict(input_scaled)[0]
    
    result = "Benign" if prediction == 1 else "Malignant"
    st.success(f"Prediction: {result}")