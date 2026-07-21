import streamlit as st
import pickle
import pandas as pd

with open('logistic_titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Titanic Survival Prediction — Logistic Regression")
st.write("Enter passenger details to predict survival likelihood.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
has_cabin = st.selectbox("Has Cabin Recorded?", ["Yes", "No"])
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

if st.button("Predict Survival"):
    sex_val = 1 if sex == "female" else 0
    has_cabin_val = 1 if has_cabin == "Yes" else 0
    embarked_q = 1 if embarked == "Q" else 0
    embarked_s = 1 if embarked == "S" else 0

    input_df = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_val],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'HasCabin': [has_cabin_val],
        'Embarked_Q': [embarked_q],
        'Embarked_S': [embarked_s]
    })

    # Ensure column order matches training data
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Predicted: Survived (Probability: {probability:.2%})")
    else:
        st.error(f"Predicted: Did Not Survive (Probability of survival: {probability:.2%})")