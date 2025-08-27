import streamlit as st
import joblib
import pandas as pd

model = joblib.load("../models/final_model.pkl")

st.title("🫀 Heart Disease Prediction")

age = st.number_input("Age", 20, 100, 50)
chol = st.number_input("Cholesterol", 100, 500, 200)
thalch = st.number_input("Max Heart Rate", 50, 250, 150)
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
ca = st.number_input("Major Vessels (0-3)", 0, 3, 0)

if st.button("Predict"):
    df = pd.DataFrame([[age, chol, thalch, oldpeak, ca]],
                      columns=["age","chol","thalch","oldpeak","ca"])
    pred = model.predict(df)[0]
    st.success(f"Predicted Heart Disease Stage: {pred}")

