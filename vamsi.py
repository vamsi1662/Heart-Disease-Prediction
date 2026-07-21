import os
import pickle

import numpy as np
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Load model and scaler
model = pickle.load(open(os.path.join(BASE_DIR, "heart_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))


st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details below")

# Inputs
age = st.number_input("Age", 1, 120, 40)

sex = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex == "Male" else 0

cp = st.selectbox("Chest Pain Type", [0,1,2,3])

trestbps = st.number_input("Resting Blood Pressure", 80,250,120)

chol = st.number_input("Cholesterol",100,600,200)

fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl",[0,1])

restecg = st.selectbox("Rest ECG",[0,1,2])

thalach = st.number_input("Maximum Heart Rate",60,250,150)

exang = st.selectbox("Exercise Induced Angina",[0,1])

oldpeak = st.number_input("Old Peak",0.0,10.0,1.0)

slope = st.selectbox("Slope",[0,1,2])

ca = st.selectbox("Number of Major Vessels",[0,1,2,3,4])

thal = st.selectbox("Thal",[0,1,2,3])

if st.button("Predict"):

    features = np.array([[age, sex, cp, trestbps, chol,
                          fbs, restecg, thalach,
                          exang, oldpeak, slope,
                          ca, thal]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")