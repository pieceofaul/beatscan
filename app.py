import streamlit as st
import numpy as np
import pickle

# Load model
try:
    model = pickle.load(open('heart_model.pkl', 'rb'))
except Exception as e:
    model = None
    st.warning("⚠️ Model not loaded. Check the heart_model.pkl file.")

# Title
st.title("❤️ Heart Disease Prediction App")

# Subtitle or instructions
st.write("### Enter your health data:")

# Input fields
age = st.number_input('Age', min_value=1, max_value=120, value=50)
sex = st.selectbox('Sex', ['Male', 'Female'])

cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120)
chol = st.number_input('Serum Cholesterol (chol)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
restecg = st.selectbox('Resting ECG (restecg)', [0, 1, 2])
thalach = st.number_input('Max Heart Rate Achieved (thalach)', min_value=60, max_value=250, value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])

# Convert input to model format
sex = 1 if sex == 'Male' else 0

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang]])

# Prediction
if st.button('Predict'):
    if model:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error('⚠️ You might be at risk of heart disease.')
        else:
            st.success('✅ You are unlikely to have heart disease.')
    else:
        st.warning("Model is not available.")
