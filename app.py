import streamlit as st
import joblib
import numpy as np

# Load the saved model and label encoder
model = joblib.load("ld_classifier_model.pkl")
le = joblib.load("label_encoder.pkl")

st.title("Learning Disability Detector")

# Input fields
reading_speed = st.number_input("Reading Speed (words per minute)", min_value=20, max_value=150, value=90)
spelling_accuracy = st.number_input("Spelling Accuracy (%)", min_value=0, max_value=100, value=75)
math_score = st.number_input("Math Score (%)", min_value=0, max_value=100, value=70)
attention_span = st.number_input("Attention Span (minutes)", min_value=5, max_value=60, value=30)

if st.button("Predict"):
    features = np.array([[reading_speed, spelling_accuracy, math_score, attention_span]])
    pred_encoded = model.predict(features)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    st.success(f"Predicted Learning Disability: **{pred_label}**")
st.write("### Model Explanation")
st.write("This model predicts the likelihood of a learning disability based on the following features:")