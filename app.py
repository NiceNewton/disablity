import streamlit as st
import joblib
import numpy as np
import os

# Page setup
st.set_page_config(page_title="Learning Disability Detector", page_icon="🧠", layout="centered")

st.title("🧠 Learning Disability Detector")
st.markdown("This app predicts the **likelihood of a learning disability** based on academic and behavioral inputs.")

# Load model and label encoder safely
@st.cache_resource
def load_model_files():
    model_path = "ld_classifier_model.pkl"
    encoder_path = "label_encoder.pkl"

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        return None, None

    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    return model, le

model, le = load_model_files()

if model is None or le is None:
    st.error("❌ Required model files not found. Please ensure `ld_classifier_model.pkl` and `label_encoder.pkl` exist in the same folder.")
    st.stop()

# Input fields
st.subheader("📊 Enter Student Information")

reading_speed = st.number_input("📖 Reading Speed (words per minute)", min_value=20, max_value=150, value=90)
spelling_accuracy = st.number_input("📝 Spelling Accuracy (%)", min_value=0, max_value=100, value=75)
math_score = st.number_input("➗ Math Score (%)", min_value=0, max_value=100, value=70)
attention_span = st.number_input("⏱️ Attention Span (minutes)", min_value=5, max_value=60, value=30)

# Prediction
if st.button("🔍 Predict"):
    features = np.array([[reading_speed, spelling_accuracy, math_score, attention_span]])
    pred_encoded = model.predict(features)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    st.success(f"✅ Predicted Learning Disability: **{pred_label}**")

# Explanation Section
with st.expander("📘 Model Explanation"):
    st.markdown("""
    The model predicts potential learning disabilities using:
    
    - **Reading Speed** (words per minute)
    - **Spelling Accuracy** (% of correct spellings)
    - **Math Score** (percentage)
    - **Attention Span** (average focus time in minutes)
    
    It was trained using labeled data and supervised learning techniques.
    """)

st.markdown("---")
st.caption("🔬 Developed as an academic ML project")
