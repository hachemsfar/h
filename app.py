import streamlit as st
import numpy as np
import joblib
import os
import sklearn

# Ensure compatibility with scikit-learn
st.sidebar.info(f"Scikit-learn version: {sklearn.__version__}")

# Load the trained model
MODEL_PATH = "diabetes_rf_model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error("Model file not found. Please check the file path.")
        return None

model = load_model()

# Define function for prediction
def predict_diabetes(inputs):
    if model is None:
        return "Model not available"
    try:
        input_array = np.array(inputs).reshape(1, -1)
        prediction = model.predict(input_array)
        return "🩺 Diabetes Detected" if prediction[0] == 1 else "✅ No Diabetes Detected"
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# Streamlit UI
st.title("🩺 Early Diabetes Prediction App")
st.write("Enter patient details below to predict the likelihood of diabetes.")

# Sidebar Input Fields
st.sidebar.header("Patient Details")
age = st.sidebar.slider("Age", 16, 90, 30)
gender = st.sidebar.radio("Gender", ("Male", "Female"))

# Symptoms input
symptoms = {
    "Polyuria (Excessive Urination)": st.sidebar.radio("Polyuria", ("Yes", "No")),
    "Polydipsia (Excessive Thirst)": st.sidebar.radio("Polydipsia", ("Yes", "No")),
    "Sudden Weight Loss": st.sidebar.radio("Sudden Weight Loss", ("Yes", "No")),
    "Weakness": st.sidebar.radio("Weakness", ("Yes", "No")),
    "Polyphagia (Excessive Hunger)": st.sidebar.radio("Polyphagia", ("Yes", "No")),
    "Genital Thrush": st.sidebar.radio("Genital Thrush", ("Yes", "No")),
    "Visual Blurring": st.sidebar.radio("Visual Blurring", ("Yes", "No")),
    "Itching": st.sidebar.radio("Itching", ("Yes", "No")),
    "Irritability": st.sidebar.radio("Irritability", ("Yes", "No")),
    "Delayed Healing": st.sidebar.radio("Delayed Healing", ("Yes", "No")),
    "Partial Paresis": st.sidebar.radio("Partial Paresis", ("Yes", "No")),
    "Muscle Stiffness": st.sidebar.radio("Muscle Stiffness", ("Yes", "No")),
    "Alopecia (Hair Loss)": st.sidebar.radio("Alopecia", ("Yes", "No")),
    "Obesity": st.sidebar.radio("Obesity", ("Yes", "No")),
}

# Convert categorical inputs to numerical
user_inputs = [
    age,
    1 if gender == "Male" else 0
] + [1 if symptoms[s] == "Yes" else 0 for s in symptoms]

# Prediction Button
if st.sidebar.button("Predict Diabetes"):  
    result = predict_diabetes(user_inputs)
    st.success(result)

# Displaying information
st.markdown("---")
st.markdown("### How does it work?")
st.write(
    "This app uses a trained Random Forest model to analyze patient symptoms "
    "and predict the likelihood of diabetes based on input features. "
    "Results are based on statistical patterns detected in historical data."
)
