import streamlit as st
import joblib

# Load model and vectorizer
try:
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    st.error("❌ Required files model.joblib or vectorizer.pkl not found. Please upload them.")
    st.stop()

st.title("Fake Job Posting Classifier")

job_description = st.text_area("Enter job description:")

if st.button("Check if Fake"):
    if job_description.strip() == "":
        st.warning("Please enter a job description.")
    else:
        features = vectorizer.transform([job_description])
        prediction = model.predict(features)[0]
        result = "✅ Real Job" if prediction == 0 else "🚨 Fake Job"
        st.success(f"Prediction: {result}")
