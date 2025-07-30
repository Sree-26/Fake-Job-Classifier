import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.joblib")         # or model.pkl
vectorizer = joblib.load("vectorizer.pkl")  # must match training

st.title("🕵️‍♀️ Fake Job Posting Classifier")

job_description = st.text_area("Enter job description:")

if st.button("Check"):
    if job_description.strip() == "":
        st.warning("Please enter a job description.")
    else:
        # Vectorize user input
        job_vec = vectorizer.transform([job_description])
        
        # Predict
        prediction = model.predict(job_vec)[0]

        if prediction == 1:
            st.error("🚨 This job is likely FAKE.")
        else:
            st.success("✅ This job seems REAL.")
