# app.py
import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🕵️ Fake Job Posting Detector")

job_description = st.text_area("Paste the job description:")

if st.button("Check if it's Fake"):
    if job_description.strip() == "":
        st.warning("Please enter a job description.")
    else:
        prediction = model.predict([job_description])[0]
        result = "✅ Legitimate Job" if prediction == 0 else "❌ Fake Job"
        st.success(result)


        if prediction == 0:
            st.success(f"✅ This looks like a **REAL** job posting. Confidence: {round(proba[0]*100, 2)}%")
        else:
            st.error(f"🚨 Warning: This may be a **FAKE** job posting. Confidence: {round(proba[1]*100, 2)}%")
