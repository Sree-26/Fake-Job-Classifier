import streamlit as st
import joblib
import os

# Title
st.title("🕵️‍♀️ Fake Job Classifier")
st.write("Paste a job description below to check if it's real or fake.")

# Check if model and vectorizer exist
if not os.path.exists("model.joblib") or not os.path.exists("vectorizer.pkl"):
    st.error("❌ Required files `model.joblib` or `vectorizer.pkl` not found. Please upload them.")
else:
    try:
        model = joblib.load("model.joblib")
        vectorizer = joblib.load("vectorizer.pkl")

        # Input
        job_description = st.text_area("✏️ Job Description")

        if st.button("🔍 Predict"):
            if job_description.strip() == "":
                st.warning("Please enter a job description.")
            else:
                # Preprocess & predict
                features = vectorizer.transform([job_description])
                prediction = model.predict(features)[0]
                result = "✅ Legit Job" if prediction == 0 else "⚠️ Fake Job Posting"

                st.subheader("Prediction Result:")
                st.success(result)

    except Exception as e:
        st.error(f"🚨 Something went wrong: {e}")
