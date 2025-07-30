import streamlit as st
import pickle

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# App title
st.title("🕵️‍♀️ Fake Job Posting Detector")
st.markdown("Paste the job posting below and find out if it's real or fake.")

# Text input
job_text = st.text_area("Job Description", height=300)

if st.button("Check"):
    if job_text.strip() == "":
        st.warning("Please enter some job description text.")
    else:
        # Vectorize input
        vect_text = tfidf.transform([job_text])

        # Predict
        prediction = model.predict(vect_text)[0]
        proba = model.predict_proba(vect_text)[0]

        if prediction == 0:
            st.success(f"✅ This looks like a **REAL** job posting. Confidence: {round(proba[0]*100, 2)}%")
        else:
            st.error(f"🚨 Warning: This may be a **FAKE** job posting. Confidence: {round(proba[1]*100, 2)}%")
