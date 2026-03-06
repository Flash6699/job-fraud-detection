import streamlit as st
import joblib

# Load model once
@st.cache_resource
def load_model():
    return joblib.load("models/fraud_pipeline.pkl")

model = load_model()

st.title("🕵️ Fraud Job Detection System")

st.write("Paste a job description below to check if it is fraudulent.")

user_input = st.text_area("Job Description")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter job text.")
    else:
        probability = model.predict_proba([user_input])[0][1]
        prediction = model.predict([user_input])[0]

        st.subheader("Result")

        if prediction == 1:
            st.error(f"⚠ Fraudulent Job Detected")
        else:
            st.success("✅ Likely Real Job")

        st.write(f"Fraud Probability: {probability:.4f}")
print("Done")        