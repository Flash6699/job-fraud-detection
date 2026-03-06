import joblib

def load_model():
    return joblib.load("models/fraud_pipeline.pkl")

def predict_job(text):
    model = load_model()
    probability = model.predict_proba([text])[0][1]
    prediction = model.predict([text])[0]

    return {
        "prediction": int(prediction),
        "fraud_probability": round(float(probability), 4)
    }

if __name__ == "__main__":
    sample_text = """
    Earn money quickly! Work from home. No experience required.
    Click the link below to start earning today!
    """

    result = predict_job(sample_text)
    print(result)