from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Define request structure
class JobInput(BaseModel):
    text: str

app = FastAPI(title="Fraud Job Detection API")

# Load trained model once at startup
model = joblib.load("models/fraud_pipeline.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(job: JobInput):
    probability = model.predict_proba([job.text])[0][1]
    prediction = int(model.predict([job.text])[0])

    return {
        "prediction": prediction,
        "fraud_probability": round(float(probability), 4)
    }