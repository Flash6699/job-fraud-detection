import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def train_model():
    print("Loading dataset...")
    df = pd.read_csv("data/fake_job_postings.csv")

    text_columns = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits"
    ]

    for col in text_columns:
        df[col] = df[col].fillna("")

    df["text"] = (
        df["title"] + " " +
        df["company_profile"] + " " +
        df["description"] + " " +
        df["requirements"] + " " +
        df["benefits"]
    )

    X = df["text"]
    y = df["fraudulent"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])

    param_grid = {
        "tfidf__max_features": [5000],
        "tfidf__ngram_range": [(1,1)],
        "model__C": [2]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    print("Training model...")
    grid.fit(X, y)

    best_model = grid.best_estimator_

    print("Saving model...")
    joblib.dump(best_model, "models/fraud_pipeline.pkl")

    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()