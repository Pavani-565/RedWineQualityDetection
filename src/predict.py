import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

feature_names = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

def predict_wine(features):
    input_df = pd.DataFrame([features], columns=feature_names)

    scaled_data = scaler.transform(input_df)

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0]

    confidence = max(probability) * 100

    return prediction, confidence