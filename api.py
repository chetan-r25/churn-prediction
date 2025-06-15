from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("xgb_churn_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

class Customer(BaseModel):
    gender: str
    tenure: int
    MonthlyCharges: float
    # Add other optional fields here...
    dependents: Optional[str] = None

def preprocess(dict_data):
    # Fill missing optional fields with default values if necessary
    # Ensure the DataFrame columns are in the correct order
    df = pd.DataFrame([dict_data])
    # You may need to specify columns order as per your model's training
    # df = df[expected_columns]
    X, _ = preprocess_data(df)
    return X

# Add or import the preprocess_data function
def preprocess_data(df):
    # Ensure columns match what encoder expects
    # If using OneHotEncoder or similar, handle unknown categories
    df_encoded = encoder.transform(df)
    df_scaled = scaler.transform(df_encoded)
    return df_scaled, None

@app.post("/predict")
def predict(data: Customer):
    X = preprocess(data.dict())
    proba = model.predict_proba(X)[0, 1]
    return {"churn_probability": proba}

# Run via: uvicorn api:app --reload --port 8000
