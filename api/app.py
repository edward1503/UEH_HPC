from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List, Optional
import os

app = FastAPI(title="Credit Default Prediction API")

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'saved_models', 'credit_model.joblib')
scaler_path = os.path.join(BASE_DIR, 'saved_models', 'scaler.joblib')
feature_names_path = os.path.join(BASE_DIR, 'saved_models', 'feature_names.joblib')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)
except Exception as e:
    print(f"Error loading model files: {str(e)}")
    model = None
    scaler = None
    feature_names = None

class CreditData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: Optional[float] = None

class PredictionResponse(BaseModel):
    default_probability: float
    prediction: int

@app.get("/")
def read_root():
    return {"message": "Credit Default Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CreditData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Prepare input data
    input_data = {
        'RevolvingUtilizationOfUnsecuredLines': data.RevolvingUtilizationOfUnsecuredLines,
        'age': data.age,
        'NumberOfTime30-59DaysPastDueNotWorse': data.NumberOfTime30_59DaysPastDueNotWorse,
        'DebtRatio': data.DebtRatio,
        'MonthlyIncome': data.MonthlyIncome,
        'NumberOfOpenCreditLinesAndLoans': data.NumberOfOpenCreditLinesAndLoans,
        'NumberOfTimes90DaysLate': data.NumberOfTimes90DaysLate,
        'NumberRealEstateLoansOrLines': data.NumberRealEstateLoansOrLines,
        'NumberOfTime60-89DaysPastDueNotWorse': data.NumberOfTime60_89DaysPastDueNotWorse,
        'NumberOfDependents': data.NumberOfDependents if data.NumberOfDependents is not None else 0.0
    }
    
    # Calculate derived features
    input_data['DebtToIncomeRatio'] = input_data['DebtRatio'] * input_data['MonthlyIncome']
    input_data['IncomePerPerson'] = input_data['MonthlyIncome'] / (input_data['NumberOfDependents'] + 1)
    
    # Convert to array and scale
    X = np.array([input_data[feature] for feature in feature_names]).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # Make prediction
    probability = model.predict_proba(X_scaled)[0, 1]
    prediction = int(probability > 0.5)
    
    return PredictionResponse(
        default_probability=float(probability),
        prediction=prediction
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 