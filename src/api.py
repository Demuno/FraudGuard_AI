import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

MODEL_PATH = os.path.join(".", "models", "anomaly_detection_model.joblib")
SCALER_PATH = os.path.join(".", "models", "scaler.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(title="API de Detecção de Fraudes", version="1.0")

class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict", tags=["Predictions"])
def predict(transaction: Transaction):
    input_data = pd.DataFrame([transaction.dict()])
    features_order = [f'V{i}' for i in range(1, 29)] + ['Amount']
    input_data = input_data[features_order]

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    status = "Suspeita de Fraude" if prediction[0] == -1 else "Transação Normal"
    
    return {"prediction": int(prediction[0]), "status": status}

@app.get("/", tags=["Health Check"])
def read_root():
     return {"message": "API está funcionando. Acesse /docs para ver a documentação."}