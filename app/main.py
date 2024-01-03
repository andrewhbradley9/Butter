from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from app.model.model import predict_pipeline, load_data, preprocess_input, scaleFun

app = FastAPI()


class TextIn(BaseModel):
    tenure: float
    PhoneService: str
    OnlineSecurity: str
    OnlineBackup: str
    PaperlessBilling: str
    MonthlyCharges: float
    TotalCharges: float
    InternetService: str
    Contract: str
    PaymentMethod: str

# class PredictionOut(BaseModel):
#     language: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict_endpoint(payload: TextIn):
    df1 = load_data()
    processed_input = preprocess_input(payload)
    scaled_input = scaleFun(processed_input, df1)
    result = predict_pipeline(scaled_input)
    return {"prediction":result}