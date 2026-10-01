import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

model = joblib.load("Credit_Default.pkl")
@app.get("/")
def home():
    return {"message": "Credit Default API is running"}


class Customer(BaseModel):
    age: int
    LIMIT_BAL: float

    Bill_amt1: float
    Bill_amt2: float
    Bill_amt3: float
    Bill_amt4: float
    Bill_amt5: float
    Bill_amt6: float

    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float

    AVG_Bill_amt: float
    PAY_TO_BILL_ratio: float

    pay_0: int
    pay_2: int
    pay_3: int
    pay_4: int
    pay_5: int
    pay_6: int

    sex: int
    education: int
    marriage: int

@app.post("/predict")
def predict(customer: Customer):
    customer_data = customer.model_dump()

    customer_df = pd.DataFrame([customer_data])

    prediction = model.predict(customer_df)

    return {
        "prediction": int(prediction[0])
    }

"""python -m uvicorn app:app --reload"""
