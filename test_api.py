from fastapi.testclient import TestClient
from apiii import app

client = TestClient(app)


def test_predict():
    data = {
        "age": 30,
        "LIMIT_BAL": 50000,
        "Bill_amt1": 20000,
        "Bill_amt2": 18000,
        "Bill_amt3": 15000,
        "Bill_amt4": 12000,
        "Bill_amt5": 10000,
        "Bill_amt6": 8000,
        "pay_amt1": 2000,
        "pay_amt2": 2000,
        "pay_amt3": 1500,
        "pay_amt4": 1500,
        "pay_amt5": 1000,
        "pay_amt6": 1000,
        "AVG_Bill_amt": 13833,
        "PAY_TO_BILL_ratio": 0.1,
        "pay_0": 0,
        "pay_2": 0,
        "pay_3": 0,
        "pay_4": 0,
        "pay_5": 0,
        "pay_6": 0,
        "sex": 1,
        "education": 2,
        "marriage": 1
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert result["prediction"] in [0, 1]