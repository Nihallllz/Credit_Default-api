import streamlit as st
import pandas as pd
# import joblib
import requests

# model = joblib.load("Credit_Default.pkl")

st.title("Credit Card Default Prediction")
st.write("This application uses a machine learning model to predict whether a customer is likely to default on their next credit card payment based on their financial and payment information.")

LIMIT_BAL = st.number_input("LIMIT_BAL", min_value=10000)
age = st.number_input("Age", max_value=79, min_value=21)

sex = st.selectbox("Sex", [0, 1])
education = st.selectbox("Education", [0, 1, 2, 3, 4, 5, 6])        
marriage = st.selectbox("Marriage", [0, 1, 2, 3])


st.subheader("Payment Status , Bill Amount & Payment Amount")

col1, col2, col3 = st.columns(3)
with col1:

    pay_0 = st.selectbox("Pay 0",[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    pay_2 = st.selectbox("Pay 2", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    pay_3 = st.selectbox("Pay 3", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    pay_4 = st.selectbox("Pay 4", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    pay_5 = st.selectbox("Pay 5", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    pay_6 = st.selectbox("Pay 6", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])

with col2:
    Bill_amt1 = st.number_input("Bill Amount 1")
    Bill_amt2 = st.number_input("Bill Amount 2")
    Bill_amt3 = st.number_input("Bill Amount 3")
    Bill_amt4 = st.number_input("Bill Amount 4")
    Bill_amt5 = st.number_input("Bill Amount 5")
    Bill_amt6 = st.number_input("Bill Amount 6")

with col3:
    pay_amt1 = st.number_input("Payment Amount 1")
    pay_amt2 = st.number_input("Payment Amount 2")
    pay_amt3 = st.number_input("Payment Amount 3")
    pay_amt4 = st.number_input("Payment Amount 4")
    pay_amt5 = st.number_input("Payment Amount 5")
    pay_amt6 = st.number_input("Payment Amount 6")

st.subheader("Additional Features")

col1, col2 = st.columns(2)

with col1:
    AVG_Bill_amt = st.number_input("Average Bill Amount")

with col2:
    PAY_TO_BILL_ratio = st.number_input("Pay to Bill Ratio")



new_customer = pd.DataFrame({
        "marriage": [marriage],
    "sex": [sex],
    "education": [education],
    "LIMIT_BAL": [LIMIT_BAL],
    "age": [age],
    "pay_0": [pay_0],
    "pay_2": [pay_2],
    "pay_3": [pay_3],
    "pay_4": [pay_4],
    "pay_5": [pay_5],
    "pay_6": [pay_6],
    "Bill_amt1": [Bill_amt1],
    "Bill_amt2": [Bill_amt2],
    "Bill_amt3": [Bill_amt3],
    "Bill_amt4": [Bill_amt4],
    "Bill_amt5": [Bill_amt5],
    "Bill_amt6": [Bill_amt6],
    "pay_amt1": [pay_amt1],
    "pay_amt2": [pay_amt2],
    "pay_amt3": [pay_amt3],
    "pay_amt4": [pay_amt4],
    "pay_amt5": [pay_amt5],
    "pay_amt6": [pay_amt6],
    "AVG_Bill_amt": [AVG_Bill_amt],
    "PAY_TO_BILL_ratio": [PAY_TO_BILL_ratio]
        

    })
if st.button("Predict"):
    
    response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=new_customer.to_dict(orient="records")[0]
)
    result = response.json()
    prediction = result["prediction"]
    st.header(prediction)


    if prediction[0] == 1:
        st.error("Customer is likely to default.")
    else:
        st.success("Customer is unlikely to default.")

    # probability = model.predict_proba(new_customer)[0][1]
    # st.write(f"Default Probability: {probability:.2%}")

st.write(new_customer)








