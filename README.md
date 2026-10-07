# Credit Card Default Prediction

A machine learning project that predicts whether a customer is likely to default on their next credit card payment.

The project covers the complete workflow from data preprocessing and model development to API creation, containerization, cloud deployment, and CI/CD.

## Project Overview

Credit card default prediction is a classification problem where the goal is to predict whether a customer will default on their next payment based on their demographic information, credit limit, previous payment history, billing amounts, and payment amounts.

In this project, a Decision Tree Classifier was trained and evaluated to perform the prediction.

## Dataset

The dataset contains customer information and historical credit card payment data.

### Features

- `LIMIT_BAL` - Credit limit
- `sex` - Customer gender
- `education` - Education level
- `marriage` - Marital status
- `age` - Customer age
- `pay_0` to `pay_6` - Previous payment status
- `Bill_amt1` to `Bill_amt6` - Previous billing amounts
- `pay_amt1` to `pay_amt6` - Previous payment amounts
- `AVG_Bill_amt` - Average billing amount
- `PAY_TO_BILL_ratio` - Payment-to-bill ratio

### Target

- `next_month_default`
  - `0` - No default
  - `1` - Default

The `Customer_ID` column was removed before model training because it does not provide useful predictive information.

## Machine Learning Workflow

The project follows these main steps:

1. Data loading and inspection
2. Data cleaning
3. Handling missing values
4. Feature and target separation
5. Categorical and numerical feature identification
6. Train-test split
7. Data preprocessing
8. Model training
9. Model evaluation
10. Model selection
11. Model serialization using Joblib
12. API development using FastAPI
13. Frontend integration using Streamlit
14. Docker containerization
15. Cloud deployment
16. CI/CD setup using GitHub Actions

## Model

The final model used in the project is:

**Decision Tree Classifier**

Configuration:

- `max_depth = 4`
- `random_state = 42`

The final model achieved approximately:

- **Training Accuracy:** 84.25%
- **Testing Accuracy:** 84.28%

The model was saved using Joblib and loaded by the FastAPI backend for making predictions.

## Application Architecture

The application follows this workflow:

```text
User
  ↓
Streamlit Interface
  ↓
FastAPI
  ↓
Machine Learning Model
  ↓
Prediction
  ↓
Streamlit Interface
```


## Technologies Used ##

# Programming & Data Processing
Python
Pandas
NumPy

# Machine Learning 
Scikit-learn
Joblib

# API & Frontend 
FastAPI
Pydantic
Streamlit
Requests

# Deployment & DevOps
Docker
GitHub
GitHub Actions
Render

## How to Run the Project Locally

Follow the steps below to run the project on your local machine.

### 1. Clone the Repository

Open a terminal and clone the repository:

```bash
git clone https://github.com/Nihallllz/Credit_Default-api.git
cd Credit_Default-api
```

### 2. Install the Required Dependencies

Install all the Python packages required for the project:

```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI Backend

The FastAPI backend handles the prediction requests and communicates with the trained machine learning model.

Start the backend using:

```bash
uvicorn apiii:app --reload
```

Once started, the API will be available at:

`http://127.0.0.1:8000`

You can also open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

### 4. Start the Streamlit Frontend

Keep the FastAPI terminal running and open a **new terminal** in the same project directory.

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will open the application in your web browser.

### 5. Use the Application

Enter the required customer information in the Streamlit interface and submit the form.

The application follows this flow:

**Streamlit → FastAPI → Machine Learning Model → Prediction → Streamlit**

The Streamlit frontend sends the customer data to the FastAPI backend. FastAPI passes the data to the trained Decision Tree model, which generates the prediction and sends the result back to the Streamlit application.