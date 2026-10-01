import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
import joblib
import warnings
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")


df = pd.read_csv("train_dataset_final1.csv")
# df.head()

# df.shape
# df.info()
# df.duplicated()
# df = df.drop("Customer_ID", axis = 1)
# df.head()

X = df.drop("next_month_default", axis=1)
y = df["next_month_default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numeric_features = ["age","LIMIT_BAL","Bill_amt1","Bill_amt2","Bill_amt3","Bill_amt4","Bill_amt5","Bill_amt6","pay_amt1","pay_amt2","pay_amt3","pay_amt4",
                    "pay_amt5","pay_amt6","AVG_Bill_amt","PAY_TO_BILL_ratio","pay_0","pay_2","pay_3","pay_4","pay_5","pay_6"]

categorical_features = ["sex","education","marriage"]

X_train_numeric = X_train[numeric_features]
X_train_categorical = X_train[categorical_features]

imputer = SimpleImputer(strategy="mean")
X_train_numeric = imputer.fit_transform(X_train)
X_test_numeric = imputer.transform(X_test)

scaler = StandardScaler()
X_train_numeric = scaler.fit_transform(X_train_numeric)
X_test_numeric = scaler.transform(X_test_numeric)

numeric_transformer = Pipeline(
    steps=[
        ("Numeric_Imputer", SimpleImputer(strategy="mean")),
        ("Numeric_Scaler", StandardScaler()),


    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("Categorical_Imputer", SimpleImputer(strategy="most_frequent")),
        ("categorical_transformer" , OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

model_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(max_depth=4,random_state=42))
    ]
)

model_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(max_depth=4,

                                    random_state=42))
    ]
)

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

# classification_report(y_test, y_pred)


# confusion_matrix(y_test, y_pred)

model_pipeline.score(X_train, y_train)
model_pipeline.score(X_test, y_test)

joblib.dump(model_pipeline, "Credit_Default.pkl")

loaded_pipeline = joblib.load("Credit_Default.pkl")

marriage = int(input("Enter marriage: "))
sex = int(input("Enter sex: "))
education = int(input("Enter education: "))
LIMIT_BAL = float(input("Enter credit limit: "))
age = int(input("Enter age: "))

pay_0 = int(input("Enter pay_0: "))
pay_2 = int(input("Enter pay_2: "))
pay_3 = int(input("Enter pay_3: "))
pay_4 = int(input("Enter pay_4: "))
pay_5 = int(input("Enter pay_5: "))
pay_6 = int(input("Enter pay_6: "))

Bill_amt1 = float(input("Enter Bill_amt1: "))
Bill_amt2 = float(input("Enter Bill_amt2: "))
Bill_amt3 = float(input("Enter Bill_amt3: "))
Bill_amt4 = float(input("Enter Bill_amt4: "))
Bill_amt5 = float(input("Enter Bill_amt5: "))
Bill_amt6 = float(input("Enter Bill_amt6: "))

pay_amt1 = float(input("Enter pay_amt1: "))
pay_amt2 = float(input("Enter pay_amt2: "))
pay_amt3 = float(input("Enter pay_amt3: "))
pay_amt4 = float(input("Enter pay_amt4: "))
pay_amt5 = float(input("Enter pay_amt5: "))
pay_amt6 = float(input("Enter pay_amt6: "))

AVG_Bill_amt = float(input("Enter AVG_Bill_amt: "))
PAY_TO_BILL_ratio = float(input("Enter PAY_TO_BILL_ratio: "))

new_customer = pd.DataFrame([{
    "marriage": marriage,
    "sex": sex,
    "education": education,
    "LIMIT_BAL": LIMIT_BAL,
    "age": age,

    "pay_0": pay_0,
    "pay_2": pay_2,
    "pay_3": pay_3,
    "pay_4": pay_4,
    "pay_5": pay_5,
    "pay_6": pay_6,

    "Bill_amt1": Bill_amt1,
    "Bill_amt2": Bill_amt2,
    "Bill_amt3": Bill_amt3,
    "Bill_amt4": Bill_amt4,
    "Bill_amt5": Bill_amt5,
    "Bill_amt6": Bill_amt6,

    "pay_amt1": pay_amt1,
    "pay_amt2": pay_amt2,
    "pay_amt3": pay_amt3,
    "pay_amt4": pay_amt4,
    "pay_amt5": pay_amt5,
    "pay_amt6": pay_amt6,

    "AVG_Bill_amt": AVG_Bill_amt,
    "PAY_TO_BILL_ratio": PAY_TO_BILL_ratio
}])

prediction = loaded_pipeline.predict(new_customer)

if prediction[0] == 1:
    print("Likely to default")
else:
    print("Likely not to default")





