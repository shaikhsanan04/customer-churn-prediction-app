# import joblib
# import pandas as pd
# from sklearn.pipeline import Pipeline

# # import everything used in your notebook
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier

# df = pd.read_csv("Telco-Customer-Churn.csv")

# # Convert TotalCharges to numeric
# df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# # Remove rows where TotalCharges became NaN
# df = df.dropna()

# # rebuild preprocessing exactly like notebook
# # (use your exact column lists here)

# num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
# cat_cols = ["gender","Partner","Dependents","Contract"]

# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), num_cols),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
#     ]
# )

# model = Pipeline([
#     ("prep", preprocessor),
#     ("clf", RandomForestClassifier())
# ])

# X = df.drop("Churn", axis=1)
# y = df["Churn"]

# model.fit(X, y)

# joblib.dump(model, "churn_model.pkl")

# print("Model saved successfully")


import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Only the columns the app will send at prediction time
num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = ["gender", "Partner", "Dependents", "Contract"]

feature_cols = num_cols + cat_cols   # 8 total — matches input_data exactly

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

model = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

X = df[feature_cols]   # ← only these 8 columns, not all columns
y = df["Churn"]

model.fit(X, y)
joblib.dump(model, "churn_model.pkl")
print("Model saved successfully")