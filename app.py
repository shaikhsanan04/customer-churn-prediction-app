import streamlit as st
import joblib
import pandas as pd

st.title("Customer Churn Prediction")

st.write("Predict whether a telecom customer will churn.")


model = joblib.load("churn_model.pkl")


gender = st.selectbox("Gender", ["Male", "Female"])

senior = st.selectbox(
    "Senior Citizen",
    [0,1]
)

partner = st.selectbox(
    "Has Partner",
    ["Yes","No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes","No"]
)

tenure = st.slider(
    "Tenure (months)",
    0,72
)

monthly_charges = st.number_input(
    "Monthly Charges"
)

total_charges = st.number_input(
    "Total Charges"
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month","One year","Two year"]
)

input_data = pd.DataFrame({
    "gender":[gender],
    "SeniorCitizen":[senior],
    "Partner":[partner],
    "Dependents":[dependents],
    "tenure":[tenure],
    "MonthlyCharges":[monthly_charges],
    "TotalCharges":[total_charges],
    "Contract":[contract]
})

if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction")

    if prediction == "Yes":
        st.error(f"Customer likely to churn (Probability {probability:.2f})")
    else:
        st.success(f"Customer likely to stay (Probability {probability:.2f})")
        
