# import streamlit as st
# import joblib
# import pandas as pd

# st.title("Customer Churn Prediction")

# st.write("Predict whether a telecom customer will churn.")


# model = joblib.load("churn_model.pkl")


# gender = st.selectbox("Gender", ["Male", "Female"])

# senior = st.selectbox(
#     "Senior Citizen",
#     [0,1]
# )

# partner = st.selectbox(
#     "Has Partner",
#     ["Yes","No"]
# )

# dependents = st.selectbox(
#     "Dependents",
#     ["Yes","No"]
# )

# tenure = st.slider(
#     "Tenure (months)",
#     0,72
# )

# monthly_charges = st.number_input(
#     "Monthly Charges"
# )

# total_charges = st.number_input(
#     "Total Charges"
# )

# contract = st.selectbox(
#     "Contract Type",
#     ["Month-to-month","One year","Two year"]
# )

# input_data = pd.DataFrame({
#     "gender":[gender],
#     "SeniorCitizen":[senior],
#     "Partner":[partner],
#     "Dependents":[dependents],
#     "tenure":[tenure],
#     "MonthlyCharges":[monthly_charges],
#     "TotalCharges":[total_charges],
#     "Contract":[contract]
# })

# if st.button("Predict Churn"):

#     prediction = model.predict(input_data)[0]
#     probability = model.predict_proba(input_data)[0][1]

#     st.subheader("Prediction")

#     if prediction == "Yes":
#         st.error(f"Customer likely to churn (Probability {probability:.2f})")
#     else:
#         st.success(f"Customer likely to stay (Probability {probability:.2f})")
        


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    return df

df = load_data()

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Data Insights", "Churn Predictor"]
)

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":
    st.title("Customer Churn Prediction App")
    st.write(
        """
        This application predicts whether a telecom customer is likely to **churn**.
        The model was trained using machine learning on a telecom customer dataset.
        """
    )
    st.subheader("Project Overview")
    st.write(
        """
        **Goal:** Predict customer churn so companies can retain customers.

        **Tech Stack**
        - Python
        - Pandas
        - Scikit-learn
        - Streamlit

        **Model**
        - Machine Learning Pipeline
        - Preprocessing + Random Forest Classification
        """
    )

# -----------------------------
# DATA INSIGHTS
# -----------------------------
elif page == "Data Insights":
    st.title("Data Insights")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Churn Distribution")
    churn_counts = df["Churn"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(churn_counts.index, churn_counts.values, color=["steelblue", "tomato"])
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    ax.set_title("Churn Distribution")
    plt.tight_layout()          # ← fix: prevents label clipping
    st.pyplot(fig)
    plt.close(fig)              # ← fix: prevents matplotlib memory warning

    st.subheader("Contract Type vs Churn")
    contract_churn = pd.crosstab(df["Contract"], df["Churn"])
    fig2, ax2 = plt.subplots()
    contract_churn.plot(kind="bar", ax=ax2, color=["steelblue", "tomato"])
    ax2.set_title("Contract Type vs Churn")
    ax2.set_xlabel("Contract Type")
    ax2.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()          # ← fix: prevents label clipping
    st.pyplot(fig2)
    plt.close(fig2)

# -----------------------------
# CHURN PREDICTOR
# -----------------------------
elif page == "Churn Predictor":
    st.title("Predict Customer Churn")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col2:
        tenure = st.slider("Tenure (months)", 0, 72)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
        total_charges = st.number_input("Total Charges", min_value=0.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Contract": [contract]
    })

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")
        if prediction == "Yes":
            st.error(f"⚠️ Customer likely to churn (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Customer likely to stay (Probability: {probability:.2f})")