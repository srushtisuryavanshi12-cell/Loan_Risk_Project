import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Loan Default Early Warning System", layout="wide")

st.title("🏦 Loan Default Early Warning & Risk Analytics System")
st.markdown("Enter loan application details below to compute the **Default Probability** and trigger risk warnings.")

# 1. Load model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("app/model.pkl")
    scaler = joblib.load("app/scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    st.sidebar.success("Model & Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")

st.header("Applicant Financial Metrics")

# 2. Input form layout
col1, col2, col3 = st.columns(3)

with col1:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500.0, max_value=100000.0, value=15000.0, step=500.0)
    int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=12.5, step=0.1)
    installment = st.number_input("Monthly Installment ($)", min_value=10.0, max_value=3000.0, value=450.0, step=10.0)
    annual_inc = st.number_input("Annual Income ($)", min_value=10000.0, max_value=1000000.0, value=65000.0, step=1000.0)

with col2:
    dti = st.number_input("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=100.0, value=18.5, step=0.5)
    delinq_2yrs = st.number_input("Delinquencies (Past 2 Yrs)", min_value=0, max_value=20, value=0, step=1)
    open_acc = st.number_input("Open Credit Lines", min_value=1, max_value=50, value=10, step=1)
    pub_rec = st.number_input("Public Derogatory Records", min_value=0, max_value=20, value=0, step=1)

with col3:
    revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, max_value=500000.0, value=12000.0, step=500.0)
    revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=150.0, value=45.0, step=1.0)
    total_acc = st.number_input("Total Credit Lines", min_value=1, max_value=100, value=22, step=1)

# 3. Assessment trigger
st.markdown("---")
if st.button("Evaluate Default Risk", type="primary"):
    input_data = pd.DataFrame([[
        loan_amnt, int_rate, installment, annual_inc,
        dti, delinq_2yrs, open_acc, pub_rec,
        revol_bal, revol_util, total_acc
    ]], columns=[
        'loan_amnt', 'int_rate', 'installment', 'annual_inc',
        'dti', 'delinq_2yrs', 'open_acc', 'pub_rec',
        'revol_bal', 'revol_util', 'total_acc'
    ])

    # Scale inputs and make prediction
    scaled_input = scaler.transform(input_data)
    default_prob = model.predict_proba(scaled_input)[0][1]

    st.subheader("Risk Analytics Output")
    st.metric(label="Predicted Default Probability", value=f"{default_prob * 100:.2f}%")

    # Early warning trigger thresholds
    if default_prob < 0.15:
        st.success("🟢 **LOW RISK**: Standard approval pathway recommended.")
    elif default_prob < 0.30:
        st.warning("🟡 **EARLY WARNING (MODERATE RISK)**: Flagged for secondary manual underwriting review.")
    else:
        st.error("🔴 **HIGH RISK ALERT**: Default likelihood exceeds risk tolerance threshold.")