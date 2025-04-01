import streamlit as st
import pandas as pd
import numpy as np
import pickle  # For loading model
import os

# Load the trained model
MODEL_PATH = "model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
else:
    st.error("Model file not found. Please upload `model.pkl`.")
    st.stop()

# Streamlit UI
st.title("📊 Insurance Claim Attorney Prediction")
st.write("Predict whether an attorney will be involved in an insurance claim.")

# User input fields
CLMSEX = st.radio("Claimant Gender:", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
CLMINSUR = st.radio("Claimant was insured:", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
SEATBELT = st.radio("Seatbelt used:", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
CLMAGE = st.number_input("Claimant Age:", min_value=18, max_value=100, step=1)
LOSS = st.number_input("Financial Loss (in $):", min_value=0.0, format="%.2f")
ACCIDENT_SEVERITY = st.selectbox("Accident Severity:", ["Minor", "Moderate", "Severe"])
CLAIM_AMOUNT_REQUESTED = st.number_input("Claim Amount Requested ($):", min_value=0.0, format="%.2f")
CLAIM_APPROVAL_STATUS = st.radio("Claim Approved:", [0, 1], format_func=lambda x: "Approved" if x == 1 else "Denied")
SETTLEMENT_AMOUNT = st.number_input("Settlement Amount ($):", min_value=0.0, format="%.2f")
POLICY_TYPE = st.selectbox("Policy Type:", ["Comprehensive", "Third-Party"])
DRIVING_RECORD = st.selectbox("Driving Record:", ["Clean", "Minor Offenses", "Major Offenses"])

# Mapping categorical variables
severity_map = {"Minor": 0, "Moderate": 1, "Severe": 2}
policy_map = {"Comprehensive": 0, "Third-Party": 1}
drive_record_map = {"Clean": 0, "Minor Offenses": 1, "Major Offenses": 2}

# Convert input to model format
input_data = np.array([
    CLMSEX, CLMINSUR, SEATBELT, CLMAGE, LOSS, severity_map[ACCIDENT_SEVERITY],
    CLAIM_AMOUNT_REQUESTED, CLAIM_APPROVAL_STATUS, SETTLEMENT_AMOUNT,
    policy_map[POLICY_TYPE], drive_record_map[DRIVING_RECORD]
]).reshape(1, -1)

# Predict attorney involvement
if st.button("🔍 Predict Attorney Involvement"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100  # Get probability for '1'
    if prediction == 1:
        st.error(f"🔴 Attorney **WILL** be involved. (Confidence: {probability:.2f}%)")
    else:
        st.success(f"🟢 No attorney involvement expected. (Confidence: {100-probability:.2f}%)")

st.write("\n📌 **Note:** This model helps insurers assess potential legal risks in claims.")
