import streamlit as st
import numpy as np
import pickle
import os

st.title("🤖 EMI Prediction")

# ======================
# PATH FIX
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# go to emi_app (parent of pages)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

model_path = os.path.join(PROJECT_ROOT, "model.pkl")

st.write("Looking for model at:", model_path)  # DEBUG

# ======================
# LOAD MODEL
# ======================
if os.path.exists(model_path):
    model = pickle.load(open(model_path, "rb"))
    st.success("Model Loaded ✅")
else:
    st.error(f"Model NOT found ❌ at: {model_path}")
    st.stop()

# ======================
# INPUT
# ======================
st.subheader("Enter Details")

salary = st.number_input("Monthly Salary", 0)
credit_score = st.slider("Credit Score", 300, 850, 650)
emi = st.number_input("Current EMI", 0)

# ======================
# PREDICT
# ======================
if st.button("Predict"):

    input_data = np.array([[salary, credit_score, emi]])

    pred = model.predict(input_data)

    if pred[0] == 1:
        st.success("✅ Eligible")
    else:
        st.error("❌ Not Eligible")