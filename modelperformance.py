import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

st.title("📊 Model Performance Dashboard")

# ==============================
# 📁 PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "emi_prediction_dataset.csv")
model_path = os.path.join(BASE_DIR, "model.pkl")

# ==============================
# 📂 LOAD DATA
# ==============================
try:
    df = pd.read_csv(data_path)
    st.success("Dataset Loaded ✅")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ==============================
# 🧹 DATA CLEANING (FINAL FIX)
# ==============================
num_cols = ['monthly_salary', 'credit_score', 'current_emi_amount']

for col in num_cols:
    df[col] = df[col].astype(str)
    
    # remove junk characters
    df[col] = df[col].str.replace(r'[^\d.]', '', regex=True)
    
    # fix multiple dots (127000.0.0 → 127000.0)
    df[col] = df[col].str.replace(r'\.(?=.*\.)', '', regex=True)
    
    # convert to float
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Clean target
df['emi_eligibility'] = df['emi_eligibility'].astype(str).str.lower().str.strip()
df['emi_eligibility'] = df['emi_eligibility'].map({
    'yes': 1,
    'no': 0,
    'eligible': 1,
    'not_eligible': 0
})

# Drop NaN
df = df.dropna()

# ==============================
# 🎯 FEATURES
# ==============================
X = df[['monthly_salary', 'credit_score', 'current_emi_amount']]
y = df['emi_eligibility']

# ==============================
# 🤖 TRAIN MODEL (fresh)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model (optional)
pickle.dump(model, open(model_path, "wb"))

# ==============================
# 🔮 PREDICTION
# ==============================
y_pred = model.predict(X_test)

# ==============================
# 📊 METRICS
# ==============================
st.subheader("📈 Model Metrics")

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{acc:.2f}")

with col2:
    st.metric("Precision", f"{prec:.2f}")

with col3:
    st.metric("Recall", f"{rec:.2f}")

with col4:
    st.metric("F1 Score", f"{f1:.2f}")

# ==============================
# 📉 VISUALIZATION
# ==============================
st.subheader("📊 Prediction Distribution")

fig, ax = plt.subplots()
ax.hist(y_pred)
ax.set_title("Prediction Distribution")
st.pyplot(fig)

# ==============================
# 📄 DATA PREVIEW
# ==============================
st.subheader("📄 Cleaned Data Sample")
st.dataframe(df.head())
