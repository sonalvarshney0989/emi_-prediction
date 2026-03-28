import streamlit as st

st.set_page_config(page_title="EMI Prediction AI", layout="wide")

st.title("💳 EMI Prediction AI System")

st.markdown("""
### 📌 Features:
- EMI Eligibility Prediction
- Maximum EMI Estimation
- Financial Risk Analysis
- MLflow Model Tracking
""")

import streamlit as st

st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", [
    "Dashboard",
    "EDA",
    "Prediction",
    "Model Performance",
    "MLflow"
])
if page == "Dashboard":
    import dashboardpage 

elif page == "EDA":
    import eda

elif page == "Prediction":
    import prediction

elif page == "Model Performance":
    import modelperformance

elif page == "MLflow":
    import mlflow_page

