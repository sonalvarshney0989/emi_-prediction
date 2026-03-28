import streamlit as st
import mlflow
import pandas as pd

st.title("🧠 MLflow Experiment Tracking")

# ======================
# MLflow Setup
# ======================
mlflow.set_tracking_uri("http://127.0.0.1:5000")

st.success("Connected to MLflow Server ✅")

# ======================
# SHOW LINK
# ======================
st.markdown("### 🔗 Open Full MLflow UI")
st.markdown("[Click here to open MLflow](http://127.0.0.1:5000)")

# ======================
# LOAD EXPERIMENTS
# ======================
try:
    experiments = mlflow.search_experiments()

    exp_data = []
    for exp in experiments:
        exp_data.append({
            "Experiment Name": exp.name,
            "Experiment ID": exp.experiment_id
        })

    df_exp = pd.DataFrame(exp_data)

    st.subheader("📊 Available Experiments")
    st.dataframe(df_exp)

except Exception as e:
    st.error(f"Error loading experiments: {e}")

# ======================
# LOAD RUNS
# ======================
st.subheader("📈 Experiment Runs")

try:
    runs = mlflow.search_runs() 

    st.dataframe(runs.head(10))

except Exception as e:
    st.error(f"Error loading runs: {e}")