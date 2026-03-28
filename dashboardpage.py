import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 EMI Dashboard")

# ======================
# LOAD + CLEAN FUNCTION
# ======================
@st.cache_data
def load_clean_data():
    df = pd.read_csv("emi_prediction_dataset.csv")

    num_cols = [
        'monthly_salary',
        'credit_score',
        'current_emi_amount',
        'groceries_utilities',
        'other_monthly_expenses'
    ]

    for col in num_cols:
        df[col] = df[col].astype(str)

        # Fix weird values like 49400.0.0
        df[col] = df[col].str.replace(r'(\d+)\.(\d+)\.(\d+)', r'\1.\2', regex=True)

        # Remove ₹, commas, text
        df[col] = df[col].str.replace(r'[^\d.]', '', regex=True)

        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df


df = load_clean_data()

# ======================
# SIDEBAR FILTERS
# ======================
st.sidebar.header("🔍 Filters")

if 'education' in df.columns:
    edu_filter = st.sidebar.multiselect(
        "Select Education",
        options=df['education'].dropna().unique(),
        default=df['education'].dropna().unique()
    )
    df = df[df['education'].isin(edu_filter)]

# ======================
# KPI SECTION
# ======================
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Avg Salary", f"₹ {int(df['monthly_salary'].mean())}")
col2.metric("Avg EMI", f"₹ {int(df['current_emi_amount'].mean())}")
col3.metric("Avg Credit Score", int(df['credit_score'].mean()))

# ======================
# EMI ELIGIBILITY COUNT
# ======================
if 'emi_eligibility' in df.columns:
    st.subheader("📊 EMI Eligibility Distribution")

    fig1, ax1 = plt.subplots()
    sns.countplot(x='emi_eligibility', data=df, ax=ax1)
    st.pyplot(fig1)

# ======================
# SALARY DISTRIBUTION
# ======================
st.subheader("💰 Salary Distribution")

fig2, ax2 = plt.subplots()
sns.histplot(df['monthly_salary'], kde=True, ax=ax2)
st.pyplot(fig2)

# ======================
# EMI VS SALARY
# ======================
st.subheader("📉 EMI vs Salary")

fig3, ax3 = plt.subplots()
sns.scatterplot(x='monthly_salary', y='current_emi_amount', data=df, ax=ax3)
st.pyplot(fig3)

# ======================
# CORRELATION HEATMAP
# ======================
st.subheader("🔗 Correlation Matrix")

num_cols = [
    'monthly_salary',
    'credit_score',
    'current_emi_amount',
    'groceries_utilities',
    'other_monthly_expenses'
]

fig4, ax4 = plt.subplots()
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
st.pyplot(fig4)

# ======================
# DATA PREVIEW
# ======================
st.subheader("📄 Data Preview")
st.dataframe(df.head())

# ======================
# DEBUG (OPTIONAL)
# ======================
# st.write(df.dtypes)