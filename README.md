📊 EMI Eligibility Prediction & Dashboard

This project is a complete Machine Learning + Streamlit Dashboard application that predicts whether a user is eligible for EMI (Equated Monthly Installment) based on financial details.

It also provides a Model Performance Dashboard with evaluation metrics and visualizations.

🚀 Features
✅ EMI Eligibility Prediction (Yes/No)
✅ Clean and interactive Streamlit UI
✅ Data preprocessing & cleaning pipeline
✅ Model training using Logistic Regression
✅ Model performance metrics:
Accuracy
Precision
Recall
F1 Score
✅ Visualization of prediction distribution
✅ Error handling for real-world messy data
🧠 Machine Learning Workflow
Data Cleaning (handling missing & invalid values)
Feature Selection:
Monthly Salary
Credit Score
Current EMI Amount
Model Training:
Logistic Regression
Model Evaluation:
Accuracy, Precision, Recall, F1 Score
Deployment:
Streamlit Dashboard
🛠️ Tech Stack
Python 🐍
Pandas
NumPy
Scikit-learn
Matplotlib
Streamlit
📁 Project Structure
├── app2.py                  # Main Streamlit app
├── modelperformance.py      # Model performance dashboard
├── model.pkl               # Trained model
├── emi_prediction_dataset.csv
├── requirements.txt
└── README.md
▶️ How to Run the Project
1️⃣ Clone Repository
git clone https://github.com/your-username/emi-prediction.git
cd emi-prediction
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Streamlit App
streamlit run app2.py
📊 Dashboard Preview
📈 Model performance metrics
📉 Prediction distribution chart
📄 Cleaned dataset preview
⚠️ Challenges Solved
Handling invalid numeric values (e.g., 127000.0.0)
Fixing missing values (NaN issues)
Feature mismatch errors
Real-time data cleaning inside Streamlit
