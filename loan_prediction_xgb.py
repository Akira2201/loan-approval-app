import json
import xgboost as xgb
import numpy as np
import pandas as pd
import streamlit as st

# Load the trained model
model = xgb.Booster()
model.load_model("xgb_model_new.json")

# Streamlit UI
st.title("Loan Approval Prediction App")
st.write("Nhập thông tin khách hàng để dự đoán kết quả khoản vay.")

# Tạo form nhập dữ liệu
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income", min_value=0, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed"])
education_level = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
experience = st.number_input("Experience (years)", min_value=0, value=5)
loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
loan_duration = st.number_input("Loan Duration (months)", min_value=1, value=12)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
number_of_dependents = st.number_input("Number of Dependents", min_value=0, value=0)
home_ownership = st.selectbox("Home Ownership Status", ["Rent", "Own", "Mortgage"])
monthly_debt = st.number_input("Monthly Debt Payments", min_value=0.0, value=500.0)
credit_utilization = st.number_input("Credit Card Utilization Rate", min_value=0.0, max_value=1.0, value=0.3)
num_credit_lines = st.number_input("Number of Open Credit Lines", min_value=0, value=5)
num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, value=1)
dti_ratio = st.number_input("Debt to Income Ratio", min_value=0.0, max_value=1.0, value=0.2)
bankruptcy_history = st.selectbox("Bankruptcy History", [0, 1])
loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Business", "Education"])
previous_defaults = st.selectbox("Previous Loan Defaults", [0, 1])
payment_history = st.number_input("Payment History Score", min_value=0, max_value=100, value=80)
length_credit_history = st.number_input("Length of Credit History (years)", min_value=0, value=10)
savings_balance = st.number_input("Savings Account Balance", min_value=0, value=5000)
checking_balance = st.number_input("Checking Account Balance", min_value=0, value=2000)
total_assets = st.number_input("Total Assets", min_value=0, value=100000)
total_liabilities = st.number_input("Total Liabilities", min_value=0, value=50000)
monthly_income = st.number_input("Monthly Income", min_value=0, value=4000)
utility_bills = st.number_input("Utility Bills Payment History", min_value=0.0, max_value=1.0, value=0.8)
job_tenure = st.number_input("Job Tenure (years)", min_value=0, value=5)
net_worth = st.number_input("Net Worth", min_value=-50000, value=50000)
base_interest_rate = st.number_input("Base Interest Rate", min_value=0.0, max_value=1.0, value=0.05)
interest_rate = st.number_input("Interest Rate", min_value=0.0, max_value=1.0, value=0.07)
monthly_loan_payment = st.number_input("Monthly Loan Payment", min_value=0.0, value=500.0)
total_dti_ratio = st.number_input("Total Debt to Income Ratio", min_value=0.0, max_value=1.0, value=0.3)

# Chuyển dữ liệu vào dictionary
input_data = {
    "Age": age,
    "AnnualIncome": annual_income,
    "CreditScore": credit_score,
    "EmploymentStatus": employment_status,
    "EducationLevel": education_level,
    "Experience": experience,
    "LoanAmount": loan_amount,
    "LoanDuration": loan_duration,
    "MaritalStatus": marital_status,
    "NumberOfDependents": number_of_dependents,
    "HomeOwnershipStatus": home_ownership,
    "MonthlyDebtPayments": monthly_debt,
    "CreditCardUtilizationRate": credit_utilization,
    "NumberOfOpenCreditLines": num_credit_lines,
    "NumberOfCreditInquiries": num_credit_inquiries,
    "DebtToIncomeRatio": dti_ratio,
    "BankruptcyHistory": bankruptcy_history,
    "LoanPurpose": loan_purpose,
    "PreviousLoanDefaults": previous_defaults,
    "PaymentHistory": payment_history,
    "LengthOfCreditHistory": length_credit_history,
    "SavingsAccountBalance": savings_balance,
    "CheckingAccountBalance": checking_balance,
    "TotalAssets": total_assets,
    "TotalLiabilities": total_liabilities,
    "MonthlyIncome": monthly_income,
    "UtilityBillsPaymentHistory": utility_bills,
    "JobTenure": job_tenure,
    "NetWorth": net_worth,
    "BaseInterestRate": base_interest_rate,
    "InterestRate": interest_rate,
    "MonthlyLoanPayment": monthly_loan_payment,
    "TotalDebtToIncomeRatio": total_dti_ratio
}

# Function to preprocess input data
def preprocess_input(data):
    df = pd.DataFrame([data])
    categorical_columns = ["EmploymentStatus", "EducationLevel", "MaritalStatus", "HomeOwnershipStatus", "LoanPurpose"]
    for col in categorical_columns:
        df[col] = df[col].astype("category").cat.codes
    df = df.astype(float)
    return df

# Create a button to trigger prediction
if st.button("Predict"):
    try:
        processed_data = preprocess_input(input_data)
        dmatrix = xgb.DMatrix(processed_data)
        prediction = model.predict(dmatrix)[0]
        result = "Loan Approved!" if prediction >= 0.5 else "Loan Rejected."
        st.write(result)  # Chỉ hiển thị kết quả mà không có màu sắc thành công/lỗi
    except Exception as e:
        pass  # Ẩn thông báo lỗi
