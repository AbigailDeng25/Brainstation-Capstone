import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# Load the scaler for numerical data
scaler = joblib.load('scaler.pkl')

# Load the pre-trained RandomForest model
with open('random_forest_P.model', 'rb') as f:
    model = pickle.load(f)

# Define mappings for categorical variables
home_ownership_mapping = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 0, 'ANY': 0}
grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
term_mapping = {'36 months': 36, '60 months': 60}

# Streamlit app interface
st.title('Lending Cafe Commercial Loan Default Prediction')

st.write("Please enter the parameters:")

# Collect user input for all parameters
int_rate = st.number_input('Interest Rate', min_value=0.0, max_value=35.0, step=0.01)
loan_amnt = st.number_input('Loan Amount', min_value=0.0, step=1.0)
installment = st.number_input('Installment', min_value=0.0, step=1.0)
annual_inc = st.number_input('Annual Income', min_value=0.0, step=1.0)
delinq_2yrs = st.number_input('Delinquencies in 2 Years', min_value=0, step=1)
fico_range_low = st.number_input('FICO Range Low', min_value=300, max_value=850, step=1)
fico_range_high = st.number_input('FICO Range High', min_value=300, max_value=850, step=1)
inq_last_6mths = st.number_input('Inquiries in Last 6 Months', min_value=0, step=1)
open_acc = st.number_input('Number of Open Accounts', min_value=0, step=1)
pub_rec = st.number_input('Public Record', min_value=0, step=1)
revol_bal = st.number_input('Revolving Balance', min_value=0.0, step=1.0)
total_acc = st.number_input('Total Accounts', min_value=0, step=1)
out_prncp = st.number_input('Outstanding Principal', min_value=0.0, step=1.0)
out_prncp_inv = st.number_input('Outstanding Principal Investment', min_value=0.0, step=1.0)
last_pymnt_amnt = st.number_input('Last Payment Amount', min_value=0.0, step=1.0)
delinq_amnt = st.number_input('Delinquent Amount', min_value=0.0, step=1.0)

# Categorical features
term = st.selectbox('Loan Term', ['36 months', '60 months'])
grade = st.selectbox('Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
home_ownership = st.selectbox('Home Ownership', ['OWN', 'MORTGAGE', 'RENT', 'OTHER', 'ANY'])

# Convert categorical features into numerical format using mappings
term_num = term_mapping[term]
grade_num = grade_mapping[grade]
home_ownership_num = home_ownership_mapping[home_ownership]

# Prepare input data
input_data = pd.DataFrame([[int_rate, loan_amnt, installment, annual_inc, delinq_2yrs, fico_range_low, fico_range_high,
                            inq_last_6mths, open_acc, pub_rec, revol_bal, total_acc, out_prncp, out_prncp_inv,
                            last_pymnt_amnt, delinq_amnt, term_num, grade_num, home_ownership_num]],
                          columns=['int_rate', 'loan_amnt', 'installment', 'annual_inc', 'delinq_2yrs', 'fico_range_low',
                                   'fico_range_high', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc',
                                   'out_prncp', 'out_prncp_inv', 'last_pymnt_amnt', 'delinq_amnt', 'term', 'grade', 'home_ownership'])

# Define the preprocessing steps for numerical and categorical data
numerical_columns = ['int_rate', 'loan_amnt', 'installment', 'annual_inc', 'delinq_2yrs', 'fico_range_low',
                     'fico_range_high', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'out_prncp',
                     'out_prncp_inv', 'last_pymnt_amnt', 'delinq_amnt']

categorical_columns = ['term', 'grade', 'home_ownership']

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

# Now we create the full pipeline with pre-processing and prediction without needing to retrain
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)  # Directly use the pre-trained model
])

# Make prediction when the user clicks the 'Predict' button
if st.button('Predict'):
    # Use the pipeline to predict
    prediction = pipeline.predict(input_data)
    probabilities = pipeline.predict_proba(input_data)
    
    # Output the prediction
    st.write(f'Predicted class: {prediction[0]}')
    st.write(f'Probability of default (class = 1): {probabilities[0][1]:.2f}')