# ==================================================================================================================================================
# A. PREPROCESSING & PIPELINE SETUP
# ==================================================================================================================================================
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_data_source():
    # Create Upload File for different file name and in case of system cannot read the dataset we hard code
    # Purpose: this use to test newly dataset but must have the EXACT SAME features
    uploaded_file = st.sidebar.file_uploader("Change Dataset (Optional)", type="csv", key="sidebar_uploader", help="This function just use in case you change the dataset name. The dataset must be the same but different name is acceptable!")
    
    # Handle read csv file if uploaded_file doesnt exist
    # -->Use provided csv file 
    # -->If cannot read, warning then upload file
    if uploaded_file is not None:
        st.sidebar.success("Using uploaded dataset!")
        return pd.read_csv(uploaded_file)
    try:
        dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        return dataset
    except FileNotFoundError:
        st.warning("Default dataset not found. Please upload your CSV file.")
        main_upload = st.file_uploader("Upload Dataset Here", type="csv", key="main_upload")
        if main_upload is not None:
            return pd.read_csv(main_upload)
        else:
            return None

@st.cache_data
def clean_data(df):
    # Differentiate 2 data for the comparison in datacheck part
    dataset_raw = df.copy()
    dataset = df.copy()
    
    # Handle wrong data type in TotalCharges
    dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce').fillna(0.0)
    return dataset_raw, dataset

# Encapsulate Scaling and Encoding in Pipeline
# Prevent Data Leakeage, Production-Like, Maintainable
def create_pipeline(model_instance):
    # Numerical feats for scaling
    num_features = ['tenure','MonthlyCharges','TotalCharges']
    # Categorical feats for encoding
    cat_features = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']
    
    # use columntransformer to handle different data type smooth
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_features)
        ],
    )
    
    # Build Pipeline
    return Pipeline([
        ('preprocessor', preprocessor),
        ('clf', model_instance)
    ])
