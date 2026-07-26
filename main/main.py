import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import shap
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, f1_score
import datetime
import os
import joblib
from preprocessing import get_data_source, clean_data
from train_model import train_eva

# ==================================================================================================================================================
# C. GUI & MAIN LOGIC
# ==================================================================================================================================================

# Main config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide", initial_sidebar_state="expanded")

# 1. Load Data (Upload or Load local file)
df_input = get_data_source()

# 2. Validation: Stop execution if no data is provided
if df_input is None:
    st.info("Please upload a dataset to proceed.")
    st.stop()

# 3. Data Cleaning & Pipeline Execution (This function is cached for performance)
@st.cache_resource
def run_pipeline(df):
    dataset_raw, dataset = clean_data(df)
    y = dataset['Churn'].map({'Yes': 1, 'No': 0})
    X = dataset.drop(['Churn', 'customerID'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Persistence: Check if trained models exist on disk to save time
    pkl_check = 'all_mod.pkl'
    if os.path.exists(pkl_check):
        results = joblib.load(pkl_check)
        # Select the best model based on F1 Score
        best_model_name = max(results, key=lambda x: results[x]['f1'])
        st.sidebar.success("Loaded models from disk")
    else:
        # Train models if no file is found
        results, best_model_name = train_eva(X_train, y_train, X_test, y_test)
        st.sidebar.warning("Models trained and saved for the first time!")
        
    return results, best_model_name, X_train, X_test, y_train, y_test, dataset_raw, dataset

with st.spinner("Training models in progress..."):
    results, best_model_name, X_train, X_test, y_train, y_test, dataset_raw, dataset = run_pipeline(df_input)
    
    best_res = results[best_model_name]
    pipeline = best_res['pipeline']
    model_columns = best_res['model_columns']

# Sidebar
st.sidebar.title("Churn Prediction Project")

page = st.sidebar.radio("Navigation:", 
    ["Homepage", "Dataset Check", "Model Comparison", "Model Evaluation", "Prediction", "Business ROI", "References"])
st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Tool")
threshold = st.sidebar.slider("Probability Threshold", min_value=0.1, max_value=0.9, step=0.05, value=0.35, help="Adjust to see how the threshold affects metrics on Model Evaluation page")

# ==================================================================================================================================================
# Page 0: Homepage
# ==================================================================================================================================================
if page == "Homepage":
    st.header("Homepage and Manual")
    st.write ("Welcome to Customer Churn Prediction App, this is the homepage.")
    st.subheader("Project Workflow and Features")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Exploration and Selection")
        st.write ("""
        - Dataset Check: Generate a full report to analyze all features of the dataset using ydata-profiling.
        - Model Comparison: Evaluates multiple algorithms (Logistic Regression, Random Forest, XGBoost) using GridSearchCV to find the optimal champion model based on F1-Score.
        """)
    
    with col2:
        st.markdown("Evaluation and Prediction")
        st.write ("""
        - Model Evaluation: View model performance results, including accuracy, ROC-AUC scores, and confusion matrix plots.
        - Prediction: Input real-time customer data to get a churn probability. Send Email (if customer churn) and view SHAP explanations.
        """)
    st.divider()
    st.markdown("Business Impact")
    st.write("""
    The Business ROI module translates technical metrics (True Positives, False Positives) 
    into actual dollars saved, helping management understand the financial benefit of deploying this AI model.
    """)

    # Quick Start Guide
    st.info("Quick Start: Use the sidebar on the left to navigate. Start with Dataset Check to understand the data, or jump to Prediction to test the AI model.")

# ==================================================================================================================================================
# Page 1: Dataset Check
# ==================================================================================================================================================  
elif page == "Dataset Check":
    st.header("Explore The Dataset (cleaned) using Ydata-profiling")
    if st.button("Start To Analyze", key="cleaned"):
        with st.spinner("Loading the report"):
            profile = ProfileReport(dataset_raw, minimal=True)
            components.html(profile.to_html(), height=800, scrolling=True)
    
# ==================================================================================================================================================
# Page 2: Model Comparison
# ==================================================================================================================================================  
elif page == "Model Comparison":
    st.header("Model Comparison")
    st.write("This section compares key performance metrics across all trained algorithms to identify the best model.")
    
    # Constructing the performance summary table
    comp_dataset = pd.DataFrame({
        "Algorithm": results.keys(),
        "F1-Score": [f"{v['f1']:.4f}" for v in results.values()],
        "ROC-AUC": [f"{v['roc']:.4f}" for v in results.values()],
        "Accuracy": [f"{v['acc']:.4f}" for v in results.values()]
    })
    
    st.subheader("Performance Summary Table")
    st.dataframe(
        comp_dataset.style.format(precision=2)
        .highlight_max(axis=0, color='#1f77b4', subset=['F1-Score', 'ROC-AUC', 'Accuracy'])
    )    
    # Display performance metrics
    st.subheader("Best Hyperparameters")
    for name in results.keys():
        with st.expander(f"View best parameters for {name}"):
            st.write(results[name].get('best_params', "Model trained using default parameters"))

    st.success(f"Best Model: {best_model_name}")

# ==================================================================================================================================================
# Page 3: Model Evaluation
# ==================================================================================================================================================  
elif page == "Model Evaluation":
    st.header(f"Detailed Evaluation: {best_model_name}")
    
    # Model selection dropdown (defaults to the best model)
    selected_model_name = st.selectbox(
        "Select model for detailed analysis:", 
        list(results.keys()), 
        index=list(results.keys()).index(best_model_name)
    )
    
    res = results[selected_model_name]
    st.write(f"Displaying technical metrics for {selected_model_name} at a classification threshold of {threshold}.")

    # 1. Classification Report
    st.subheader(f"1. Classification Report (Default Threshold = 0.5)")
    
    y_default = (res['y_prob'] >= 0.5).astype(int)
    # Generate a fresh classification report based on the selected threshold
    report_og = classification_report(y_test, y_default, output_dict=True)
    report_og_df = pd.DataFrame(report_og).transpose()
    
    # Display the dataframe with highlighting for better readability
    st.dataframe(report_og_df.style.highlight_max(axis=0, color='#1f77b4'))
    
    # 2. Dynamic Classification Report
    st.subheader(f"2. Dynamic Classification Report (Threshold = {threshold})")
    
    # Calculate predictions based on the dynamic threshold from the sidebar
    y_custom = (res['y_prob'] >= threshold).astype(int)
    
    # Generate a fresh classification report based on the selected threshold
    custom_report = classification_report(y_test, y_custom, output_dict=True)
    custom_report_df = pd.DataFrame(custom_report).transpose()
    
    # Display the dataframe with highlighting for better readability
    st.dataframe(custom_report_df.style.highlight_max(axis=0, color='#1f77b4'))
    st.caption("Note: Precision, Recall, and F1-score update instantly when you adjust the Threshold slider in the sidebar.")
    
    # 3. Metrics Display
    st.subheader ("3. Metrics of Model")
    m1, m2, m3, m4 = st.columns(4)
    
    y_custom = (res['y_prob'] >= threshold).astype(int)
    current_f1 = f1_score(y_test, y_custom)
    current_acc = accuracy_score(y_test, y_custom)
    
    m1.metric("Accuracy", f"{current_acc:.2%}")
    m2.metric("F1-Score", f"{current_f1:.2%}")
    m3.metric("ROC-AUC", f"{res['roc']:.2%}")
    m4.metric("Optimal Threshold", f"{res['optimal_threshold']:.2f}")
    
    # 4. Confusion Matrix & ROC Curve
    st.subheader("4. Performance Visualization")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Confusion Matrix")
        fig_cm, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_custom)
        ConfusionMatrixDisplay(cm, display_labels=['Stay', 'Churn']).plot(ax=ax, cmap='Blues', values_format='d')
        st.pyplot(fig_cm)
        st.caption("The Confusion Matrix visualizes correct vs. incorrect predictions (False Negatives are often the most critical for churn).")
    with col2:
        st.write("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {res['roc']:.2f}")
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
        st.caption("The ROC Curve demonstrates the model's ability to distinguish between churning and loyal customers.")
    # 5. Top 10 Feature Importance
    st.divider()
    st.subheader(f"5. Top 10 Feature Importance - {selected_model_name}")
    
    # Access the classifier object inside the Pipeline
    model_obj = res['pipeline'].named_steps['clf']
    
    # Check model type to retrieve the correct importance weights (handling coef_ vs feature_importances_)
    importance = None
    if hasattr(model_obj, 'coef_'):
        # Logic for Logistic Regression
        importance = np.abs(model_obj.coef_[0])
    elif hasattr(model_obj, 'feature_importances_'):
        # Logic for Tree-based models (Random Forest, XGBoost)
        importance = model_obj.feature_importances_
        
    if importance is not None:
        feat_imp = pd.Series(importance, index=res['model_columns']).sort_values(ascending=False).head(10)        
        fig_feat, ax_feat = plt.subplots(figsize=(10, 6))
        feat_imp.plot(kind='barh', ax=ax_feat, color='#2ca02c')
        ax_feat.invert_yaxis()
        ax_feat.set_title(f"Key Drivers for Churn Prediction ({selected_model_name})")
        st.pyplot(fig_feat)
        st.info("This chart highlights which specific customer features have the strongest impact on the model's prediction.")
    else:
        st.warning("Feature importance is not directly available for this model type.")
 
# ==================================================================================================================================================
# Page 4: Prediction
# ================================================================================================================================================== 
elif page == "Prediction":
    st.header("Prediction and Retention Strategy Simulation")
    
    # Retrieve the optimal threshold calculated during model training
    opt_t = best_res['optimal_threshold']
    
    # 1. Initialize input form for data entry
    with st.form("churn_form_full"):
        st.subheader("Customer Data Input")
        customer_name = st.text_input("Customer Name", value="Valued Customer")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.checkbox("Senior Citizen")
            partner = st.checkbox("Partner")
            dependents = st.checkbox("Dependents")
            tenure = st.slider("Tenure (Months)", 0, 72, 12)

        with col2:
            st.markdown("Telecom Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

        with col3:
            st.markdown("Contract and Billing")
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
            
        submit = st.form_submit_button("Run AI Analysis")

    # 2. Process logic after Form submission
    if submit:
        # Create a dictionary for the input features
        raw_input_dict = {
            'gender': [gender], 'SeniorCitizen': [1 if senior else 0], 'Partner': ['Yes' if partner else 'No'],
            'Dependents': ['Yes' if dependents else 'No'], 'tenure': [tenure], 'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines], 'InternetService': [internet_service], 'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup], 'DeviceProtection': [device_protection], 'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv], 'StreamingMovies': [streaming_movies], 'Contract': [contract],
            'PaperlessBilling': [paperless], 'PaymentMethod': [payment_method], 
            'MonthlyCharges': [monthly_charges], 'TotalCharges': [total_charges]
        }
        
        # Convert dictionary to DataFrame and align columns for the model
        input_dataset = pd.DataFrame(raw_input_dict)
        
        # Generate churn probability prediction
        prob = pipeline.predict_proba(input_dataset)[:, 1][0]
        
        st.divider()
        st.subheader("Analysis Result")
        
        # Evaluation based on the pre-calculated optimal threshold
        if prob >= opt_t:
            st.error(f"Churn Probability: {prob:.2%} -> HIGH RISK (Optimal Threshold: {opt_t:.2f})")
            st.write("Action Required: This customer is likely to churn. Consider immediate retention offers.")
        else:
            st.success(f"Churn Probability: {prob:.2%} -> SAFE (Optimal Threshold: {opt_t:.2f})")
            st.write("Status: Customer is likely to stay loyal.")

        st.progress(prob)

        # 3. What-if Simulation and SHAP Interpretability
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("AI Recommendations")
            if prob >= opt_t:
                if contract == "Month-to-month":
                    st.warning("Strategy: Offer 1-year contract voucher to improve stability.")
                if tech_support == "No":
                    st.info("Strategy: Offer 1-month free trial of Tech Support service.")
            else:
                st.success("Strategy: Maintain current relationship and focus on up-selling.")

        with col_right:
            st.subheader("Why did AI predict this?")
            with st.spinner("Calculating SHAP values"):

                # 1) Extract model and preprocessor
                model_part = pipeline.named_steps['clf']
                preprocessor = pipeline.named_steps['preprocessor']

                # 2) Transform with preprocessor (pipeline)
                X_train_summary = preprocessor.transform(X_train.sample(100, random_state=42))
                input_trans = preprocessor.transform(input_dataset)

                # 3) SHAP explainer
                if best_model_name == "Logistic Regression":
                    explainer = shap.LinearExplainer(model_part, X_train_summary)
                else:
                    explainer = shap.TreeExplainer(model_part)

                shap_values = explainer.shap_values(input_trans)

                # 4) Normalize shap_values
                if isinstance(shap_values, list):
                    current_shap = shap_values[1][0]
                elif len(shap_values.shape) == 3:
                    current_shap = shap_values[0, :, 1]
                else:
                    current_shap = shap_values[0]

                # 5) Base value
                base_value = explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[1]

                # 6) Plot
                fig_shap, ax_shap = plt.subplots()
                exp = shap.Explanation(
                    values=current_shap,
                    base_values=base_value,
                    data=input_trans[0],
                    feature_names=model_columns.tolist()
                )
                shap.plots.bar(exp, max_display=8, show=False)
                st.pyplot(fig_shap)


        # 4. Email Campaign Generation
        st.divider()
        st.subheader("AI-Powered Email Draft")
        today = datetime.date.today().strftime("%B %d, %Y")
        
        if prob >= opt_t:
            if internet_service == 'Fiber optic':
                email_subject = "Special Service Update"
                reason_text = "Ensuring your Fiber Optic connection is working perfectly."
                offer_text = "3 months of Premium Tech Support for FREE."
            elif contract == 'Month-to-month':
                email_subject = "Exclusive Reward for Loyalty"
                reason_text = "We appreciate your flexibility with us."
                offer_text = "20% OFF your monthly bill by switching to a 1-year plan today."
            else:
                email_subject = "A Gift for our Valued Customer"
                reason_text = "We value your continued business."
                offer_text = "A $25 loyalty credit has been added to your account."

            body = f"Dear {customer_name},\n\n{reason_text}\n\nTo show our appreciation, {offer_text}\n\nReply YES to activate.\n\nBest regards,\nTelco Team\nDate: {today}"
            st.text_area("Retention Email Draft:", value=body, height=200)
            if st.button("Send Retention Email"):
                st.success(f"Retention email sent to {customer_name} successfully")
        else:
            upsell_body = f"Dear {customer_name},\n\nAs one of our loyal customers, enjoy 1 month of Premium Streaming for free!\n\nBest regards,\nTelco Team"
            st.text_area("Upsell Email Draft:", value=upsell_body, height=150)
            if st.button("Send Upsell Offer"):
                st.success("Upsell offer sent successfully")
# ==================================================================================================================================================
# Page 5: Business ROI
# ==================================================================================================================================================  
elif page == "Business ROI":
    st.header("Business ROI Analysis")
    
    col_a, col_b = st.columns(2)
    with col_a:
        avg_val = st.number_input("Average Customer Lifetime Value ($)", 100, 5000, 500)
        cost_ret = st.number_input("Retention Cost per Customer ($)", 10, 500, 50)
    with col_b:
        success_rate = st.slider("Retention Success Rate (%)", 0, 100, 30) / 100

    # Calculate metrics based on Threshold
    y_prob_best = best_res['y_prob']
    y_custom = (y_prob_best >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_custom).ravel()

    # Financial Formulas
    loss_no_model = (tp + fn) * avg_val
    saved_rev = (tp * success_rate) * avg_val
    campaign_cost = (tp + fp) * cost_ret
    net_profit = saved_rev - campaign_cost

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Customers Saved", f"{int(tp * success_rate)} people")
    m2.metric("Retained Revenue", f"${saved_rev:,.0f}")
    m3.metric("Net Profit", f"${net_profit:,.0f}", delta=f"{(net_profit/campaign_cost)*100:.1f}% ROI" if campaign_cost > 0 else "0%")

    # Visualize ROI
    fig_roi, ax_roi = plt.subplots()
    
    labels = ['Loss (No Model)', 'Net Profit (With Model)']
    values = [loss_no_model, net_profit]
    colors = ['#E63946', '#2A9D8F'] 
    
    bars = ax_roi.bar(labels, values, color=colors)
    ax_roi.set_ylabel("Financial Value ($)")
    ax_roi.set_title("Return on Investment Analysis")
    
    for bar in bars:
        height = bar.get_height()
        ax_roi.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom', fontweight='bold')
                
    st.pyplot(fig_roi)

# ==================================================================================================================================================
# Page 5: Business ROI
# ==================================================================================================================================================  
elif page == "References":
    st.header("Project References & Credits")
    st.write("This project was built using resources from the following expert blogs, tutorials, and documentation.")

    st.subheader("Technical Articles & Blogs")
    st.markdown("""
    - **D. Breton**, "A Full Guide on Choosing the Right Machine Learning Algorithm," *Medium*. [Link](https://medium.com/@davidbreton03/a-full-guide-on-choosing-the-right-machine-learning-algorithm-5fa282a0b2a1)
    - **K. Naminas**, "Machine Learning Algorithm: When to Use Which One," *LabelYourData*. [Link](https://labelyourdata.com/articles/how-to-choose-a-machine-learning-algorithm)
    - **B. Prasad**, "Telecom Customer Churn Prediction: Data Preprocessing & Analysis," *Kaggle*. [Link](https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction#-7.-Data-Preprocessing)
    - **S. Ray**, "8 Ways to Improve Accuracy of Machine Learning Models," *Analytics Vidhya*. [Link](https://www.analyticsvidhya.com/blog/2015/12/improve-machine-learning-results/)
    - **S. Tripathi**, "Pandas Profiling / YData Profiling in Python Guide," *DataCamp*. [Link](https://www.datacamp.com/tutorial/pandas-profiling-ydata-profiling-in-python-guide)
    - **V. S. Wijesinghe**, "Explaining Random Forest Model with Shapely Values," *Kaggle*. [Link](https://www.kaggle.com/code/vikumsw/explaining-random-forest-model-with-shapely-values)
    - **GeeksforGeeks**, "Machine Learning Pipeline," *GeeksforGeeks Blog*. [Link](https://www.geeksforgeeks.org/blogs/machine-learning-pipeline/)
    - **F. Pedregosa et al.**, "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*. [Official Docs](https://scikit-learn.org/stable/)
    - **XGBoost Developers**, "XGBoost Documentation: Get Started with XGBoost." [Official Docs](https://xgboost.readthedocs.io/en/stable/get_started.html)
    - **S. Lundberg**, "SHAP (SHapley Additive exPlanations) Documentation." [ReadTheDocs](https://shap.readthedocs.io)
    - **SHAP Examples**, "Sentiment Analysis with Logistic Regression (Linear Models)." [Tutorial](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/linear_models/Sentiment%20Analysis%20with%20Logistic%20Regression.html)
    - **R. Duarte**, "Class Weight, SMOTE, Random Over and Under Sampling," *Medium*. [Link](https://medium.com/@rafaelnduarte/class-weight-smote-random-over-and-under-sampling-bca603378e02)
    - **Greg J**, "Using Class Weight to Compensate for Imbalanced Data," *Medium*. [Link](https://medium.com/@bubbapora_76246/using-class-weight-to-compensate-for-imbalanced-data-6eff370185d3)
    - **GeeksforGeeks**, "SMOTE for Imbalanced Classification with Python." [Link](https://www.geeksforgeeks.org/machine-learning/smote-for-imbalanced-classification-with-python/)
    """)


    st.divider()
    st.subheader("Video Tutorials & YouTube Channels")
    st.markdown("""
    - **Hackers Realm**, "Customer Churn Prediction Analysis | Classification | Python," *YouTube*. [Watch Video](https://www.youtube.com/watch?v=40N9zFKrj_s)
    - **Code with Josh**, "Data Profiling with YData Profiling," *YouTube*. [Watch Video](https://www.youtube.com/watch?v=777Qb0gHuJU)
    - **DATA JARVIS**, "Customer Churn Prediction using Streamlit and Scikit-Learn," *YouTube*. [Watch Video](https://www.youtube.com/watch?v=yuBcZynmJzI)
    """)

    st.divider()
    st.subheader("AI Support & Optimization")
    st.info("""
    **OpenAI ChatGPT-4o / Google Gemini**: This AI was utilized as a collaborative tool for the following purposes:
    - **Suggestions & Structure:** Provided architectural recommendations (MVC pattern) and structured the technical report.
    - **Debugging & Optimization:** Assisted in code review, error handling, and performance tuning for the Streamlit app.
    - **SHAP Implementation:** Aided in developing the logic for `LinearExplainer` and `TreeExplainer` to ensure model transparency.
    - **Business ROI Modeling:** Helped formulate the mathematical logic for financial simulations, translating technical metrics into business value.
    """)
    
    st.divider()
    st.caption("Submitted by Bao Phuoc Quy Tai & Khuong Ho Anh Duc | Instructor: Mr. Pole")
