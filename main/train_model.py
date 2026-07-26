# ==================================================================================================================================================
# B. Train and Evaluate Model
# ===================================================================================================================================================
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from preprocessing import create_pipeline

@st.cache_resource
def train_eva(_X_train, _y_train, _X_test, _y_test):
    # Calculate im ratio for XGboost imbalance handling
    num_yes = np.sum(_y_train == 1)
    num_no = np.sum(_y_train == 0)
    imbalance_ratio = float(num_no) / num_yes if num_yes > 0 else 1.0
    # Define hyperparameter for algorithm tunning
    param_grids = {
        "Logistic Regression": {
            'clf__C': [0.1, 1, 10],
            'clf__solver': ['lbfgs', 'liblinear']
        },
        "Random Forest": {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 10, 20],
            'clf__min_samples_split': [2, 5]
        },
        "XGBoost": {
            'clf__n_estimators': [100, 200],
            'clf__learning_rate': [0.01, 0.1],
            'clf__max_depth': [3, 6],
            'clf__scale_pos_weight': [1.0, imbalance_ratio]
        }
    }

    # Initialize models using dict
    # Instead of using SMOTE, we handle imbalanced directly using class_weight and eval_metric
    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    results = {}
    
    # Loop for each model
    for name, model in base_models.items():
        pipe = create_pipeline(model) 
        
        # GridSearchCV setup
        grid_search = GridSearchCV(
            estimator=pipe, # pass the entire pipeline to prevent data leakage
            param_grid=param_grids[name], 
            cv=3, 
            scoring='f1', 
            n_jobs=-1 # maximize CPU
        )
        # model train and find best params 
        grid_search.fit(_X_train, _y_train)
        
        # take best params of each model and store
        best_pipe = grid_search.best_estimator_
        
        # Prediction
        y_pred = best_pipe.predict(_X_test)
        # calculate result in prob but select only churn column
        y_prob = best_pipe.predict_proba(_X_test)[:, 1]
        
        # Find Optimal threshold for F1 (normally default = 0.5)
        threshold = np.arange(0.1, 0.9, 0.01)
        # loop all threshold then compare with test and calculate f1_score
        f1_scores = [f1_score(_y_test,(y_prob >= t).astype(int)) for t in threshold]
        opt_threshold = threshold[np.argmax(f1_scores)]
        
        # save relsult
        results[name] = {
            "pipeline": best_pipe,
            #use for shap
            "model_columns": best_pipe.named_steps['preprocessor'].get_feature_names_out(), #take new column from one-hot encode from preprocessor
            "best_params": grid_search.best_params_,
            "acc": accuracy_score(_y_test, y_pred),
            "roc": roc_auc_score(_y_test, y_prob),
            "f1": f1_score(_y_test, y_pred),
            "f1_optimal": max(f1_scores),
            "report": classification_report(_y_test, y_pred, output_dict=True),
            "y_pred": y_pred,
            "y_prob": y_prob,
            "optimal_threshold": opt_threshold
        }
    
    # Choose best model by f1
    # max() use alphabet to take results, so we have to use lambda to extract f1 and find max
    best_name = max(results, key=lambda best: results[best]['f1'])
    
    # save model to .pkl
    joblib.dump(results[best_name], 'best_mod.pkl')
    joblib.dump(results,'all_mod.pkl')
    
    return results, best_name
