
from sklearn import metrics

import pandas as pd
import os
from huggingface_hub import login
from huggingface_hub import hf_hub_download


# for data preprocessing and pipeline creation
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score


# for model serialization
import joblib

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow



mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_CICD_Predictive_Maintenance")


# Define repo and filenames
repo_id = "sudharshanc/predictive-vehicle-maintenance"

Xtrain_path = hf_hub_download(repo_id=repo_id,repo_type="dataset", filename="Xtrain.csv")
Xtest_path  = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="Xtest.csv")
ytrain_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="ytrain.csv")
ytest_path  = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="ytest.csv")


Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


    
# Grid of parameters to choose from
parameters = {
    "xgbclassifier__n_estimators": [100, 150],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__subsample": [0.8],
    "xgbclassifier__colsample_bytree": [0.8],
    "xgbclassifier__gamma": [1, 5],
    "xgbclassifier__reg_alpha": [0, 1, 10],
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score, pos_label=1)

# Define which columns to scale
sensor_cols = ['Engine rpm', 'Fuel pressure', 
                'lub oil temp K', 'Coolant temp K',
                'Coolant pressure','Lub oil pressure',
               'Lub_oil_p_t_ratio', 'Coolant_p_t_ratio']  

# Set up the preprocessor

## Scaling as columns are representing different physical measurements(scale)
# Scaling AFTER splitting to prevent data leakage
#Picked RobustScaler as it is robust to outliers and our data has outliers as seen in the histograms plotted above.

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), sensor_cols)
    ]
)


# Create the complete pipeline
xgb_pipe_tuned = make_pipeline(
    preprocessor,
    XGBClassifier(random_state=1, verbosity=0)
)

mlflow.end_run()

# Start a parent run for the Grid Search activity
with mlflow.start_run(run_name="XGB_Engine_GridSearch"):
    
    # 1. Initialize and Run Grid Search
    grid_obj = GridSearchCV(xgb_pipe_tuned, parameters, scoring=scorer, cv=5, n_jobs=-1)
    grid_obj.fit(Xtrain, ytrain)
    
    # 2. Log every parameter combination as a Nested Run
    results = grid_obj.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True, run_name=f"Trial_{i}"):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("cv_mean_recall", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # 3. Finalize the Best Model
    best_model_xgb = grid_obj.best_estimator_
    mlflow.log_params(grid_obj.best_params_)
    mlflow.log_metric("best_cv_recall", grid_obj.best_score_)

    y_prob_train_xgb = best_model_xgb.predict_proba(Xtrain)[:, 1]
    y_prob_test_xgb = best_model_xgb.predict_proba(Xtest)[:, 1]

    # 4. Generate Predictions & Log Final Performance Reasoning
    operational_threshold = 0.55
    y_pred_train_xgb = (y_prob_train_xgb >= operational_threshold).astype(int)
    y_pred_test_xgb = (y_prob_test_xgb >= operational_threshold).astype(int)

    # Logging  the threshold itself 
    mlflow.log_param("decision_threshold", operational_threshold)
    # Logging Train vs Test to detect any remaining overfitting
    mlflow.log_metric("final_train_recall", recall_score(ytrain, y_pred_train_xgb, pos_label=1))
    mlflow.log_metric("final_test_recall", recall_score(ytest, y_pred_test_xgb, pos_label=1))
    mlflow.log_metric("final_test_f1", f1_score(ytest, y_pred_test_xgb, pos_label=1))

    # 5. Log the Model Artifact
    mlflow.sklearn.log_model(best_model_xgb, "optimized_engine_model")

    print(f"Optimization complete. Best Params: {grid_obj.best_params_}")

    # Classification metrics
train_accuracy_xgb = accuracy_score(ytrain, y_pred_train_xgb)
test_accuracy_xgb = accuracy_score(ytest, y_pred_test_xgb)

train_precision_xgb = precision_score(ytrain, y_pred_train_xgb, pos_label=1)
test_precision_xgb = precision_score(ytest, y_pred_test_xgb, pos_label=1)

train_recall_xgb = recall_score(ytrain, y_pred_train_xgb, pos_label=1)
test_recall_xgb = recall_score(ytest, y_pred_test_xgb, pos_label=1)

train_f1_xgb = f1_score(ytrain, y_pred_train_xgb, pos_label=1)
test_f1_xgb = f1_score(ytest, y_pred_test_xgb, pos_label=1)

# Log metrics
mlflow.log_metrics({
    "train_accuracy": train_accuracy_xgb,
    "test_accuracy": test_accuracy_xgb,
    "train_precision": train_precision_xgb,
    "test_precision": test_precision_xgb,
    "train_recall": train_recall_xgb,
    "test_recall": test_recall_xgb,
    "train_f1": train_f1_xgb,
    "test_f1": test_f1_xgb
})
