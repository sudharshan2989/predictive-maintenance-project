import pandas as pd
import os
from huggingface_hub import login
from huggingface_hub import hf_hub_download


# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score


# for model serialization
import joblib

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow


# Initialize API client
api = HfApi(token=HF_TOKEN)


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
