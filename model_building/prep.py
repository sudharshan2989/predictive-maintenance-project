from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
from huggingface_hub import hf_hub_download
import pandas as pd
import os


# Define repo and filenames
repo_id = "sudharshanc/predictive-vehicle-maintenance"
csv_path = hf_hub_download(repo_id=repo_id,repo_type="dataset", filename="engine_data.csv")
# Read the CSV into a DataFrame
df_hf_main = pd.read_csv(csv_path)


#Converting the temperature columns from Celsius to Kelvin
df_hf_main['lub oil temp K'] = df_hf_main['lub oil temp'] + 273.15
df_hf_main['Coolant temp K'] = df_hf_main['Coolant temp'] + 273.15

## Lets add pressure to temperature ratio features for coolant and lub oil
df_hf_main['Coolant_p_t_ratio'] = df_hf_main['Coolant pressure'] / df_hf_main['Coolant temp K'] #helps to keep a tab on coolant system health
df_hf_main['Lub_oil_p_t_ratio'] = df_hf_main['Lub oil pressure'] / df_hf_main['lub oil temp K'] #helps to keep a tab on lub oil system health


# 1. Define Features and Target
# Drop the target class AND the redundant Celsius columns to isolate Kelvin
X = df_hf_main.drop(['Engine Condition', 'lub oil temp', 'Coolant temp'  ], axis=1)
y = df_hf_main['Engine Condition']

# 2. Split with Stratification
# 'stratify=y' ensures the 0/1 ratio is preserved in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Class distribution in train:", y_train.value_counts())
print("Class distribution in test:", y_test.value_counts())


## Uploading the train and test files to Hugging Face Hub
X_train.to_csv("Xtrain.csv",index=False)
X_test.to_csv("Xtest.csv",index=False)
y_train.to_csv("ytrain.csv",index=False)
y_test.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

HF_TOKEN = os.getenv("HF_TOKEN")
# Initialize API client
api = HfApi(token=HF_TOKEN)

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=repo_id,
        repo_type="dataset",
    )
