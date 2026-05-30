from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os



# 1. Define Features and Target
# Drop the target class AND the redundant Celsius columns to isolate Kelvin
X = df_hf_copy.drop(['Engine Condition', 'lub oil temp', 'Coolant temp'  ], axis=1)
y = df_hf_copy['Engine Condition']

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
