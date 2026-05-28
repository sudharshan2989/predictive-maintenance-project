
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "sudharshanc/predictive-vehicle-maintenance"
repo_type = "dataset"


HF_TOKEN = os.getenv("HF_TOKEN")
# Initialize API client
api = HfApi(token=HF_TOKEN)

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")


api.upload_folder(
    folder_path="data",
    repo_id=repo_id,
    repo_type=repo_type,
)
