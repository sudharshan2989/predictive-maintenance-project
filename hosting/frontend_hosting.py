
from huggingface_hub import HfApi
import os

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="predictive-vehicle-maintenance/deployment/frontend_files",     # the local folder containing your files
    repo_id="sudharshanc/predictive-vehicle-maintenance-frontend",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
