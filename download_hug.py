from huggingface_hub import snapshot_download
import os

# Define the repository ID on Hugging Face
repo_id = "mm-graph-org/mm-graph"

# Define the *base* local directory where you want to save all datasets
# The content of the 'mm-graph' repo will be downloaded directly here.
base_local_dir = "./Multimodal-Graph-Completed-Graph"

# Ensure the base directory exists
os.makedirs(base_local_dir, exist_ok=True)

# List of specific dataset folders you want to download
# These correspond to the top-level folders within the mm-graph repo.
dataset_folders = [
    "books-lp",
    "books-nc",
    "cloth-copurchase",
    "ele-fashion",
    "mm-codex-m",
    "mm-codex-s",
    "raw-data",
    "sports-copurchase",
]

print(f"Downloading datasets from {repo_id} to {base_local_dir}...")

for folder in dataset_folders:
    print(f"Downloading {folder}...")
    try:
        # snapshot_download will create the 'folder' directly inside base_local_dir
        # because the 'allow_patterns' will restrict it to that specific top-level folder.
        snapshot_download(
            repo_id=repo_id,
            local_dir=base_local_dir, # <--- Changed: now using the base directory
            allow_patterns=[f"{folder}/**"], # <--- This ensures only 'folder' content is downloaded
            repo_type="dataset",
            # revision="main" # Optional: specify a particular branch or commit
        )
        print(f"Finished downloading {folder}.")
    except Exception as e:
        print(f"Error downloading {folder}: {e}")
        print("Please double check the folder name and ensure it exists in the Hugging Face repository.")

print("All specified datasets download attempt completed.")