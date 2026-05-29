"""
Push the fine-tuned model to the Hugging Face Hub as a model repo, with the model card.

Reads the cached HF token (from huggingface_hub.login()), so it never receives the token.
Uploads everything in model/ plus MODEL_CARD.md as the repo README.md.

Run only AFTER training (model/ exists) and after MODEL_CARD.md metrics are filled in:
    python3 _push_model.py
"""
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "LaelaZ/distilbert-emotion"
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"

api = HfApi()
print("Authenticated as:", api.whoami()["name"])

create_repo(REPO_ID, repo_type="model", exist_ok=True)
print("model repo ready")

api.upload_folder(
    repo_id=REPO_ID, repo_type="model", folder_path=str(MODEL_DIR),
    commit_message="Add fine-tuned DistilBERT emotion classifier",
)
print("uploaded model files")

api.upload_file(
    path_or_fileobj=str(HERE / "MODEL_CARD.md"),
    path_in_repo="README.md", repo_id=REPO_ID, repo_type="model",
    commit_message="Add model card",
)
print(f"model card set -> https://huggingface.co/{REPO_ID}")
