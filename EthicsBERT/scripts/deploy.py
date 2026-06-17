"""
EthicsBERT – Hugging Face Hub Deployment Script
Pushes the trained model, tokenizer, and model card to the HF Hub.

Usage:
    python3 EthicsBERT/scripts/deploy.py \
        --model_dir       EthicsBERT/model \
        --repo_id         nexageapps/EthicsBERT \
        --model_card_path EthicsBERT/MODEL_CARD.md \
        --token           hf_...yourtoken...

Or set the token via environment variable (safer — avoids it appearing in shell history):
    export HF_TOKEN=hf_...yourtoken...
    python3 EthicsBERT/scripts/deploy.py \
        --model_dir       EthicsBERT/model \
        --repo_id         nexageapps/EthicsBERT \
        --model_card_path EthicsBERT/MODEL_CARD.md

Optional flags:
    --private          Create a private repository (default: public)
    --commit_message   Custom commit message
"""

import argparse
import os
import shutil

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Push EthicsBERT to Hugging Face Hub")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="EthicsBERT/model",
        help="Local directory containing trained model artifacts",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="nexageapps/EthicsBERT",
        help="Hub repo id in the form <username>/<repo-name>",
    )
    parser.add_argument(
        "--model_card_path",
        type=str,
        default="EthicsBERT/MODEL_CARD.md",
        help="Path to the model card Markdown file",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face write token. Falls back to HF_TOKEN env var if not set.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload EthicsBERT fine-tuned DistilBERT model",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve token — arg > env var
    # ------------------------------------------------------------------
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "No Hugging Face token found.\n"
            "Pass it with --token hf_... or set the HF_TOKEN environment variable:\n"
            "  export HF_TOKEN=hf_..."
        )

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {args.model_dir}\n"
            "Run train.py first to generate model artifacts."
        )

    # Pass token directly to HfApi — no login() call needed
    api = HfApi(token=token)

    # ------------------------------------------------------------------
    # 1. Create repository (safe to call if it already exists)
    # ------------------------------------------------------------------
    print(f"Creating / confirming repository: {args.repo_id}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # 2. Copy model card into model dir as README.md
    # ------------------------------------------------------------------
    readme_dest = os.path.join(args.model_dir, "README.md")
    if os.path.exists(args.model_card_path):
        shutil.copy(args.model_card_path, readme_dest)
        print(f"Model card copied to {readme_dest}")
    else:
        print(f"Warning: model card not found at {args.model_card_path} — uploading without README.")

    # ------------------------------------------------------------------
    # 3. Upload the model directory
    # ------------------------------------------------------------------
    print(f"\nUploading {args.model_dir}  →  {args.repo_id} …")
    api.upload_folder(
        folder_path=args.model_dir,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
        ignore_patterns=["eval_tmp/**", "logs/**", "*.ckpt", "checkpoint-*/**"],
    )

    print(f"\nDeployment complete!")
    print(f"View your model at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
