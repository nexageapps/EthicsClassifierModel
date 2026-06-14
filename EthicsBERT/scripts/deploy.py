"""
EthicsBERT – Hugging Face Hub Deployment Script
Pushes the trained model, tokenizer, and model card to the HF Hub.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login          # authenticate once

Usage:
    python EthicsBERT/scripts/deploy.py \
        --model_dir EthicsBERT/model \
        --repo_id nexageapps/EthicsBERT \
        --model_card_path EthicsBERT/MODEL_CARD.md

Optional flags:
    --private          Create a private repository (default: public)
    --commit_message   Custom commit message for the Hub push
"""

import argparse
import os
import shutil

from huggingface_hub import HfApi, create_repo


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
        required=True,
        help="Hub repo id in the form <username>/<repo-name>",
    )
    parser.add_argument(
        "--model_card_path",
        type=str,
        default="EthicsBERT/MODEL_CARD.md",
        help="Path to the model card Markdown file",
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

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {args.model_dir}. "
            "Run train.py first to generate model artifacts."
        )

    api = HfApi()

    # ------------------------------------------------------------------
    # 1. Create repository (idempotent – safe to call if it exists)
    # ------------------------------------------------------------------
    print(f"Creating / confirming repository: {args.repo_id}")
    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # 2. Copy model card to model dir so it is uploaded as README.md
    # ------------------------------------------------------------------
    readme_dest = os.path.join(args.model_dir, "README.md")
    if os.path.exists(args.model_card_path):
        shutil.copy(args.model_card_path, readme_dest)
        print(f"Copied model card to {readme_dest}")
    else:
        print(
            f"Warning: model card not found at {args.model_card_path}. "
            "Uploading without README."
        )

    # ------------------------------------------------------------------
    # 3. Upload the entire model directory
    # ------------------------------------------------------------------
    print(f"Uploading model from {args.model_dir} to {args.repo_id} …")
    api.upload_folder(
        folder_path=args.model_dir,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
        ignore_patterns=["eval_tmp/**", "logs/**", "*.ckpt", "checkpoint-*/**"],
    )

    print(f"\nDeployment complete.")
    print(f"View your model at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
