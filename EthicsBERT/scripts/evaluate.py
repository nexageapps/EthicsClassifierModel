"""
EthicsBERT – Evaluation Script
Loads a trained model and produces a full classification report.

Usage:
    python EthicsBERT/scripts/evaluate.py \
        --dataset_path EthicsBERT/data/sample_ethics_dataset.csv \
        --model_dir EthicsBERT/model
"""

import argparse
import json
import os

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)


def tokenize_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained EthicsBERT model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="EthicsBERT/data/sample_ethics_dataset.csv",
        help="Path to CSV with 'text' and 'label' columns",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="EthicsBERT/model",
        help="Directory containing trained model files",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--output_report",
        type=str,
        default=None,
        help="Optional path to save the classification report as a text file",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load label mapping
    # ------------------------------------------------------------------
    mapping_path = os.path.join(args.model_dir, "label_mapping.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"Label mapping not found at {mapping_path}. "
            "Run train.py first or ensure --model_dir is correct."
        )
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    label2id = mapping["label2id"]
    id2label = {int(k): v for k, v in mapping["id2label"].items()}

    # ------------------------------------------------------------------
    # Load & map dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(args.dataset_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    df["labels"] = df["label"].map(label2id)
    unmapped = df[df["labels"].isnull()]["label"].unique().tolist()
    if unmapped:
        raise ValueError(f"Labels not in model mapping: {sorted(unmapped)}")
    df["labels"] = df["labels"].astype(int)

    # ------------------------------------------------------------------
    # Tokenise
    # ------------------------------------------------------------------
    dataset = Dataset.from_pandas(df[["text", "labels"]].reset_index(drop=True))
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    dataset = dataset.map(
        tokenize_batch,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)
    eval_args = TrainingArguments(
        output_dir=os.path.join(args.model_dir, "eval_tmp"),
        report_to="none",
        per_device_eval_batch_size=16,
        do_train=False,
    )
    trainer = Trainer(model=model, args=eval_args)

    preds = trainer.predict(dataset)
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(axis=-1)

    target_names = [id2label[i] for i in sorted(id2label.keys())]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0, digits=4
    )
    cm = confusion_matrix(y_true, y_pred)

    output_lines = [
        "=" * 60,
        "EthicsBERT – Evaluation Report",
        "=" * 60,
        f"Dataset : {args.dataset_path}",
        f"Model   : {args.model_dir}",
        f"Samples : {len(df)}",
        "",
        f"Overall Accuracy : {accuracy:.4f}",
        "",
        "Classification Report:",
        report,
        "Confusion Matrix (rows=true, cols=pred):",
        f"Labels  : {target_names}",
        str(cm),
        "=" * 60,
    ]

    full_report = "\n".join(output_lines)
    print(full_report)

    if args.output_report:
        os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\nReport saved to: {args.output_report}")


if __name__ == "__main__":
    main()
