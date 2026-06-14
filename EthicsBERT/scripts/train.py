"""
EthicsBERT – Training Script
Fine-tunes DistilBERT for 9-class AI-ethics topic classification.

Usage:
    python EthicsBERT/scripts/train.py \
        --dataset_path EthicsBERT/data/sample_ethics_dataset.csv \
        --output_dir EthicsBERT/model
"""

import argparse
import json
import os

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Canonical label ordering keeps id2label deterministic across runs.
LABELS = [
    "Agency",
    "AI Governance",
    "Bias",
    "Consciousness",
    "Ethical Reasoning",
    "Explainability",
    "Fairness",
    "Intelligence",
    "Privacy",
]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def tokenize_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def build_label_maps(labels):
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    return label2id, id2label


def main():
    parser = argparse.ArgumentParser(description="Train EthicsBERT with DistilBERT")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="EthicsBERT/data/sample_ethics_dataset.csv",
        help="Path to CSV with 'text' and 'label' columns",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="EthicsBERT/model",
        help="Directory for trained model artifacts",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Base DistilBERT checkpoint on Hugging Face Hub",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
        help="Stop training if metric does not improve for N evaluations",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load & validate data
    # ------------------------------------------------------------------
    df = pd.read_csv(args.dataset_path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required}")

    unknown = sorted(set(df["label"].unique()) - set(LABELS))
    if unknown:
        raise ValueError(
            f"Dataset contains labels not in the canonical label list: {unknown}\n"
            f"Expected one of: {LABELS}"
        )

    label2id, id2label = build_label_maps(LABELS)
    df["labels"] = df["label"].map(label2id)

    # ------------------------------------------------------------------
    # Train / eval split
    # ------------------------------------------------------------------
    train_df, eval_df = train_test_split(
        df[["text", "labels"]],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["labels"],
        shuffle=True,
    )
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))

    print(f"Train examples : {len(train_dataset)}")
    print(f"Eval  examples : {len(eval_dataset)}")

    # ------------------------------------------------------------------
    # Tokenise
    # ------------------------------------------------------------------
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    fn_kwargs = {"tokenizer": tokenizer, "max_length": args.max_length}
    train_dataset = train_dataset.map(tokenize_batch, fn_kwargs=fn_kwargs, batched=True)
    eval_dataset = eval_dataset.map(tokenize_batch, fn_kwargs=fn_kwargs, batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=cols)
    eval_dataset.set_format(type="torch", columns=cols)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,          # keep only the single best checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        report_to="none",
        seed=args.seed,
        fp16=False,  # set True on CUDA GPUs for speed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    # ------------------------------------------------------------------
    # Train & save
    # ------------------------------------------------------------------
    print("Starting training …")
    trainer.train()

    final_metrics = trainer.evaluate()
    print("\nFinal evaluation metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Remove any leftover checkpoint subdirectories — only the final model files are needed
    import shutil
    for entry in os.listdir(args.output_dir):
        if entry.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(args.output_dir, entry))

    mapping_path = os.path.join(args.output_dir, "label_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    print(f"\nModel, tokenizer, and label mapping saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
