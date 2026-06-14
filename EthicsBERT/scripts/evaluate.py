import argparse
import json
import os

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments


def tokenize_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained EthicsBERT model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="EthicsBERT/data/sample_ethics_dataset.csv",
        help="Path to CSV file with text,label columns",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="EthicsBERT/model",
        help="Directory containing trained model files",
    )
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    mapping_path = os.path.join(args.model_dir, "label_mapping.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Missing label mapping file: {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    label2id = mapping["label2id"]
    df["labels"] = df["label"].map(label2id)
    if df["labels"].isnull().any():
        unknown_labels = sorted(df[df["labels"].isnull()]["label"].unique())
        raise ValueError(f"Found labels not present in model mapping: {unknown_labels}")
    df["labels"] = df["labels"].astype(int)

    dataset = Dataset.from_pandas(df[["text", "labels"]].reset_index(drop=True))
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    dataset = dataset.map(
        tokenize_batch, fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length}, batched=True
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)
    eval_args = TrainingArguments(output_dir=os.path.join(args.model_dir, "eval_tmp"), report_to="none")
    trainer = Trainer(model=model, args=eval_args, tokenizer=tokenizer)

    preds = trainer.predict(dataset)
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(axis=-1)

    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    target_names = [id2label[idx] for idx in sorted(id2label.keys())]

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))


if __name__ == "__main__":
    main()
