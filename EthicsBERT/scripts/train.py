import argparse
import json
import os

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def tokenize_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def main():
    parser = argparse.ArgumentParser(description="Train EthicsBERT with DistilBERT")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="EthicsBERT/data/sample_ethics_dataset.csv",
        help="Path to CSV file with text,label columns",
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
        help="Base DistilBERT checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.dataset_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    label_encoder = LabelEncoder()
    df["labels"] = label_encoder.fit_transform(df["label"])

    train_df, eval_df = train_test_split(
        df[["text", "labels"]], test_size=args.test_size, random_state=args.seed, shuffle=True
    )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    train_dataset = train_dataset.map(
        tokenize_batch, fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length}, batched=True
    )
    eval_dataset = eval_dataset.map(
        tokenize_batch, fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length}, batched=True
    )

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in id2label.items()}

    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    mapping_path = os.path.join(args.output_dir, "label_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    print(f"Training completed. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
