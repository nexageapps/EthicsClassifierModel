import argparse
import json
import os

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


def main():
    parser = argparse.ArgumentParser(description="Run EthicsBERT inference")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="EthicsBERT/model",
        help="Directory containing trained model files",
    )
    parser.add_argument("--text", type=str, required=True, help="Input text for classification")
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    mapping_path = os.path.join(args.model_dir, "label_mapping.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Missing label mapping file: {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    id2label = {int(k): v for k, v in mapping["id2label"].items()}

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    encoded = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=args.max_length)
    with torch.no_grad():
        logits = model(**encoded).logits
        predicted_id = int(torch.argmax(logits, dim=-1).item())

    print(f"Predicted label: {id2label[predicted_id]}")


if __name__ == "__main__":
    main()
