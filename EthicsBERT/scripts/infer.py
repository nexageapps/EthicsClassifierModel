"""
EthicsBERT – Inference Script
Classifies one or more text inputs using a trained EthicsBERT model.

Usage (single text):
    python EthicsBERT/scripts/infer.py \
        --model_dir EthicsBERT/model \
        --text "The board requested clear model documentation and audit trails."

Usage (batch from file):
    python EthicsBERT/scripts/infer.py \
        --model_dir EthicsBERT/model \
        --input_file my_texts.txt        # one sentence per line

Usage (Hugging Face Hub model):
    python EthicsBERT/scripts/infer.py \
        --model_dir your-hf-username/EthicsBERT \
        --text "Privacy-preserving federated learning avoids data centralisation."
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


def load_model(model_dir: str):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Try label mapping from a local file first; fall back to model config.
    mapping_path = os.path.join(model_dir, "label_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        id2label = {int(k): v for k, v in mapping["id2label"].items()}
    elif hasattr(model.config, "id2label"):
        id2label = {int(k): v for k, v in model.config.id2label.items()}
    else:
        raise RuntimeError(
            "Cannot find label mapping. Provide label_mapping.json or ensure "
            "the model config contains id2label."
        )
    return tokenizer, model, id2label


def classify(
    texts: list[str],
    tokenizer,
    model,
    id2label: dict,
    max_length: int = 128,
    top_k: int = 3,
) -> list[dict]:
    """Return predictions with confidence scores for a list of texts."""
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = F.softmax(logits, dim=-1)

    results = []
    for i, prob_row in enumerate(probs):
        top_indices = prob_row.argsort(descending=True)[:top_k].tolist()
        top_preds = [
            {"label": id2label[idx], "score": round(prob_row[idx].item(), 4)}
            for idx in top_indices
        ]
        results.append(
            {
                "text": texts[i],
                "predicted_label": id2label[int(prob_row.argmax())],
                "confidence": round(prob_row.max().item(), 4),
                "top_k": top_preds,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Run EthicsBERT inference")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="EthicsBERT/model",
        help="Local directory or Hugging Face Hub model ID",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text string to classify",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to a plain-text file with one sentence per line",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top label predictions to display per input",
    )
    args = parser.parse_args()

    if not args.text and not args.input_file:
        parser.error("Provide either --text or --input_file.")

    tokenizer, model, id2label = load_model(args.model_dir)

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = [args.text]

    results = classify(texts, tokenizer, model, id2label, args.max_length, args.top_k)

    for res in results:
        print("\n" + "─" * 60)
        print(f"Text       : {res['text']}")
        print(f"Prediction : {res['predicted_label']}  (confidence: {res['confidence']:.2%})")
        print(f"Top-{args.top_k} :")
        for rank, pred in enumerate(res["top_k"], 1):
            print(f"  {rank}. {pred['label']:<22} {pred['score']:.2%}")
    print("─" * 60)


if __name__ == "__main__":
    main()
