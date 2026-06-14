# EthicsBERT

A fine-tuned **DistilBERT** model for classifying AI ethics course content into nine
topic areas. Built to support educators, researchers, and curriculum designers.

## Labels

| Label | Description |
|-------|-------------|
| **Agency** | Human control, override rights, and autonomy-preserving AI design |
| **AI Governance** | Regulation, audits, policy, accountability, and existential risk |
| **Bias** | Systematic errors, skewed data, proxy discrimination |
| **Consciousness** | Machine sentience, subjective experience, philosophical debates |
| **Ethical Reasoning** | Moral frameworks, dilemmas, applied ethics principles |
| **Explainability** | SHAP/LIME, interpretability, attention, model transparency |
| **Fairness** | Equitable outcomes, anti-discrimination, parity metrics |
| **Intelligence** | Reasoning, transfer learning, AGI, cognitive benchmarks |
| **Privacy** | Data protection, consent, PII, differential privacy |

## Project Structure

```
EthicsClassifierModel/
├── EthicsBERT/
│   ├── data/
│   │   └── sample_ethics_dataset.csv   # ~200 labelled training examples
│   ├── scripts/
│   │   ├── train.py                    # Fine-tune DistilBERT
│   │   ├── evaluate.py                 # Full classification report
│   │   ├── infer.py                    # Single / batch inference
│   │   └── deploy.py                   # Push to Hugging Face Hub
│   ├── app.py                          # Gradio demo (HF Spaces ready)
│   └── MODEL_CARD.md                   # Hugging Face model card
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Core ML deps (train, evaluate, infer, deploy)
pip install -r requirements.txt

# Gradio demo only (separate due to huggingface_hub version split)
pip install -r requirements-app.txt
```

## Train

```bash
python3 EthicsBERT/scripts/train.py \
  --dataset_path EthicsBERT/data/sample_ethics_dataset.csv \
  --output_dir   EthicsBERT/model \
  --epochs       5 \
  --batch_size   16 \
  --learning_rate 2e-5
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_path` | `EthicsBERT/data/sample_ethics_dataset.csv` | Training CSV |
| `--output_dir` | `EthicsBERT/model` | Where to save the model |
| `--model_name` | `distilbert-base-uncased` | Base HF checkpoint |
| `--epochs` | `5` | Number of training epochs |
| `--batch_size` | `16` | Per-device batch size |
| `--learning_rate` | `2e-5` | AdamW learning rate |
| `--early_stopping_patience` | `2` | Stop if no F1 improvement |

## Evaluate

```bash
python3 EthicsBERT/scripts/evaluate.py \
  --dataset_path EthicsBERT/data/sample_ethics_dataset.csv \
  --model_dir    EthicsBERT/model \
  --output_report EthicsBERT/eval_report.txt
```

## Inference

Single text:

```bash
python3 EthicsBERT/scripts/infer.py \
  --model_dir EthicsBERT/model \
  --text "The board requested clear model documentation and audit trails."
```

Batch from file (one sentence per line):

```bash
python3 EthicsBERT/scripts/infer.py \
  --model_dir   EthicsBERT/model \
  --input_file  my_texts.txt \
  --top_k       3
```

Using a Hub-hosted model directly:

```bash
python3 EthicsBERT/scripts/infer.py \
  --model_dir nexageapps/EthicsBERT \
  --text "Differential privacy protects individuals in aggregate queries."
```

## Deploy to Hugging Face Hub

```bash
# Authenticate once
huggingface-cli login

# Push model + tokenizer + model card
python3 EthicsBERT/scripts/deploy.py \
  --model_dir       EthicsBERT/model \
  --repo_id         nexageapps/EthicsBERT \
  --model_card_path EthicsBERT/MODEL_CARD.md
```

## Gradio Demo (Local or HF Spaces)

```bash
pip install gradio
python3 EthicsBERT/app.py
```

To deploy to Hugging Face Spaces:
1. Create a new Space (Gradio SDK).
2. Push `EthicsBERT/app.py` and `requirements.txt` to the Space repo.
3. Set `HF_MODEL_ID=nexageapps/EthicsBERT` in the Space secrets.

## Quick API Usage (after Hub deployment)

```python
from transformers import pipeline

clf = pipeline("text-classification", model="nexageapps/EthicsBERT", top_k=3)
print(clf("SHAP values explain each feature's contribution to the prediction."))
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Transformers ≥ 4.41
- See `requirements.txt` for full pinned dependency list
