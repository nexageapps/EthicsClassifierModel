# EthicsBERT

Hugging Face text classification project for AI ethics topic detection.

## Project Structure

```text
EthicsBERT/
├── data/
│   └── sample_ethics_dataset.csv
└── scripts/
    ├── evaluate.py
    ├── infer.py
    └── train.py
```

## Labels

- Fairness
- Bias
- Privacy
- Transparency
- Accountability
- Agency
- Consciousness

## Requirements

- Python
- Hugging Face Transformers
- DistilBERT
- Pandas
- Datasets
- Scikit-learn

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install transformers datasets pandas scikit-learn torch
```

## Train

```bash
python EthicsBERT/scripts/train.py \
  --dataset_path EthicsBERT/data/sample_ethics_dataset.csv \
  --output_dir EthicsBERT/model
```

## Evaluate

```bash
python EthicsBERT/scripts/evaluate.py \
  --dataset_path EthicsBERT/data/sample_ethics_dataset.csv \
  --model_dir EthicsBERT/model
```

## Inference

```bash
python EthicsBERT/scripts/infer.py \
  --model_dir EthicsBERT/model \
  --text "The board requested clear model documentation and audit trails."
```
