---
language: en
license: apache-2.0
library_name: transformers
tags:
  - text-classification
  - ethics
  - ai-ethics
  - distilbert
  - education
datasets:
  - custom
metrics:
  - accuracy
  - f1
pipeline_tag: text-classification
model-index:
  - name: EthicsBERT
    results:
      - task:
          type: text-classification
          name: Text Classification
        metrics:
          - type: accuracy
            name: Accuracy
          - type: f1
            name: F1 (weighted)
---

# EthicsBERT

**EthicsBERT** is a fine-tuned [DistilBERT](https://huggingface.co/distilbert-base-uncased)
model for classifying AI ethics course content into nine topic areas. It is designed to
support educators, researchers, and curriculum designers working in AI ethics.

## Labels

| ID | Label | Description |
|----|-------|-------------|
| 0 | **Agency** | Human control, override rights, autonomy-preserving design, and delegated authority. |
| 1 | **AI Governance** | Regulation, accountability, audits, policy frameworks, and existential risk oversight. |
| 2 | **Bias** | Systematic errors, skewed training data, unfair representations, and proxy discrimination. |
| 3 | **Consciousness** | Machine sentience, subjective experience, philosophical debates, and moral patienthood. |
| 4 | **Ethical Reasoning** | Moral frameworks (utilitarian, deontological, virtue), dilemmas, and applied ethics. |
| 5 | **Explainability** | Interpretability, SHAP/LIME, attention visualization, and model transparency. |
| 6 | **Fairness** | Equitable outcomes, anti-discrimination, group/individual fairness metrics. |
| 7 | **Intelligence** | Cognitive capabilities, reasoning, transfer learning, AGI, and benchmarks. |
| 8 | **Privacy** | Data protection, consent, PII handling, differential privacy, and encryption. |

## Model Details

| Property | Value |
|----------|-------|
| Base model | `distilbert-base-uncased` |
| Architecture | DistilBERT + classification head |
| Task | Multi-class text classification (9 classes) |
| Max sequence length | 128 tokens |
| Training epochs | 5 (with early stopping) |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |

## Usage

### With the `pipeline` API (simplest)

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="nexageapps/EthicsBERT",
    top_k=3,
)

result = classifier("The hiring algorithm must produce equal outcomes across demographic groups.")
# [{'label': 'Fairness', 'score': 0.92}, ...]
print(result)
```

### Direct usage

```python
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

model_id = "nexageapps/EthicsBERT"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_id)
model = DistilBertForSequenceClassification.from_pretrained(model_id)
model.eval()

text = "SHAP values quantify each feature's contribution to a specific model prediction."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)

id2label = model.config.id2label
predicted_label = id2label[int(probs.argmax())]
print(f"Predicted: {predicted_label} ({probs.max().item():.2%})")
```

## Training Data

The model was fine-tuned on a curated dataset of approximately 200 sentences covering
all nine AI ethics topic areas. Sentences were authored to reflect the vocabulary,
framing, and conceptual depth typical of AI ethics course materials.

**Label distribution:** roughly balanced, ~20–25 examples per class.

## Limitations

- The training dataset is small (~200 examples). Performance may degrade on highly
  technical or domain-specific text not represented in training.
- The model was trained on English only.
- Boundary cases between semantically similar labels (e.g., *Fairness* vs *Bias*,
  or *AI Governance* vs *Ethical Reasoning*) may be misclassified.
- The model should not be used as the sole arbiter in automated grading or gatekeeping systems.

## Ethical Considerations

This model is intended for research and educational purposes. Automated topic
classification of ethics content should always be reviewed by a human expert before
consequential use.

## Citation

If you use EthicsBERT in your research or course materials, please cite:

```bibtex
@misc{ethicsbert2024,
  title   = {EthicsBERT: A DistilBERT Model for AI Ethics Topic Classification},
  author  = {nexageapps},
  year    = {2024},
  url     = {https://huggingface.co/nexageapps/EthicsBERT}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
