"""
EthicsBERT – Gradio Demo App
Runs a web UI for the EthicsBERT classifier.

Deploy to Hugging Face Spaces:
    1. Create a new Space (Gradio SDK) at https://huggingface.co/spaces
    2. Push this file and requirements.txt to the Space repository.
    3. Set HF_MODEL_ID to your model repo id in the Space settings (optional).

Local run:
    pip install gradio
    python EthicsBERT/app.py
"""

import os

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Model can be overridden via an environment variable for Spaces deployment.
MODEL_ID = os.getenv("HF_MODEL_ID", "EthicsBERT/model")
MAX_LENGTH = 128

LABEL_DESCRIPTIONS = {
    "Agency": "Human control, autonomy, override rights, and delegated authority.",
    "AI Governance": "Regulation, accountability, audits, oversight, and policy frameworks.",
    "Bias": "Systematic errors, unfair representations, and skewed training data.",
    "Consciousness": "Sentience, subjective experience, and machine awareness debates.",
    "Ethical Reasoning": "Moral frameworks, dilemmas, principles, and applied ethics.",
    "Explainability": "Interpretability, transparency, SHAP/LIME, and model explanations.",
    "Fairness": "Equitable outcomes, anti-discrimination, and statistical parity.",
    "Intelligence": "Cognitive capabilities, reasoning, transfer learning, and AGI.",
    "Privacy": "Data protection, consent, PII handling, and encryption.",
}

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
print(f"Loading model from: {MODEL_ID}")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_ID)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------
def predict(text: str) -> dict:
    """Return a {label: probability} dict for Gradio's Label component."""
    if not text or not text.strip():
        return {}

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = F.softmax(logits, dim=-1)[0]

    scores = {id2label[i]: round(probs[i].item(), 4) for i in range(len(id2label))}
    return scores


def predict_with_description(text: str):
    """Wrapper that also returns the description of the top label."""
    if not text or not text.strip():
        return {}, "Please enter some text."

    scores = predict(text)
    top_label = max(scores, key=scores.get)
    description = LABEL_DESCRIPTIONS.get(top_label, "")
    detail = f"**{top_label}** — {description}"
    return scores, detail


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------
examples = [
    "The hiring algorithm must produce equal pass rates across all demographic groups.",
    "SHAP values quantify each feature's contribution to the model's output.",
    "Users must be able to override the AI's recommendation at any point.",
    "Some researchers argue that sufficiently complex systems may develop sentience.",
    "The EU AI Act imposes conformity assessments on high-risk AI systems.",
    "Differential privacy protects individuals in aggregate statistical queries.",
    "Recidivism tools assign higher risk scores due to biased historical policing data.",
    "Utilitarian ethics maximises aggregate welfare, sometimes at the cost of individual rights.",
    "Chain-of-thought prompting enables large models to perform multi-step reasoning.",
]

with gr.Blocks(title="EthicsBERT – AI Ethics Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # EthicsBERT
        ### AI Ethics Topic Classifier
        Fine-tuned **DistilBERT** that categorises text about AI ethics into one of
        9 topic areas: *Agency, AI Governance, Bias, Consciousness, Ethical Reasoning,
        Explainability, Fairness, Intelligence,* and *Privacy*.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Enter an AI ethics related sentence or paragraph…",
                lines=4,
            )
            classify_btn = gr.Button("Classify", variant="primary")
            gr.Examples(examples=examples, inputs=input_text, label="Try an example")

        with gr.Column(scale=2):
            label_output = gr.Label(num_top_classes=9, label="Topic Probabilities")
            top_label_md = gr.Markdown(label="Top Prediction")

    classify_btn.click(
        fn=predict_with_description,
        inputs=input_text,
        outputs=[label_output, top_label_md],
    )
    input_text.submit(
        fn=predict_with_description,
        inputs=input_text,
        outputs=[label_output, top_label_md],
    )

    gr.Markdown(
        """
        ---
        Model: [EthicsBERT](https://huggingface.co/nexageapps/EthicsBERT) |
        Base: `distilbert-base-uncased` |
        Labels: Agency · AI Governance · Bias · Consciousness ·
        Ethical Reasoning · Explainability · Fairness · Intelligence · Privacy
        """
    )

if __name__ == "__main__":
    demo.launch()
