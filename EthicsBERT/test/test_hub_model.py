from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="nexageapps/EthicsBERT",
    top_k=3,
)

result = classifier("The hiring algorithm must produce equal outcomes across demographic groups.")
# [{'label': 'Fairness', 'score': 0.92}, ...]
print(result)
