import pandas as pd
from sklearn.metrics import f1_score

# Load predictions
df = pd.read_json("outputs/llama_chatdoctor_medmcqa_50.jsonl", lines=True)

# Normalize to uppercase for fair comparison
gold = df.gold.str.upper()
pred = df.pred.str.upper()

# Accuracy
accuracy = (gold == pred).mean()

# F1 Score (macro, across classes A/B/C/D)
f1 = f1_score(gold, pred, average="macro", labels=["A", "B", "C", "D"])

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {f1:.4f}")
