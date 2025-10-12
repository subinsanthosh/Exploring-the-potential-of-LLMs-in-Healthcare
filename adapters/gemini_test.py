import pandas as pd
from sklearn.metrics import f1_score

# Load predictions
df = pd.read_json("outputs/gemini_medmcqa.jsonl", lines=True)

# Accuracy
acc = (df.gold.str.upper() == df.pred.str.upper()).mean()
print("Accuracy:", acc)

# Macro F1 (averages across A/B/C/D)
f1 = f1_score(df.gold.str.upper(), df.pred.str.upper(), average="macro")
print("Macro F1:", f1)

# If you want per-class F1 (A, B, C, D)
# f1_per_class = f1_score(df.gold.str.upper(), df.pred.str.upper(), average=None, labels=["A","B","C","D"])
# print("F1 per class:", dict(zip(["A","B","C","D"], f1_per_class)))
