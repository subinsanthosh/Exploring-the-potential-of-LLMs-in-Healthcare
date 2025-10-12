# eval_viz.py
import json, os, math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ---------------------------
# 1) Point these to YOUR files
# ---------------------------
# Replace the paths below with your actual output files for each model
FILES = {
    "ChatDoctor (Llama)": {
        "medmcqa": "outputs/llama_chatdoctor_medmcqa_10000.jsonl",
        "pubmedqa": "outputs/llama_chatdoctor_pubmedqa_10000.jsonl",
    },
    "Gemini 2.5 Flash Lite": {
        "medmcqa": "outputs/gemini_flashlite_medmcqa_10000.jsonl",
        "pubmedqa": "outputs/gemini_flashlite_pubmedqa_10000.jsonl",
    },
    "Grok 3": {
        "medmcqa": "outputs/grok3_medmcqa_10000.jsonl",
        "pubmedqa": "outputs/grok3_pubmedqa_10000.jsonl",
    },
    "GPT-4o mini": {
        "medmcqa": "outputs/gpt4omini_medmcqa_10000.jsonl",
        "pubmedqa": "outputs/gpt4omini_pubmedqa_10000.jsonl",
    },
    "Llama 3.1-8B Instruct (OpenRouter)": {
        "medmcqa": "outputs/or_llama31_8b_medmcqa_10000.jsonl",
        "pubmedqa": "outputs/or_llama31_8b_pubmedqa_10000.jsonl",
    },
}

# Output directory for figures & CSV
OUTDIR = Path("viz_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 2) Helpers
# ---------------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def acc(golds, preds):
    g = [str(x).upper() for x in golds]
    p = [str(x).upper() for x in preds]
    correct = sum(1 for x, y in zip(g, p) if x == y)
    return correct / max(1, len(g))

def macro_f1_mcq(golds, preds):
    labels = ["A","B","C","D"]
    return f1_score([str(x).upper() for x in golds],
                    [str(x).upper() for x in preds],
                    average="macro", labels=labels, zero_division=0)

def macro_f1_ynm(golds, preds):
    labels = ["YES","NO","MAYBE"]
    return f1_score([str(x).upper() for x in golds],
                    [str(x).upper() for x in preds],
                    average="macro", labels=labels, zero_division=0)

def norm_choice(text):
    t = (text or "").strip().upper()
    for ch in ["A","B","C","D"]:
        if t.startswith(ch):
            return ch
    # fallbacks
    if t and t[0] in {"A","B","C","D"}:
        return t[0]
    return t

def norm_ynm(text):
    t = (text or "").strip().upper().replace(".", "")
    if t.startswith("YES"): return "YES"
    if t.startswith("NO"): return "NO"
    if t.startswith("MAYBE"): return "MAYBE"
    if t in {"Y","TRUE"}: return "YES"
    if t in {"N","FALSE"}: return "NO"
    return "MAYBE" if "MAYBE" in t else t.split()[0] if t else ""

def compute_metrics(file_path, task):
    rows = list(load_jsonl(file_path))
    if not rows:
        return math.nan, math.nan, 0
    golds = [r.get("gold","") for r in rows]
    preds = [r.get("pred","") for r in rows]
    if task == "medmcqa":
        golds = [str(g).upper()[:1] for g in golds]
        preds = [norm_choice(p) for p in preds]
        return acc(golds, preds), macro_f1_mcq(golds, preds), len(rows)
    elif task == "pubmedqa":
        golds = [norm_ynm(g) for g in golds]
        preds = [norm_ynm(p) for p in preds]
        return acc(golds, preds), macro_f1_ynm(golds, preds), len(rows)
    else:
        raise ValueError("Unknown task")

# ---------------------------
# 3) Aggregate
# ---------------------------
records = []
for model, paths in FILES.items():
    # MedMCQA: accuracy & F1
    med_path = paths.get("medmcqa")
    pub_path = paths.get("pubmedqa")

    if med_path and os.path.exists(med_path):
        A_med, F1_med, N_med = compute_metrics(med_path, "medmcqa")
    else:
        A_med = F1_med = float("nan"); N_med = 0

    if pub_path and os.path.exists(pub_path):
        A_pub, F1_pub, N_pub = compute_metrics(pub_path, "pubmedqa")
    else:
        A_pub = F1_pub = float("nan"); N_pub = 0

    records.append({
        "Model": model,
        "MedMCQA_Accuracy": A_med,
        "MedMCQA_F1": F1_med,
        "MedMCQA_N": N_med,
        "PubMedQA_Accuracy": A_pub,
        "PubMedQA_F1": F1_pub,
        "PubMedQA_N": N_pub,
        "Avg_Accuracy": pd.Series([A_med, A_pub]).mean(skipna=True),
        "Avg_F1": pd.Series([F1_med, F1_pub]).mean(skipna=True),
    })

df = pd.DataFrame(records)
# Order models by Avg_Accuracy desc for consistent plotting
df = df.sort_values("Avg_Accuracy", ascending=False).reset_index(drop=True)

# Save CSV summary
df.to_csv(OUTDIR / "summary_metrics.csv", index=False)
print("Saved table:", OUTDIR / "summary_metrics.csv")
print(df[["Model","MedMCQA_Accuracy","MedMCQA_F1","PubMedQA_Accuracy","PubMedQA_F1","Avg_Accuracy","Avg_F1"]])

# ---------------------------
# 4) Visualization
# ---------------------------
plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

def annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        if not math.isnan(h):
            ax.annotate(f"{h:.3f}", (p.get_x()+p.get_width()/2, h),
                        ha="center", va="bottom", fontsize=9, rotation=0, xytext=(0,3), textcoords="offset points")

# --- Chart 1: Accuracy by Model (two groups: MedMCQA vs PubMedQA) ---
fig1, ax1 = plt.subplots(figsize=(10, 5.2))
x = range(len(df))
width = 0.38
ax1.bar([i - width/2 for i in x], df["MedMCQA_Accuracy"], width=width, label="MedMCQA")
ax1.bar([i + width/2 for i in x], df["PubMedQA_Accuracy"], width=width, label="PubMedQA")
ax1.set_xticks(list(x))
ax1.set_xticklabels(df["Model"], rotation=15, ha="right")
ax1.set_ylabel("Accuracy")
ax1.set_title("Accuracy by Model (Higher is better)")
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.legend()
annotate_bars(ax1)
fig1.tight_layout()
fig1.savefig(OUTDIR / "accuracy_by_model.png")

# --- Chart 2: Macro-F1 by Model (two groups) ---
fig2, ax2 = plt.subplots(figsize=(10, 5.2))
ax2.bar([i - width/2 for i in x], df["MedMCQA_F1"], width=width, label="MedMCQA")
ax2.bar([i + width/2 for i in x], df["PubMedQA_F1"], width=width, label="PubMedQA")
ax2.set_xticks(list(x))
ax2.set_xticklabels(df["Model"], rotation=15, ha="right")
ax2.set_ylabel("Macro-F1")
ax2.set_title("Macro-F1 by Model (Class-balanced; higher is better)")
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.legend()
annotate_bars(ax2)
fig2.tight_layout()
fig2.savefig(OUTDIR / "f1_by_model.png")

# --- Chart 3: Accuracy vs F1 (scatter), per dataset, per model ---
fig3, ax3 = plt.subplots(figsize=(8.5, 6))
# MedMCQA points
ax3.scatter(df["MedMCQA_Accuracy"], df["MedMCQA_F1"], s=55, label="MedMCQA")
for i, row in df.iterrows():
    ax3.annotate(row["Model"], (row["MedMCQA_Accuracy"], row["MedMCQA_F1"]), fontsize=8, xytext=(5,3), textcoords="offset points")

# PubMedQA points
ax3.scatter(df["PubMedQA_Accuracy"], df["PubMedQA_F1"], s=55, marker="s", label="PubMedQA")
for i, row in df.iterrows():
    ax3.annotate(row["Model"], (row["PubMedQA_Accuracy"], row["PubMedQA_F1"]), fontsize=8, xytext=(5,-10), textcoords="offset points")

ax3.set_xlabel("Accuracy")
ax3.set_ylabel("Macro-F1")
ax3.set_title("Accuracy vs Macro-F1 (per dataset)")
ax3.grid(True, linestyle="--", alpha=0.4)
ax3.legend()
fig3.tight_layout()
fig3.savefig(OUTDIR / "accuracy_vs_f1_scatter.png")

print("Saved figures in:", OUTDIR.resolve())
