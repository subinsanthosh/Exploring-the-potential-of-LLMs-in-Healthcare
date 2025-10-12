#!/usr/bin/env python3
# eval_viz_acc_f1_split.py — Accuracy + Macro-F1, separate charts for MedMCQA and PubMedQA

import json, os, math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ---------------------------
# 1) Point these to YOUR files
# ---------------------------
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

# Output directory for figures & CSVs
OUTDIR = Path("viz_outputs_acc_f1_split")
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
    return sum(1 for x, y in zip(g, p) if x == y) / max(1, len(g))

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
    return t[0] if t[:1] in {"A","B","C","D"} else t

def norm_ynm(text):
    t = (text or "").strip().upper().replace(".", "")
    if t.startswith("YES"): return "YES"
    if t.startswith("NO"): return "NO"
    if t.startswith("MAYBE"): return "MAYBE"
    if t in {"Y","TRUE"}: return "YES"
    if t in {"N","FALSE"}: return "NO"
    return "MAYBE" if "MAYBE" in t else (t.split()[0] if t else "")

def compute_metrics(file_path, task):
    if not os.path.exists(file_path):
        return math.nan, math.nan, 0
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

def annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        if not math.isnan(h):
            ax.annotate(f"{h:.3f}", (p.get_x()+p.get_width()/2, h),
                        ha="center", va="bottom", fontsize=9, xytext=(0,3), textcoords="offset points")

def make_bars(df, col, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(df["Model"], df[col])
    ax.set_xticklabels(df["Model"], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    annotate_bars(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / filename)
    plt.close(fig)

# Style (neutral defaults)
plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

# ---------------------------
# 3) Compute per-dataset tables
# ---------------------------
med_rows, pub_rows = [], []

for model, paths in FILES.items():
    # MedMCQA
    A_med, F1_med, N_med = compute_metrics(paths.get("medmcqa",""), "medmcqa")
    med_rows.append({"Model": model, "Accuracy": A_med, "MacroF1": F1_med, "N": N_med})

    # PubMedQA
    A_pub, F1_pub, N_pub = compute_metrics(paths.get("pubmedqa",""), "pubmedqa")
    pub_rows.append({"Model": model, "Accuracy": A_pub, "MacroF1": F1_pub, "N": N_pub})

df_med = pd.DataFrame(med_rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
df_pub = pd.DataFrame(pub_rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)

# Save CSVs
df_med.to_csv(OUTDIR / "medmcqa_summary.csv", index=False)
df_pub.to_csv(OUTDIR / "pubmedqa_summary.csv", index=False)

print("Saved:", OUTDIR / "medmcqa_summary.csv")
print("Saved:", OUTDIR / "pubmedqa_summary.csv")

# ---------------------------
# 4) Visualizations (separate for each dataset; Accuracy and Macro-F1)
# ---------------------------
# MedMCQA
make_bars(df_med, "Accuracy", "MedMCQA — Accuracy by Model (higher is better)", "Accuracy",
          "medmcqa_accuracy_by_model.png")
make_bars(df_med, "MacroF1", "MedMCQA — Macro-F1 by Model (higher is better)", "Macro-F1",
          "medmcqa_f1_by_model.png")

# PubMedQA
make_bars(df_pub, "Accuracy", "PubMedQA — Accuracy by Model (higher is better)", "Accuracy",
          "pubmedqa_accuracy_by_model.png")
make_bars(df_pub, "MacroF1", "PubMedQA — Macro-F1 by Model (higher is better)", "Macro-F1",
          "pubmedqa_f1_by_model.png")

print("Saved figures in:", OUTDIR.resolve())
