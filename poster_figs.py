#!/usr/bin/env python3
# poster_figs_final.py — final polished version for poster visuals (brand colours applied)

import os, json, math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ===== Brand colours =====
COL_PRIMARY = "#001b47"   # deep blue (single + first bar)
COL_SECOND  = "#9e3638"   # crimson red (second bar)

# ============ CONFIG: set your files here ============

QA_FILES = {
    "ChatDoctor": {
        "medmcqa": "outputs/llama_chatdoctor_medmcqa_10000.jsonl",
        "pubmedqa": "outputs/llama_chatdoctor_pubmedqa_10000.jsonl",
    },
    "Gemini 2.5 FlashLite": {
        "medmcqa": "outputs/gemini_flashlite_medmcqa_10000.jsonl",
        "pubmedqa": "outputs/gemini_flashlite_pubmedqa_10000.jsonl",
    },
    "Grok 3 Mini": {
        "medmcqa": "outputs/grok3_medmcqa_10000.jsonl",
        "pubmedqa": "outputs/grok3_pubmedqa_10000.jsonl",
    },
    "GPT-4o mini": {
        "medmcqa": "outputs/gpt4omini_medmcqa_10000.jsonl",
        "pubmedqa": "outputs/gpt4omini_pubmedqa_10000.jsonl",
    },
    "Llama-3.1-8B": {
        "medmcqa": "outputs/or_llama31_8b_medmcqa_10000.jsonl",
        "pubmedqa": "outputs/or_llama31_8b_pubmedqa_10000.jsonl",
    },
}

SUMM_FILES = {
    "ChatDoctor":           "outputs/chatdoctor_cpp_asclepius_10000.metrics.json",
    "Gemini 2.5 FlashLite": "outputs/gemini_flashlite_asclepius_10000.metrics.json",
    "Grok 3 Mini":          "outputs/openrouter_grok3mini_asclepius_1000.metrics.json",
    "GPT-4o mini":          "outputs/gpt4omini_asclepius_10000.metrics.json",
    "Llama-3.1-8B":         "outputs/or_llama31_8b_asclepius_10000.metrics.json",
}

OUTDIR = Path("viz_poster"); OUTDIR.mkdir(parents=True, exist_ok=True)

# ============ Helpers ============

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

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
        if t.startswith(ch): return ch
    return t[:1] if t[:1] in {"A","B","C","D"} else t

def norm_ynm(text):
    t = (text or "").strip().upper().replace(".", "")
    if t.startswith("YES"): return "YES"
    if t.startswith("NO"): return "NO"
    if t.startswith("MAYBE"): return "MAYBE"
    if t in {"Y","TRUE"}: return "YES"
    if t in {"N","FALSE"}: return "NO"
    return "MAYBE" if "MAYBE" in t else (t.split()[0] if t else "")

def compute_qa_metrics(file_path, task):
    if not (file_path and os.path.exists(file_path)):
        return math.nan, math.nan
    rows = list(load_jsonl(file_path))
    if not rows:
        return math.nan, math.nan
    golds = [r.get("gold","") for r in rows]
    preds = [r.get("pred","") for r in rows]
    if task == "medmcqa":
        golds = [str(g).upper()[:1] for g in golds]
        preds = [norm_choice(p) for p in preds]
        return acc(golds, preds), macro_f1_mcq(golds, preds)
    elif task == "pubmedqa":
        golds = [norm_ynm(g) for g in golds]
        preds = [norm_ynm(p) for p in preds]
        return acc(golds, preds), macro_f1_ynm(golds, preds)
    return math.nan, math.nan

def _f(x):
    try: return float(x)
    except Exception: return math.nan

def load_summ_metrics(path):
    if not (path and os.path.exists(path)):
        return math.nan, math.nan
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    r = js.get("ROUGE") or js.get("rouge") or {}
    rougeLsum = r.get("rougeLsum", js.get("rougeLsum", r.get("rougeL", js.get("rougeL"))))
    b = js.get("BERTScore") or js.get("bertscore") or {}
    if isinstance(b, dict) and "f1" in b:
        bert_f1 = b["f1"]
    else:
        bert_f1 = js.get("bertscore_f1")
    return _f(rougeLsum), _f(bert_f1)

def setup_style():
    plt.rcParams.update({
        "figure.dpi": 220,
        "savefig.dpi": 350,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
    })

# ============ Plotting ============

def single_bar(df, metric, title, ylabel, out_png):
    models = df["Model"].tolist()
    vals = df[metric].tolist()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bars = ax.bar(range(len(models)), vals, width=0.55, color=COL_PRIMARY)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=18, ha="center")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.margins(x=0.06)  # slight right/left breathing room

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def grouped_bar(df, m1, m2, title, ylabel1, ylabel2, out_png):
    models = df["Model"].tolist()
    a = df[m1].tolist()
    b = df[m2].tolist()

    x = range(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar([i - width/2 for i in x], a, width=width, label=ylabel1, color=COL_PRIMARY)
    ax.bar([i + width/2 for i in x], b, width=width, label=ylabel2, color=COL_SECOND)

    # Center tick exactly between the pair
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=18, ha="center")

    ax.margins(x=0.05)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    for i, v in enumerate(a):
        if not math.isnan(v):
            ax.text(i - width/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(b):
        if not math.isnan(v):
            ax.text(i + width/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=350)
    plt.close(fig)

# ============ Main ============

def main():
    setup_style()

    # ---- MedMCQA (Accuracy only) ----
    med_rows = []
    for model, paths in QA_FILES.items():
        A, _ = compute_qa_metrics(paths.get("medmcqa",""), "medmcqa")
        med_rows.append({"Model": model, "Accuracy": A})
    df_med = pd.DataFrame(med_rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    df_med.to_csv(OUTDIR / "medmcqa_poster_table.csv", index=False)
    single_bar(df_med, "Accuracy", "MedMCQA — Accuracy", "Accuracy", OUTDIR / "medmcqa_poster.png")

    # ---- PubMedQA (Accuracy + Macro-F1) ----
    pub_rows = []
    for model, paths in QA_FILES.items():
        A, F1 = compute_qa_metrics(paths.get("pubmedqa",""), "pubmedqa")
        pub_rows.append({"Model": model, "Accuracy": A, "MacroF1": F1})
    df_pub = pd.DataFrame(pub_rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    df_pub.to_csv(OUTDIR / "pubmedqa_poster_table.csv", index=False)
    grouped_bar(df_pub, "Accuracy", "MacroF1",
                "PubMedQA — Accuracy & Macro-F1",
                "Accuracy", "Macro-F1", OUTDIR / "pubmedqa_poster.png")

    # ---- Asclepius Summarization (ROUGE-Lsum + BERTScore F1) ----
    summ_rows = []
    for model, path in SUMM_FILES.items():
        rouge, bert = load_summ_metrics(path)
        summ_rows.append({"Model": model, "ROUGE_Lsum": rouge, "BERTScore_F1": bert})
    df_summ = pd.DataFrame(summ_rows).sort_values("ROUGE_Lsum", ascending=False).reset_index(drop=True)
    df_summ.to_csv(OUTDIR / "asclepius_poster_table.csv", index=False)
    grouped_bar(df_summ, "ROUGE_Lsum", "BERTScore_F1",
                "Asclepius Summarization — ROUGE-Lsum & BERTScore F1",
                "ROUGE-Lsum", "BERTScore F1", OUTDIR / "asclepius_poster.png")

    print("\n✅ Poster figures saved in:", OUTDIR.resolve())
    print(" - medmcqa_poster.png")
    print(" - pubmedqa_poster.png")
    print(" - asclepius_poster.png\n")

if __name__ == "__main__":
    main()
