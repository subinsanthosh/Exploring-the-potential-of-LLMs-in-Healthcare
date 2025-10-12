#!/usr/bin/env python3
# summ_graphs_like_paper.py
# Make separate plots for BLEU, ROUGE (1/2/L/Lsum), and BERTScore F1.

import json, argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)

    out = {}
    out["n"] = js.get("N") or js.get("n")

    r = js.get("ROUGE") or js.get("rouge") or {}
    def _f(x):
        try:
            return float(x)
        except Exception:
            return None

    # ROUGE
    out["rouge1"]   = _f(r.get("rouge1", js.get("rouge1")))
    out["rouge2"]   = _f(r.get("rouge2", js.get("rouge2")))
    rl              = r.get("rougeL", js.get("rougeL"))
    rls             = r.get("rougeLsum", js.get("rougeLsum", rl))
    out["rougeL"]   = _f(rl)
    out["rougeLsum"]= _f(rls)

    # BLEU
    bleu = js.get("BLEU") or js.get("bleu")
    if isinstance(bleu, dict) and "score" in bleu:
        bleu = bleu["score"]
    out["bleu"] = _f(bleu)

    # BERTScore
    b = js.get("BERTScore") or js.get("bertscore") or {}
    if isinstance(b, dict) and "f1" in b:
        out["bertscore_f1"] = _f(b["f1"])
    else:
        out["bertscore_f1"] = _f(js.get("bertscore_f1"))

    return out

def plot_bar(series: pd.Series, title: str, ylabel: str, out_path: Path):
    labels = series.index.tolist()
    values = series.fillna(0.0).tolist()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}" if v < 1.0 else f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Summarization metric graphs (separate per metric).")
    ap.add_argument("--out_dir", required=True, help="Where to save CSV and plots")
    # Default paths — override via flags if your filenames differ
    ap.add_argument("--llama",        default="outputs/or_llama31_8b_asclepius_10000.metrics.json")
    ap.add_argument("--grokmini",     default="outputs/openrouter_grok3mini_asclepius_1000.metrics.json")
    ap.add_argument("--gpt4omini",    default="outputs/gpt4omini_asclepius_10000.metrics.json")
    ap.add_argument("--gemini",       default="outputs/gemini_flashlite_asclepius_10000.metrics.json")
    ap.add_argument("--chatdoctorcpp",default="outputs/chatdoctor_cpp_asclepius_10000.metrics.json")
    args = ap.parse_args()

    paths = {
        "Llama3.1-8B (OpenRouter)": args.llama,
        "Grok3-Mini (OpenRouter)": args.grokmini,
        "GPT-4o-mini": args.gpt4omini,
        "Gemini-FlashLite": args.gemini,
        "ChatDoctor-CPP": args.chatdoctorcpp,
    }

    rows, missing = [], []
    for label, path in paths.items():
        p = Path(path)
        if not p.exists():
            missing.append((label, str(p)))
            continue
        m = load_metrics_json(str(p))
        m["label"] = label
        rows.append(m)

    if not rows:
        raise SystemExit("No metrics files found. Check the paths.")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).set_index("label").sort_index()
    # Save the table
    (out_dir / "summ_metrics_table.csv").write_text(df.to_csv(), encoding="utf-8")

    # === Separate visualizations ===
    # ROUGE-1
    if "rouge1" in df.columns:
        plot_bar(df["rouge1"], "ROUGE-1 by Model (higher is better)", "ROUGE-1",
                 out_dir / "rouge1_by_model.png")
    # ROUGE-2
    if "rouge2" in df.columns:
        plot_bar(df["rouge2"], "ROUGE-2 by Model (higher is better)", "ROUGE-2",
                 out_dir / "rouge2_by_model.png")
    # ROUGE-L
    if "rougeL" in df.columns:
        plot_bar(df["rougeL"], "ROUGE-L by Model (higher is better)", "ROUGE-L",
                 out_dir / "rougeL_by_model.png")
    # ROUGE-Lsum
    if "rougeLsum" in df.columns:
        plot_bar(df["rougeLsum"], "ROUGE-Lsum by Model (higher is better)", "ROUGE-Lsum",
                 out_dir / "rougeLsum_by_model.png")

    # BLEU
    if "bleu" in df.columns:
        plot_bar(df["bleu"], "sacreBLEU by Model (higher is better)", "BLEU",
                 out_dir / "bleu_by_model.png")

    # BERTScore F1
    if "bertscore_f1" in df.columns:
        plot_bar(df["bertscore_f1"], "BERTScore F1 by Model (higher is better)", "F1",
                 out_dir / "bertscore_f1_by_model.png")

    # Log any missing files
    if missing:
        print("Missing files:")
        for lab, pp in missing:
            print(f"  {lab}: {pp}")
    print(f"Saved CSV and plots to: {out_dir}")

if __name__ == "__main__":
    main()
