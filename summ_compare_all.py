#!/usr/bin/env python3
# summ_compare_all.py — Auto-discover & graph summarization metrics for many runs
#
# Example:
#   python summ_compare_all.py --glob "outputs/*.metrics.json" --out_dir outputs/compare_all
#
# If you want to combine multiple folders:
#   python summ_compare_all.py --glob "outputs/*.metrics.json" --glob "more_out/*.metrics.json" --out_dir out/compare
#
# Requirements in venv:
#   pip install matplotlib pandas

import os, re, json, argparse, glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def _load_metrics_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    # normalize
    out = {}
    out["n"] = js.get("N") or js.get("n")
    r = js.get("ROUGE") or js.get("rouge") or {}
    def _f(x, default=None):
        try:
            return float(x)
        except Exception:
            return default
    out["rouge1"]   = _f(r.get("rouge1", js.get("rouge1")))
    out["rouge2"]   = _f(r.get("rouge2", js.get("rouge2")))
    rl_from_block   = r.get("rougeL", js.get("rougeL"))
    rls_from_block  = r.get("rougeLsum", js.get("rougeLsum", rl_from_block))
    out["rougeL"]   = _f(rl_from_block)
    out["rougeLsum"]= _f(rls_from_block)
    bleu = js.get("BLEU") or js.get("bleu")
    if isinstance(bleu, dict) and "score" in bleu: bleu = bleu["score"]
    out["bleu"] = _f(bleu)
    b = js.get("BERTScore") or js.get("bertscore") or {}
    if isinstance(b, dict) and "f1" in b:
        out["bertscore_f1"] = _f(b["f1"])
    else:
        out["bertscore_f1"] = _f(js.get("bertscore_f1"))
    # optional metadata common in your naming: infer label from filename
    out["path"] = path
    out["label"] = Path(path).stem  # e.g., openrouter_grok3mini_asclepius_10.metrics
    # friendlier label: try to pull model name from filename
    nice = out["label"]
    nice = re.sub(r"\.metrics$", "", nice)
    nice = nice.replace("_metrics", "")
    out["label"] = nice
    return out

def _plot_bar(labels, values, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    # annotate bars
    for i, v in enumerate(values):
        if v is None: continue
        ax.text(i, v, f"{v:.3f}" if v < 1.0 else f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Auto-compare summarization metrics and plot graphs.")
    ap.add_argument("--glob", action="append", required=True, help="Glob for metrics json, e.g. 'outputs/*.metrics.json'")
    ap.add_argument("--out_dir", required=True, help="Dir to save CSV and charts")
    ap.add_argument("--csv_name", default="summ_compare_all.csv")
    args = ap.parse_args()

    paths = []
    for g in args.glob:
        paths.extend(glob.glob(g))
    paths = sorted(set(p for p in paths if Path(p).exists()))
    if not paths:
        raise SystemExit("No metrics files found for the given --glob.")

    rows = []
    for p in paths:
        try:
            rows.append(_load_metrics_json(p))
        except Exception as e:
            print(f"[warn] skipping {p}: {e}")

    if not rows:
        raise SystemExit("No valid metrics parsed.")

    df = pd.DataFrame(rows).set_index("label")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / args.csv_name
    df.to_csv(csv_path)
    print(f"Saved table to {csv_path}")

    labels = df.index.tolist()

    # Prefer ROUGE-Lsum if available
    if "rougeLsum" in df.columns and df["rougeLsum"].notna().any():
        _plot_bar(labels, df["rougeLsum"].fillna(0.0).tolist(),
                  "ROUGE-Lsum (higher is better)", "ROUGE-Lsum",
                  out_dir / "rougeLsum.png")
    elif "rougeL" in df.columns and df["rougeL"].notna().any():
        _plot_bar(labels, df["rougeL"].fillna(0.0).tolist(),
                  "ROUGE-L (higher is better)", "ROUGE-L",
                  out_dir / "rougeL.png")

    if "rouge1" in df.columns and df["rouge1"].notna().any():
        _plot_bar(labels, df["rouge1"].fillna(0.0).tolist(),
                  "ROUGE-1 (higher is better)", "ROUGE-1",
                  out_dir / "rouge1.png")

    if "rouge2" in df.columns and df["rouge2"].notna().any():
        _plot_bar(labels, df["rouge2"].fillna(0.0).tolist(),
                  "ROUGE-2 (higher is better)", "ROUGE-2",
                  out_dir / "rouge2.png")

    if "bleu" in df.columns and df["bleu"].notna().any():
        _plot_bar(labels, df["bleu"].fillna(0.0).tolist(),
                  "sacreBLEU (higher is better)", "BLEU",
                  out_dir / "bleu.png")

    if "bertscore_f1" in df.columns and df["bertscore_f1"].notna().any():
        _plot_bar(labels, df["bertscore_f1"].fillna(0.0).tolist(),
                  "BERTScore F1 (higher is better)", "F1",
                  out_dir / "bertscore_f1.png")

    # quick ranks CSV (descending; skip NAs)
    rank_cols = [c for c in ["rougeLsum","rougeL","rouge1","rouge2","bleu","bertscore_f1"] if c in df.columns]
    ranks = {}
    for c in rank_cols:
        s = df[c].dropna().sort_values(ascending=False)
        ranks[c+"_rank"] = s.index.tolist()
    if ranks:
        with open(out_dir / "ranks.txt", "w", encoding="utf-8") as f:
            for k, v in ranks.items():
                f.write(f"{k}:\n")
                for i, lab in enumerate(v, 1):
                    f.write(f"  {i}. {lab} ({df.loc[lab, k.replace('_rank','')]})\n")
        print(f"Saved {out_dir / 'ranks.txt'}")

    print("Done.")

if __name__ == "__main__":
    main()
