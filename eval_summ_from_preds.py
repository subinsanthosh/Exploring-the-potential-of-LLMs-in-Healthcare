#!/usr/bin/env python3
# eval_summ_from_preds.py
import argparse, json
from pathlib import Path

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Predictions JSONL from llm_eval --task summ")
    ap.add_argument("--out_json", default=None, help="Where to save metrics JSON (default: <preds>.metrics.json)")
    ap.add_argument("--out_csv",  default=None, help="Optional: save per-example gold/pred to CSV")
    args = ap.parse_args()

    preds_path = Path(args.preds)
    if args.out_json is None:
        args.out_json = str(preds_path.with_suffix(preds_path.suffix + ".metrics.json"))

    golds, preds = [], []
    rows = []
    for ex in load_jsonl(args.preds):
        g = (ex.get("gold") or ex.get("reference_summary") or "").strip()
        p = (ex.get("pred") or "").strip()
        if g and p:
            golds.append(g)
            preds.append(p)
            rows.append({"id": ex.get("id"), "gold": g, "pred": p})

    n = len(golds)
    if n == 0:
        print(f"[err] No valid (gold, pred) pairs found in {args.preds}.")
        return

    rouge_out, bleu_out, bert_out = {}, None, {}

    # ------------ ROUGE ------------
    try:
        from evaluate import load as load_metric
        rouge = load_metric("rouge")
        r = rouge.compute(predictions=preds, references=golds)

        def _safe_rouge(v):
            # Newer evaluate may return float; older returns object with .mid.fmeasure
            try:
                return float(getattr(v, "mid").fmeasure)
            except Exception:
                return float(v)

        rouge_out = {k: round(_safe_rouge(v), 4) for k, v in r.items()}
    except Exception as e:
        print("[warn] ROUGE failed:", e)

    # ------------ sacreBLEU ------------
    try:
        sacre = load_metric("sacrebleu")
        refs = [[g] for g in golds]
        b = sacre.compute(predictions=preds, references=refs)
        bleu_out = round(float(b["score"]), 4)
    except Exception as e:
        print("[warn] sacreBLEU failed:", e)

    # ------------ BERTScore ------------
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(preds, golds, lang="en", rescale_with_baseline=True)
        bert_out = {
            "precision": round(float(P.mean()), 4),
            "recall":    round(float(R.mean()), 4),
            "f1":        round(float(F1.mean()), 4),
        }
    except Exception as e:
        print("[warn] BERTScore failed:", e)

    # ------------ Print + Save ------------
    print("== Summarization Metrics ==")
    if rouge_out: print("ROUGE:", rouge_out)
    if bleu_out is not None: print("BLEU :", bleu_out)
    if bert_out: print("BERTScore:", bert_out)
    print("N evaluated:", n)

    out_payload = {
        "n": n,
        "rouge": rouge_out,
        "bleu": bleu_out,
        "bertscore": bert_out,
        "source_preds_file": str(preds_path),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {args.out_json}")

    if args.out_csv:
        try:
            import csv
            Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["id", "gold", "pred"])
                w.writeheader()
                w.writerows(rows)
            print(f"Saved pairs to {args.out_csv}")
        except Exception as e:
            print("[warn] CSV save failed:", e)

if __name__ == "__main__":
    main()
