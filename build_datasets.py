# build_datasets.py
# Build JSONL datasets for:
# - MedMCQA (MCQ)
# - PubMedQA (Yes/No/Maybe)
# - Asclepius Synthetic Clinical Notes (summarisation)
#
# Usage example:
#   python build_datasets.py \
#     --medmcqa_out data/medmcqa_mid.jsonl \
#     --pubmedqa_out data/pubmedqa_mid.jsonl \
#     --asclepius_out data/asclepius_mid.jsonl \
#     --sizes 20000 1000 10000 \
#     --pubmedqa_config pqa_artificial

import argparse, json, os, random, re
from typing import List, Dict, Any
from datasets import load_dataset

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------------- MedMCQA (MCQ) ----------------------
def build_medmcqa(out_path: str, size: int):
    """
    Build MedMCQA JSONL with fields:
      id, question, options (list of "A. ...", ...), answer (A/B/C/D)
    Pulls from TRAIN (+VALIDATION) so you can get >> 4k examples.
    """
    ds = load_dataset("openlifescienceai/medmcqa")
    parts = []
    if "train" in ds:       parts.extend(list(ds["train"]))
    if "validation" in ds:  parts.extend(list(ds["validation"]))
    if not parts:           parts = list(next(iter(ds.values())))  # fallback

    def row_to_record(r):
        # question
        q = r.get("question") or r.get("ques") or r.get("Question") or ""
        # options (either unified "options" list or opa/opb/opc/opd)
        if isinstance(r.get("options"), list) and len(r["options"]) >= 4:
            opts = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(r["options"][:4])]
        else:
            opts = [
                f"A. {r.get('opa') or r.get('A') or r.get('op1') or ''}",
                f"B. {r.get('opb') or r.get('B') or r.get('op2') or ''}",
                f"C. {r.get('opc') or r.get('C') or r.get('op3') or ''}",
                f"D. {r.get('opd') or r.get('D') or r.get('op4') or ''}",
            ]
        # gold answer -> normalize to A/B/C/D
        raw = r.get("cop") or r.get("answer") or r.get("label")
        amap = {"1":"A","2":"B","3":"C","4":"D","a":"A","b":"B","c":"C","d":"D"}
        if isinstance(raw, int):
            gold = ["A","B","C","D"][raw] if 0 <= raw < 4 else "A"
        else:
            s = str(raw or "").strip().lower()
            gold = amap.get(s, s.upper() if s in {"a","b","c","d","A","B","C","D"} else "A")
            if gold not in {"A","B","C","D"}: gold = "A"
        return {
            "id": str(r.get("id") or r.get("qid") or random.getrandbits(32)),
            "question": q,
            "options": opts,
            "answer": gold,
        }

    rows = [row_to_record(r) for r in parts]
    random.seed(42)
    if size > 0 and size < len(rows):
        rows = random.sample(rows, k=size)  # sample from full set (train+val)
    write_jsonl(out_path, rows)
    print(f"Wrote {out_path} {len(rows)}")

# ---------------------- PubMedQA (Y/N/Maybe) ----------------------
def build_pubmedqa(out_path: str, size: int, config: str = "pqa_artificial"):
    """
    Build PubMedQA JSONL with fields:
      id, context, question, answer (yes/no/maybe)
    """
    ds = load_dataset("qiaojin/PubMedQA", config)

    def norm(v: Any) -> str:
        s = str(v).strip().lower()
        if s in {"yes","y","1","true"}: return "yes"
        if s in {"no","n","0","false"}: return "no"
        return "maybe"

    rows = []
    for _, split in ds.items():
        for r in split:
            ctx = r.get("context") or r.get("contexts") or r.get("CONTEXTS")
            if isinstance(ctx, list): ctx = " ".join(ctx)
            q = r.get("question") or r.get("QUESTION") or ""
            lab = r.get("final_decision") or r.get("label") or r.get("LONG_ANSWER")
            rows.append({
                "id": str(r.get("id") or r.get("qid") or random.getrandbits(32)),
                "context": ctx or "",
                "question": q,
                "answer": norm(lab),
            })

    random.seed(42)
    if size > 0 and size < len(rows):
        rows = random.sample(rows, k=size)
    write_jsonl(out_path, rows)
    print(f"Wrote {out_path} {len(rows)}")

# ---------------------- Asclepius (Summarisation) ----------------------
def build_asclepius(out_path: str, size: int):
    """
    Build Asclepius JSONL with fields:
      id, note, reference_summary

    Your split has columns: ['answer','note','patient_id','question','task'].
    We:
      - Filter rows where task/question mentions "summar" (case-insensitive).
      - Map note -> note, answer -> reference_summary.
      - If no rows match, fall back to using ALL rows with answer as the target.
    """
    from datasets import load_dataset
    import re, random, os, json

    ds = load_dataset("starmpcc/Asclepius-Synthetic-Clinical-Notes")
    split = ds.get("train") or list(ds.values())[0]

    def is_summarization(rec):
        t = (rec.get("task") or "") + " " + (rec.get("question") or "")
        return re.search(r"summar", t, re.I) is not None

    # keep only summarisation items if present
    summar_rows = [r for r in split if is_summarization(r)]
    rows_src = summar_rows if summar_rows else list(split)

    def to_record(r):
        return {
            "id": str(r.get("id") or r.get("patient_id") or random.getrandbits(32)),
            "note": r["note"],
            "reference_summary": r["answer"],  # use 'answer' as gold summary
        }

    rows = [to_record(r) for r in rows_src]

    random.seed(42)
    if size > 0 and size < len(rows):
        rows = random.sample(rows, k=size)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path} {len(rows)} (summarisation-only: {len(summar_rows)} rows)")


# ---------------------- CLI ----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--medmcqa_out", default="data/medmcqa_mid.jsonl")
    ap.add_argument("--pubmedqa_out", default="data/pubmedqa_mid.jsonl")
    ap.add_argument("--asclepius_out", default="data/asclepius_mid.jsonl")
    ap.add_argument("--sizes", nargs=3, type=int, default=[20000, 1000, 10000],
                    help="sizes for MedMCQA, PubMedQA, Asclepius (0 = use ALL available)")
    ap.add_argument("--pubmedqa_config", default="pqa_artificial",
                    choices=["pqa_artificial", "pqa_labeled"])
    args = ap.parse_args()

    build_medmcqa(args.medmcqa_out, args.sizes[0])
    build_pubmedqa(args.pubmedqa_out, args.sizes[1], config=args.pubmedqa_config)
    build_asclepius(args.asclepius_out, args.sizes[2])
