#!/usr/bin/env python3
# scripts/build_asclepius_summ_only.py

import sys, json

def main(in_path, out_path, limit=None):
    n = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            ex = json.loads(line)

            note = ex.get("note") or ex.get("text") or ex.get("question")
            ref = ex.get("reference_summary") or ex.get("summary") or ex.get("answer")

            if not note or not ref:
                continue

            obj = {
                "id": ex.get("id"),
                "text": note,
                "reference_summary": ref
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            n += 1
            if limit and n >= limit:
                break

    print(f"✅ Wrote {n} examples to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: build_asclepius_summ_only.py input.jsonl output.jsonl [limit]")
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(in_path, out_path, limit)
