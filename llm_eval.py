#!/usr/bin/env python3
# llm_eval.py  — fast, large-run friendly with input truncation + summary cleaning

import argparse, json, time, sys, os, re
from pathlib import Path

# ---- Adapters (providers) ----
from adapters import grok_adapter as GROK
from adapters import grok_fast_adapter as GROK_FAST
from adapters import openrouter_adapter as OR
from adapters import gemini_adapter as GEM
from adapters import chatdoctor_adapter as CHD

# Optional: local gguf (chatdoctor_cpp)
try:
    from adapters import chatdoctor_cpp_adapter as CHD_CPP
    HAVE_CHD_CPP = True
except Exception:
    HAVE_CHD_CPP = False

try:
    from adapters import openai_adapter as OAI
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False


# ------------------------
# I/O helpers
# ------------------------
def load_jsonl(path, limit=None, skip=0):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if skip > 0:
                skip -= 1
                continue
            yield json.loads(line)
            n += 1
            if limit and n >= limit:
                break

def append_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def file_line_count(path):
    if not path or not Path(path).exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ------------------------
# Prompt formatting
# ------------------------
def format_prompt_mcq(ex):
    q = ex.get("question") or ex.get("ques")
    opts = ex.get("options") or {}

    if isinstance(opts, dict):
        a = opts.get("A") or opts.get("a") or ""
        b = opts.get("B") or opts.get("b") or ""
        c = opts.get("C") or opts.get("c") or ""
        d = opts.get("D") or opts.get("d") or ""
    elif isinstance(opts, list):
        a, b, c, d = (opts + ["", "", "", ""])[:4]
    else:
        a = b = c = d = ""

    prompt = (
        "You are a medical QA assistant. Answer with a single letter: A, B, C, or D.\n\n"
        f"Question: {q}\n"
        f"A) {a}\n"
        f"B) {b}\n"
        f"C) {c}\n"
        f"D) {d}\n\n"
        "Answer (just one letter):"
    )
    return prompt

def format_prompt_ynm(ex):
    q = ex.get("question") or ex.get("ques")
    prompt = (
        "You are a biomedical QA assistant. Respond with only one word: YES, NO, or MAYBE.\n\n"
        f"Question: {q}\n\n"
        "Answer (YES/NO/MAYBE only):"
    )
    return prompt


# ------------------------
# Normalization helpers
# ------------------------
def normalize_choice(text):
    t = (text or "").strip().upper()
    for ch in ["A", "B", "C", "D"]:
        if t.startswith(ch):
            return ch
    if " A" in f" {t} ": return "A"
    if " B" in f" {t} ": return "B"
    if " C" in f" {t} ": return "C"
    if " D" in f" {t} ": return "D"
    return t[:1] if t[:1] in {"A","B","C","D"} else t

def normalize_ynm(text):
    t = (text or "").strip().upper().replace(".", "")
    if t.startswith("YES"): return "YES"
    if t.startswith("NO"): return "NO"
    if t.startswith("MAYBE"): return "MAYBE"
    if t in {"Y","YA","YEP","TRUE"}: return "YES"
    if t in {"N","NOPE","FALSE"}: return "NO"
    return "MAYBE" if "MAYBE" in t else (t if t in {"YES","NO","MAYBE"} else t.split()[0] if t else "")

def _norm(s: str) -> str:
    return " ".join((s or "").strip().split())


# ------------------------
# Tokenizer-aware truncation (speeds up summarization massively)
# ------------------------
_tok_cache = {}

def _get_tok(model_name):
    """
    Tries to load a tokenizer for HF models. For API providers (Gemini/OpenAI/OpenRouter),
    we just return None and fall back to char-based truncation.
    """
    if model_name in _tok_cache:
        return _tok_cache[model_name]
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        _tok_cache[model_name] = tok
        return tok
    except Exception:
        _tok_cache[model_name] = None
        return None

def truncate_text(text: str, model_name: str, max_tokens: int = 1024, max_chars_fallback: int = 4000) -> str:
    if not text:
        return text
    tok = _get_tok(model_name)
    if tok is None:
        # Fallback: approx ~1k tokens ≈ 4k chars (good enough and fast)
        return text[:max_chars_fallback]
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    return tok.decode(ids, skip_special_tokens=True)


# ------------------------
# Summary cleaner (removes Grok-style meta chatter)
# ------------------------
_META_PATTERNS = [
    r"^\s*first, the user (said|asked).*?$",
    r"^\s*the (user|task) (is|was):.*?$",
    r"^\s*analysis:.*?$",
    r"^\s*reasoning:.*?$",
    r"^\s*i (need|have) to (write|provide).*?$",
    r"^\s*this (seems|looks) (straightforward|simple).*?$",
]

def clean_generative_summary(text: str) -> str:
    """
    - Strips obvious instruction-echo meta lines
    - Normalizes whitespace
    - Keeps the model's actual summary content
    """
    if not text:
        return ""
    # Normalize line breaks and split into lines/sentences
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    # Drop leading meta-like lines
    cleaned_lines = []
    for i, ln in enumerate(lines):
        low = ln.lower()
        if i == 0 and any(re.match(p, low) for p in _META_PATTERNS):
            # skip the first meta line and continue (only for the leading line)
            continue
        cleaned_lines.append(ln)
    out = " ".join(cleaned_lines).strip()
    # If still empty, fallback to original
    if not out:
        out = text.strip()
    # Collapse whitespace
    out = re.sub(r"\s+", " ", out).strip()
    return out


# ------------------------
# Model wrapper
# ------------------------
class Model:
    def __init__(self, provider, name):
        self.provider = provider.lower()
        self.name = name

    def generate(self, prompt, max_tokens=256):
        if self.provider == "gemini":
            return GEM.generate(self.name, prompt, max_tokens=max_tokens)
        elif self.provider == "chatdoctor":
            return CHD.generate(self.name, prompt, max_tokens=max_tokens)
        elif self.provider == "grok":
            return GROK.generate(self.name, prompt, max_tokens=max_tokens)
        elif self.provider == "grok_fast":
            return GROK_FAST.generate(self.name, prompt, max_tokens=max_tokens)
        elif self.provider == "openrouter":
            return OR.generate(self.name, prompt, max_tokens=max_tokens)
        elif self.provider == "chatdoctor_cpp":
            if not HAVE_CHD_CPP:
                raise RuntimeError("chatdoctor_cpp adapter not available")
            return CHD_CPP.generate(self.name, prompt, max_tokens=max_tokens)
        elif self.provider == "openai":
            if not HAVE_OPENAI:
                raise RuntimeError("openai_adapter not available")
            return OAI.generate(self.name, prompt, max_tokens=max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate_batch(self, prompts, max_tokens=256):
        if self.provider == "chatdoctor":
            return CHD.generate_batch(self.name, prompts, max_tokens=max_tokens)
        elif self.provider == "chatdoctor_cpp":
            if not HAVE_CHD_CPP:
                raise RuntimeError("chatdoctor_cpp adapter not available")
            return CHD_CPP.generate_batch(self.name, prompts, max_tokens=max_tokens)
        # fallback loop for providers without batch support
        outs = []
        for p in prompts:
            outs.append(self.generate(p, max_tokens=max_tokens))
        return outs


# ------------------------
# Tasks
# ------------------------
def run_qa_mcq(model, dataset, limit=None, out_path=None, resume=False):
    golds, preds, recs = [], [], []
    skip = file_line_count(out_path) if (resume and out_path) else 0
    processed = 0
    rows_to_append = []

    for i, ex in enumerate(load_jsonl(dataset, limit, skip=skip), 1):
        prompt = format_prompt_mcq(ex)
        text = model.generate(prompt, max_tokens=6)
        pred = normalize_choice(text)
        gold = (ex.get("answer") or ex.get("gold") or "").strip().upper()[:1]

        rec = {"id": ex.get("id"), "question": ex.get("question"), "gold": gold, "pred": pred}
        golds.append(gold); preds.append(pred); recs.append(rec)

        if out_path:
            rows_to_append.append({"gold": gold, "pred": pred, **rec})
            if len(rows_to_append) >= 50:
                append_jsonl(out_path, rows_to_append); rows_to_append = []

        processed += 1
        if processed % 50 == 0:
            print(f"...processed {processed} (+{skip} resumed)", flush=True)

    if out_path and rows_to_append:
        append_jsonl(out_path, rows_to_append)

    return golds, preds, recs

def run_qa_ynm(model, dataset, limit=None, out_path=None, resume=False):
    golds, preds, recs = [], [], []
    skip = file_line_count(out_path) if (resume and out_path) else 0
    processed = 0
    rows_to_append = []

    for i, ex in enumerate(load_jsonl(dataset, limit, skip=skip), 1):
        prompt = format_prompt_ynm(ex)
        last = ""
        for attempt in range(4):
            try:
                last = model.generate(prompt, max_tokens=16)
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"[warn] generate failed (attempt {attempt+1}/4): {e}. Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)

        pred = normalize_ynm(last)
        gold = normalize_ynm((ex.get("answer") or ex.get("gold") or "").strip().upper())

        rec = {"id": ex.get("id"), "question": ex.get("question"), "gold": gold, "pred": pred}
        golds.append(gold); preds.append(pred); recs.append(rec)

        if out_path:
            rows_to_append.append({"gold": gold, "pred": pred, **rec})
            if len(rows_to_append) >= 50:
                append_jsonl(out_path, rows_to_append); rows_to_append = []

        processed += 1
        if processed % 50 == 0:
            print(f"...processed {processed} (+{skip} resumed)", flush=True)

    if out_path and rows_to_append:
        append_jsonl(out_path, rows_to_append)

    return golds, preds, recs

# Short, efficient instruction to speed generation
def _summ_prompt(text: str) -> str:
    return (
        "You are a biomedical summarization assistant. "
        "Write a concise 2–3 sentence discharge summary. "
        "No preamble; return only the summary text.\n\n"
        f"{text}\n\nSummary:"
    )

def run_summ(model, dataset, limit=None, batch_size=1, max_tokens=64, out_path=None, resume=False,
             max_input_tokens=1024, max_input_chars_fallback=4000):
    """
    Fast summarization:
      - truncates input to ~max_input_tokens (or ~max_input_chars_fallback)
      - shorter generations (default 64)
      - small batch (default 1) to avoid MPS/offload thrash
      - incremental append + resume
      - cleans model outputs to remove instruction-echo meta lines
    """
    golds, preds, recs = [], [], []
    rows_to_append = []

    skip = file_line_count(out_path) if (resume and out_path) else 0
    processed = 0

    batch_prompts, batch_meta = [], []

    def _flush():
        nonlocal golds, preds, recs, batch_prompts, batch_meta, rows_to_append, processed
        if not batch_prompts:
            return
        outs = model.generate_batch(batch_prompts, max_tokens=max_tokens)
        for out, meta in zip(outs, batch_meta):
            cleaned = clean_generative_summary(out)
            golds.append(meta["gold"]); preds.append(cleaned); recs.append(meta["rec"])
            if out_path:
                rows_to_append.append({"gold": meta["gold"], "pred": cleaned, **meta["rec"]})
        batch_prompts.clear(); batch_meta.clear()

        if out_path and rows_to_append and len(rows_to_append) >= 20:
            append_jsonl(out_path, rows_to_append); rows_to_append = []

    for i, ex in enumerate(load_jsonl(dataset, limit, skip=skip), 1):
        raw_text = ex.get("text") or ex.get("note") or ex.get("question")
        gold = ex.get("reference_summary") or ex.get("answer") or ex.get("gold")
        if not raw_text or not gold:
            continue

        # --- truncate long inputs (HUGE speedup) ---
        text = truncate_text(
            raw_text, model.name,
            max_tokens=max_input_tokens,
            max_chars_fallback=max_input_chars_fallback
        )

        prompt = _summ_prompt(text)
        batch_prompts.append(prompt)
        gold_norm = _norm(gold)
        batch_meta.append({"gold": gold_norm, "rec": {"id": ex.get("id"), "text": text, "gold": gold_norm}})

        if len(batch_prompts) >= batch_size:
            _flush()

        processed += 1
        if processed % 10 == 0:
            print(f"...processed {processed} (+{skip} resumed) | batch={len(batch_prompts)}", flush=True)

    _flush()
    if out_path and rows_to_append:
        append_jsonl(out_path, rows_to_append)

    return golds, preds, recs


# ------------------------
# Metrics
# ------------------------
def accuracy(golds, preds):
    correct = sum(1 for g, p in zip(golds, preds) if str(g).upper() == str(p).upper())
    return correct / max(1, len(golds))


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["qa_mcq", "qa_ynm", "summ"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)

    # Fast defaults for summarization on Mac
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--resume", action="store_true", help="resume from --out if exists (skip already written rows)")
    ap.add_argument("--eval_metrics", action="store_true", help="compute metrics at the end")

    # Input truncation knobs
    ap.add_argument("--max_input_tokens", type=int, default=1024)
    ap.add_argument("--max_input_chars_fallback", type=int, default=4000)

    args = ap.parse_args()
    model = Model(args.provider, args.model_name)

    if args.task == "qa_mcq":
        golds, preds, recs = run_qa_mcq(model, args.dataset, args.limit, out_path=args.out, resume=args.resume)
    elif args.task == "qa_ynm":
        golds, preds, recs = run_qa_ynm(model, args.dataset, args.limit, out_path=args.out, resume=args.resume)
    elif args.task == "summ":
        golds, preds, recs = run_summ(
            model, args.dataset, args.limit,
            batch_size=args.batch_size,
            max_tokens=args.max_new_tokens,
            out_path=args.out,
            resume=args.resume,
            max_input_tokens=args.max_input_tokens,
            max_input_chars_fallback=args.max_input_chars_fallback
        )

    # Optional metrics (keep off during very large runs)
    if args.eval_metrics:
        if args.task in ["qa_mcq", "qa_ynm"]:
            acc = accuracy(golds, preds)
            macro_f1 = None
            try:
                from sklearn.metrics import f1_score
                if args.task == "qa_mcq":
                    macro_f1 = f1_score([g.upper() for g in golds], [p.upper() for p in preds],
                                        average="macro", labels=["A","B","C","D"])
                elif args.task == "qa_ynm":
                    macro_f1 = f1_score([g.upper() for g in golds], [p.upper() for p in preds],
                                        average="macro", labels=["YES","NO","MAYBE"])
            except Exception:
                pass
            print(f"=== Results [{args.task}] {args.model_name} ===")
            print(f"Accuracy : {acc:.4f}")
            if macro_f1 is not None:
                print(f"F1 (avg): {macro_f1:.4f}")

        if args.task == "summ":
            try:
                from evaluate import load
                rouge = load("rouge")
                bleu = load("bleu")
                from bert_score import score as bert_score

                preds_norm = [_norm(p) for p in preds]
                golds_norm = [_norm(g) for g in golds]

                rouge_raw = rouge.compute(predictions=preds_norm, references=golds_norm)
                def _rouge_val(v):
                    try:
                        return float(getattr(v, "mid").fmeasure)
                    except Exception:
                        return float(v)
                rouge_scores = {k: _rouge_val(v) for k, v in rouge_raw.items()}

                bleu_score = bleu.compute(predictions=preds_norm, references=golds_norm)["bleu"]

                P, R, F1 = bert_score(
                    preds_norm, golds_norm,
                    lang="en", model_type="roberta-large", rescale_with_baseline=True
                )
                print("ROUGE:", rouge_scores)
                print("BLEU:", bleu_score)
                print("BERTScore F1:", float(F1.mean()))
            except Exception as e:
                print(f"[warn] Skipped advanced metrics: {e}")

    if args.out:
        print(f"Predictions appended to {args.out}")


if __name__ == "__main__":
    # For large runs on MPS, you can also set:
    #   export HF_MPS_MAX_MEM=4GiB
    #   export TRANSFORMERS_OFFLOAD_DIR=./offload
    main()
