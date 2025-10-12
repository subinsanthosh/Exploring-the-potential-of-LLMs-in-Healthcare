# adapters/openrouter_adapter.py
# Full, drop-in OpenRouter adapter (requests-only).
# Works with models like "x-ai/grok-3-mini".
# - Robust JSON handling & retries
# - Handles string, multipart, and "reasoning" outputs
# - Optional <final>...</final> enforcement to force clean answers
# - No venv required; just: pip install requests

import os
import re
import json
import time
import requests
from typing import List, Optional

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# --------- Headers / HTTP --------- #
def _headers():
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Optional but recommended by OpenRouter for attribution/rate limits
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "llm_eval"),
        "User-Agent": "llm-eval/1.0",
    }

def _post(payload, retries: int = 4, timeout: int = 90):
    last = None
    for i in range(retries):
        r = requests.post(
            CHAT_URL,
            headers=_headers(),
            data=json.dumps(payload),
            timeout=timeout,
        )
        # Retry on rate limits / transient server errors
        if r.status_code in (429, 529) or 500 <= r.status_code < 600:
            time.sleep(min(2 ** i, 16))
            last = r
            continue
        if not (200 <= r.status_code < 300):
            raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:400]}")
        try:
            return r.json()
        except Exception:
            raise RuntimeError(f"Non-JSON: {r.text[:200]}")
    raise RuntimeError(f"OpenRouter retry fail: {last.status_code if last else 'no resp'}")

# --------- Text extraction --------- #
def _gather_text(obj):
    """Recursively collect plausible text fields from any OpenRouter response."""
    out = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k in ("content", "text", "output_text", "output", "reasoning"):
            v = obj.get(k)
            if isinstance(v, (str, list, dict)):
                out.extend(_gather_text(v))
        if "message" in obj:
            out.extend(_gather_text(obj["message"]))
        if "choices" in obj:
            out.extend(_gather_text(obj["choices"]))
        if "data" in obj:
            out.extend(_gather_text(obj["data"]))
    elif isinstance(obj, list):
        for x in obj:
            out.extend(_gather_text(x))
    return out

def _extract(resp) -> str:
    """Prefer message.content; fall back to parts; then reasoning; then any text we can salvage."""
    try:
        choice0 = (resp.get("choices") or [])[0]
        msg = choice0.get("message", {})
        # 1) plain string content
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
        # 2) list of parts
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, dict):
                    t = c.get("text") or c.get("content") or ""
                    if isinstance(t, str):
                        parts.append(t)
            txt = " ".join(p.strip() for p in parts if p).strip()
            if txt:
                return txt
        # 3) some models (e.g., Grok) may put text into "reasoning"
        reason = msg.get("reasoning")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
    except Exception:
        pass
    # 4) last resort: gather any text fragments anywhere
    texts = [t.strip() for t in _gather_text(resp) if isinstance(t, str) and t.strip()]
    seen, flat = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            flat.append(t)
    return " ".join(flat).strip()

# --------- <final> tag helpers (optional) --------- #
_TAG_RE = re.compile(r"<final>(.*?)</final>", re.DOTALL)

def _wrap_with_final_tags(text: str) -> str:
    return (
        "Return the final answer ONLY inside <final>...</final> tags.\n"
        "No analysis or reasoning outside the tags.\n\n"
        f"USER TASK:\n{text}\n\n"
        "Output strictly:\n<final>...</final>\n"
    )

def _extract_final_tag(text: str) -> str:
    m = _TAG_RE.search(text or "")
    return (m.group(1).strip() if m else "").strip()

# --------- Public API --------- #
def generate(
    model_name: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    system: Optional[str] = None,
    enforce_final_tag: bool = False,
) -> str:
    """
    Call OpenRouter chat completions and return plain text.
    - Works with: model_name="x-ai/grok-3-mini"
    - Grok models do NOT support 'stop' sequences; avoid passing them.
    - If enforce_final_tag=True, the prompt is wrapped so the answer is inside <final>...</final>,
      and we post-process to return only that inner text.
    """
    if enforce_final_tag:
        prompt = _wrap_with_final_tags(prompt)

    # Keep it simple: no system message by default (Grok sometimes reacts to it verbosely)
    messages = [{"role": "user", "content": prompt}] if system is None else [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        # Ask Grok to minimize explicit reasoning budget
        "reasoning": {"effort": "low"},
    }

    resp = _post(payload)
    txt = _extract(resp)
    if enforce_final_tag:
        inner = _extract_final_tag(txt)
        if inner:
            txt = inner
    if not txt:
        raise RuntimeError(f"Empty content: {json.dumps(resp)[:400]}")
    return txt

def generate_batch(
    model_name: str,
    prompts: List[str],
    max_tokens: int = 128,
    temperature: float = 0.0,
    system: Optional[str] = None,
    enforce_final_tag: bool = False,
) -> List[str]:
    """Sequential batching for reliability with provider limits."""
    outs = []
    for p in prompts:
        outs.append(
            generate(
                model_name=model_name,
                prompt=p,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                enforce_final_tag=enforce_final_tag,
            )
        )
    return outs
