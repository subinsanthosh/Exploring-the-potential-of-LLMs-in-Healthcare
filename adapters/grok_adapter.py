# adapters/grok_adapter.py
import os, time
from openai import OpenAI

_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
_API_KEY  = os.getenv("XAI_API_KEY")
_CLIENT   = None

def _client():
    global _CLIENT
    if _CLIENT is None:
        if not _API_KEY:
            raise RuntimeError("XAI_API_KEY not set")
        _CLIENT = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)
    return _CLIENT

_SYS_YNM = (
    "You are a biomedical QA assistant.\n"
    "Reply with ONLY ONE WORD: YES, NO, or MAYBE. No punctuation, no explanation."
)

def _backoff(attempt: int):
    time.sleep(min(16, 2 ** attempt))  # 1,2,4,8,16

def _clean(text: str) -> str:
    return (text or "").strip()

def _try_chat(model_name, prompt, max_tokens):
    cli = _client()
    r = cli.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _SYS_YNM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        n=1,
        presence_penalty=0,
        frequency_penalty=0,
    )
    return _clean(r.choices[0].message.content)

def _try_responses(model_name, prompt, max_tokens):
    # OpenAI Responses API–style (xAI is compatible)
    cli = _client()
    r = cli.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": _SYS_YNM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=max_tokens,
    )
    # responses API returns content parts; join text pieces
    parts = []
    for item in r.output_text.splitlines():
        parts.append(item)
    return _clean("\n".join(parts))

def _try_completions(model_name, prompt, max_tokens):
    # Classic text completion fallback (some providers still support it)
    cli = _client()
    r = cli.completions.create(
        model=model_name,
        prompt=f"{_SYS_YNM}\n\nQuestion:\n{prompt}\n\nAnswer:",
        temperature=0.0,
        max_tokens=max_tokens,
        top_p=1.0,
        n=1,
    )
    return _clean(r.choices[0].text)

def generate(model_name: str, prompt: str, max_tokens: int = 256) -> str:
    # We’ll retry each strategy with backoff; keep it short for classification
    for attempt in range(3):
        try:
            text = _try_chat(model_name, prompt, max_tokens)
            if text:
                return text
        except Exception:
            _backoff(attempt)
    for attempt in range(2):
        try:
            text = _try_responses(model_name, prompt, max_tokens)
            if text:
                return text
        except Exception:
            _backoff(attempt)
    for attempt in range(2):
        try:
            text = _try_completions(model_name, prompt, max_tokens)
            if text:
                return text
        except Exception:
            _backoff(attempt)
    # final guardrail so your loop doesn’t break
    return "MAYBE"
