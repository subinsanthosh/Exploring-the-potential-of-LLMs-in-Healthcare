# adapters/grok_fast_adapter.py
import os, time
from openai import OpenAI

# xAI uses an OpenAI-compatible API
_CLIENT = None
def _get_client():
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("XAI_API_KEY not set")
        _CLIENT = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    return _CLIENT

def generate(model_name: str, prompt: str, max_tokens: int = 8) -> str:
    """
    Minimal, deterministic, single-call path (no fallbacks).
    Keep max_tokens tiny for classification tasks to maximize speed.
    """
    client = _get_client()
    # brief, polite retry (helps transient 5xx)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model_name,                    # e.g. "grok-3-mini" or "grok-3"
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=1.0,
                n=1,
                max_tokens=max_tokens,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return ""
