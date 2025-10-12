# adapters/chatdoctor_adapter.py
from pathlib import Path
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Optional 4-bit quantization (CUDA only). Safe if missing.
try:
    from transformers import BitsAndBytesConfig
    HAVE_BNB = True
except Exception:
    HAVE_BNB = False

_pipe_cache = {}

def _device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _pad_safe(tok):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def _bnb_cfg():
    # bitsandbytes is CUDA-only
    if not (HAVE_BNB and torch.cuda.is_available()):
        return None
    try:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    except Exception:
        return None

def _offload_dir():
    # Ensure accelerate has a safe place to offload if memory is tight
    p = Path(os.getenv("TRANSFORMERS_OFFLOAD_DIR", "offload"))
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def _mps_max_memory():
    """
    Conservative MPS cap to avoid huge warmup allocations.
    Override with env: HF_MPS_MAX_MEM=8GiB (or 4GiB) if needed.
    """
    return os.getenv("HF_MPS_MAX_MEM", "6GiB")

def _load(model_name: str):
    dev = _device()
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    _pad_safe(tok)

    quant = _bnb_cfg()
    offload_folder = _offload_dir()

    # Use new `dtype` kwarg
    if dev in ("cuda", "mps"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    common = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=dtype,
        # helps stability on some MPS setups; silently ignored if unsupported
        attn_implementation="eager",
    )

    device_map = "auto"
    max_memory = None
    if dev == "mps":
        max_memory = {"mps": _mps_max_memory(), "cpu": "64GiB"}

    try:
        if quant is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                max_memory=max_memory,
                quantization_config=quant,
                offload_folder=offload_folder,
                **common
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                **common
            )
    except (RuntimeError, ValueError):
        # Fallback #1: try without attn_implementation
        try:
            common2 = dict(common)
            common2.pop("attn_implementation", None)
            if quant is not None:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map,
                    max_memory=max_memory,
                    quantization_config=quant,
                    offload_folder=offload_folder,
                    **common2
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map,
                    max_memory=max_memory,
                    offload_folder=offload_folder,
                    **common2
                )
        except Exception:
            # Fallback #2: full CPU load (slow but reliable)
            common3 = dict(common)
            common3.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=None,
                **common3
            )
            # Do NOT .to("mps") here; keep CPU if memory was the issue.

    # IMPORTANT: when device_map="auto", do NOT move the model manually.
    return tok, model

def _get_pipe(model_name: str):
    if model_name in _pipe_cache:
        return _pipe_cache[model_name]

    tok, model = _load(model_name)
    # No device arg when accelerate manages placement
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        return_full_text=False,
    )
    _pipe_cache[model_name] = gen
    return gen

def generate(model_name: str, prompt: str, max_tokens: int = 256) -> str:
    pipe = _get_pipe(model_name)
    out = pipe(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=0.0,
        use_cache=True,
        pad_token_id=pipe.tokenizer.pad_token_id or pipe.tokenizer.eos_token_id,
        eos_token_id=pipe.tokenizer.eos_token_id,
    )
    text = out[0]["generated_text"]
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()

def generate_batch(model_name: str, prompts: list[str], max_tokens: int = 256) -> list[str]:
    """
    Batched generation to amortize overhead. Pass a list of string prompts.
    Returns a list of strings (one per prompt).
    """
    if not prompts:
        return []
    pipe = _get_pipe(model_name)

    outs = pipe(
        prompts,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=0.0,
        use_cache=True,
        num_return_sequences=1,  # one completion per prompt
        pad_token_id=pipe.tokenizer.pad_token_id or pipe.tokenizer.eos_token_id,
        eos_token_id=pipe.tokenizer.eos_token_id,
    )

    results = []
    # When inputs is a list, `outs` is a list (len==len(prompts)) of lists (len==num_return_sequences)
    for o, p in zip(outs, prompts):
        sample = o[0] if isinstance(o, list) else o  # handle both shapes
        text = sample.get("generated_text", "")
        if not text and "generated_token_ids" in sample:
            text = pipe.tokenizer.decode(sample["generated_token_ids"], skip_special_tokens=True)
        if text.startswith(p):
            text = text[len(p):]
        results.append(text.strip())
    return results
