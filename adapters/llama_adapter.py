# adapters/llama_adapter.py
import os
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Lazy, single global pipeline to avoid reloading the model every call
_PIPE = None
_MODEL_ID = None

def _get_pipe(model_name: str):
    """
    Build a deterministic text-generation pipeline for local LLaMA models.
    You can pass any HF model id/path compatible with causal LM inference.
    """
    global _PIPE, _MODEL_ID
    if _PIPE is not None and _MODEL_ID == model_name:
        return _PIPE

    # You can control dtype / device via env vars if needed:
    #   HF_HUB_ENABLE_HF_TRANSFER=1 (faster download)
    #   CUDA_VISIBLE_DEVICES=0
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",           # auto place on GPU/CPU
        torch_dtype="auto",          # use bf16/float16 if GPU supports
    )

    _PIPE = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # IMPORTANT: keep deterministic across runs
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    _MODEL_ID = model_name
    return _PIPE

def _strip_echo(prompt: str, text: str) -> str:
    # Some local models echo the prompt; remove it if present.
    if text.startswith(prompt):
        return text[len(prompt):].lstrip()
    return text

def generate(model_name: str, prompt: str, max_tokens: int = 256) -> str:
    """
    Generate with a local LLaMA(-style) model using transformers.
    Pass, for example:  model_name="meta-llama/Llama-3-8b-instruct"
    """
    pipe = _get_pipe(model_name)
    out = pipe(
        prompt,
        max_new_tokens=max_tokens,
        return_full_text=True,  # easier to remove echo
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )[0]["generated_text"]
    return _strip_echo(prompt, out).strip()
