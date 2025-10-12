import os
import google.generativeai as genai
_models={}

def _get_model(name):
    if name not in _models:
        key=os.getenv('GOOGLE_API_KEY')
        if not key: raise RuntimeError('GOOGLE_API_KEY not set')
        genai.configure(api_key=key)
        _models[name]=genai.GenerativeModel(name)
    return _models[name]

def generate(model_name, prompt, max_tokens=256):
    m=_get_model(model_name)
    resp=m.generate_content(prompt, generation_config={"temperature":0.0, "max_output_tokens": max_tokens})
    return (getattr(resp,'text','') or '').strip()
