import os
import requests

def ollama_generate(prompt: str, model: str, base_url: str) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096
        }
    }
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def llm_complete(prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "phi3:mini")
        base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        return ollama_generate(prompt, model=model, base_url=base_url)
    raise ValueError(f"Unsupported LLM_PROVIDER={provider}")
