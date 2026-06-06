"""
llm_client.py
--------------
Thin wrapper around the Ollama HTTP API.
Handles retries, timeouts, and Apple Silicon optimization hints.

Supported models (auto-detected based on what's installed):
  - mistral         (7B, excellent quality)
  - llama3.2        (3B, faster, great for Apple Silicon)
  - phi3            (3.8B, Microsoft, very efficient)
  - gemma2:2b       (2B, fastest option)
  - qwen2.5:3b      (good multilingual support)
"""

import json
import time
import urllib.request
import urllib.error
from typing import Optional

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"          # overridden by auto-detect
REQUEST_TIMEOUT = 120              # seconds
MAX_RETRIES = 2


def query_llm(
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    model: Optional[str] = None,
) -> Optional[str]:
    """
    Send a prompt to the local Ollama LLM and return the response text.
    Returns None on failure.
    """
    model = model or get_best_available_model()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 4096,
        },
    }
    if system:
        payload["system"] = system

    for attempt in range(MAX_RETRIES + 1):
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()

        except urllib.error.URLError as e:
            if attempt < MAX_RETRIES:
                time.sleep(2)
                continue
            print(f"\n❌ Ollama connection failed: {e}")
            print("   Make sure Ollama is running: ollama serve")
            return None
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            print(f"\n❌ LLM error: {e}")
            return None

    return None


def get_best_available_model() -> str:
    """
    Query Ollama for installed models and pick the best one for study tasks.
    Returns a fallback model name if Ollama is unreachable.
    """
    preference_order = [
        "mistral",
        "llama3.2",
        "llama3",
        "phi3",
        "phi3.5",
        "gemma2",
        "qwen2.5",
        "llama2",
        "orca-mini",
    ]
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/tags",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            installed = [m["name"].split(":")[0] for m in data.get("models", [])]

        for preferred in preference_order:
            if preferred in installed:
                return preferred

        # Return first available model
        if installed:
            return data["models"][0]["name"]

    except Exception:
        pass

    return DEFAULT_MODEL


def check_ollama_running() -> tuple[bool, str]:
    """
    Check if Ollama is running and return (ok, model_name).
    """
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("models", [])
            if not models:
                return False, ""
            model = get_best_available_model()
            return True, model
    except Exception:
        return False, ""


def list_installed_models() -> list:
    """Return list of installed model names."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
