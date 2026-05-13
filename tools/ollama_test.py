"""Quick smoke test for local Ollama endpoint."""
from backend.ollama import generate

import os

DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b")


if __name__ == "__main__":
    try:
        resp = generate(DEFAULT_URL, DEFAULT_MODEL, "Why is the sky blue?")
        print("Response:\n", resp)
    except Exception as e:
        print("Ollama test failed:", e)
