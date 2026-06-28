"""Thin OpenRouter chat client.

Both the agent and the proposer call the *same* fixed model, matching the
paper's setup where M acts as agent and proposer under harness h_t.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Minimal OpenRouter chat client (stdlib only -- Colab-friendly)."""

    def __init__(self, model: str, api_key: str, retries: int = 4):
        self.model = model
        self.api_key = api_key
        self.retries = retries

    def chat(self, messages, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        if not self.api_key:
            raise LLMError("OpenRouter API key not set")

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode()

        last_err = None
        for attempt in range(self.retries):
            req = urllib.request.Request(
                OPENROUTER_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    body = json.loads(resp.read().decode())
                return body["choices"][0]["message"]["content"]
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                KeyError,
                TimeoutError,
            ) as e:
                last_err = e
                detail = ""
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        detail = e.read().decode()[:500]
                    except Exception:
                        detail = ""
                wait = 2 ** attempt
                print(
                    f"[llm] attempt {attempt+1} failed: {e} {detail}; retry in {wait}s",
                    flush=True,
                )
                time.sleep(wait)
        raise LLMError(f"chat failed after {self.retries} attempts: {last_err}")
