"""OpenAI-compatible client for the Gemma-4-it judge.

Endpoints are populated by :func:`configure` (called from ``eval.py``).
Until configured, calling :func:`chat` raises a clear error pointing the user
at ``endpoints.json`` / ``MULTITALKBENCH_JUDGE_ENDPOINTS``. We never ship
default URLs.
"""
from __future__ import annotations

import json
import threading
from typing import Dict, List, Optional, Sequence
from urllib import error, request

import config


# ────────────────────────── runtime state ───────────────────────────────────

_endpoints: List[str] = []
_rr_lock = threading.Lock()
_rr_index = 0


def configure(*, endpoints: Sequence[str]) -> None:
    """Set the active judge endpoint pool. Idempotent.

    Pass ``config.judge_endpoints(...)`` to keep precedence rules consistent.
    """
    global _endpoints, _rr_index
    cleaned = [str(e).strip() for e in endpoints if str(e).strip()]
    if not cleaned:
        raise ValueError("configure() got an empty endpoint list")
    with _rr_lock:
        _endpoints = list(cleaned)
        _rr_index = 0


def endpoints() -> List[str]:
    return list(_endpoints)


# ────────────────────────── HTTP helpers ────────────────────────────────────

def _next_index() -> int:
    global _rr_index
    with _rr_lock:
        if not _endpoints:
            raise RuntimeError(
                "Judge endpoints not configured. Either call judge.gemma4.configure() "
                "or pass --judge-endpoints / set MULTITALKBENCH_JUDGE_ENDPOINTS / "
                "create endpoints.json (see endpoints.example.json)."
            )
        idx = _rr_index % len(_endpoints)
        _rr_index = (_rr_index + 1) % len(_endpoints)
        return idx


def _chat_url(base: str) -> str:
    return base.rstrip("/") + "/chat/completions"


def _post(base: str, payload: dict, timeout: int) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        _chat_url(base),
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.judge_api_key()}",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


# ────────────────────────── public API ──────────────────────────────────────

def chat(
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 16384,
    temperature: float = 1.0,
    timeout: int = 300,
) -> str:
    """Call the judge LLM with round-robin load balancing and per-endpoint failover.

    The response content is returned verbatim; callers handle JSON parsing.
    """
    payload = {
        "model": config.judge_model(),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 64,
    }

    n = len(_endpoints)
    start = _next_index()
    last_exc: Optional[Exception] = None
    for offset in range(n):
        base = _endpoints[(start + offset) % n]
        try:
            body = _post(base, payload, timeout=timeout)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if 400 <= exc.code < 500:
                raise RuntimeError(f"Judge HTTP {exc.code} from {base}: {detail}") from exc
            last_exc = RuntimeError(f"Judge HTTP {exc.code} from {base}: {detail}")
            continue
        except (error.URLError, TimeoutError, OSError) as exc:
            last_exc = RuntimeError(f"Judge connection failed at {base}: {exc}")
            continue

        parsed = json.loads(body)
        try:
            return parsed["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected judge payload from {base}: {parsed}") from exc

    raise RuntimeError(f"All {n} judge endpoints failed. Last error: {last_exc}")


def chat_text(prompt: str, *, max_tokens: int = 2048) -> str:
    return chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)


if __name__ == "__main__":
    # Quick connectivity check: read endpoints from endpoints.json / env, ping once.
    configure(endpoints=config.judge_endpoints())
    print(f"Configured {len(_endpoints)} endpoint(s):")
    for i, e in enumerate(_endpoints):
        print(f"  [{i}] {e}")
    print(chat_text("Reply with one word: pong.", max_tokens=8))
