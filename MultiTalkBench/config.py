"""Centralised configuration for the MultiTalkBench evaluation pipeline.

All paths and endpoints are resolved here so that no other module hard-codes
absolute paths or internal URLs. Resolution order: explicit argument > env var
> sensible relative default.

Environment variables
---------------------
MULTITALKBENCH_DATA
    Path to a local clone of the `MultiTalk/MultiTalkBench` HuggingFace dataset
    (the directory containing `metadata.jsonl` and `audio/`, `alignments/`,
    `personas/`, `meetings/`).

MULTITALKBENCH_RESULTS
    Output root for benchmark runs. Default: `./results`.

MULTITALKBENCH_JUDGE_ENDPOINTS
    Comma-separated list of OpenAI-compatible base URLs serving the judge LLM
    (e.g. a vLLM Gemma-4-it deployment). Used as fallback if `endpoints.json`
    is absent.

MULTITALKBENCH_JUDGE_MODEL
    Judge model name passed in OpenAI requests. Default: `google/gemma-4-31B-it`.

MULTITALKBENCH_JUDGE_API_KEY
    Bearer token for the judge endpoint. Default: `EMPTY` (vLLM accepts this).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_ENDPOINTS_FILE = REPO_ROOT / "endpoints.json"


def data_root(explicit: Optional[Path] = None) -> Path:
    """Resolve the dataset root. Raises a clear error if the user forgot to set it."""
    if explicit is not None:
        path = Path(explicit)
    else:
        env = os.environ.get("MULTITALKBENCH_DATA")
        if not env:
            raise RuntimeError(
                "MultiTalkBench dataset root not set. Either pass --data-root or "
                "export MULTITALKBENCH_DATA. Download with:\n"
                "  huggingface-cli download MultiTalk/MultiTalkBench "
                "--repo-type dataset --local-dir ./data"
            )
        path = Path(env)
    metadata = path / "metadata.jsonl"
    if not metadata.exists():
        raise FileNotFoundError(
            f"{metadata} not found. Is {path} a clone of the MultiTalk/MultiTalkBench "
            "dataset?"
        )
    return path.resolve()


def results_root(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    env = os.environ.get("MULTITALKBENCH_RESULTS")
    return Path(env).resolve() if env else DEFAULT_RESULTS_ROOT


def judge_endpoints(
    explicit: Optional[List[str]] = None,
    *,
    endpoints_file: Optional[Path] = None,
) -> List[str]:
    """Resolve a list of OpenAI-compatible judge base URLs.

    Priority:
      1. ``explicit`` argument (e.g. from ``--judge-endpoints``).
      2. ``endpoints_file`` (defaults to ``./endpoints.json``).
      3. ``MULTITALKBENCH_JUDGE_ENDPOINTS`` env var (comma-separated).
    """
    if explicit:
        return [u.strip() for u in explicit if u.strip()]

    path = endpoints_file or DEFAULT_ENDPOINTS_FILE
    if path.exists():
        data = json.loads(path.read_text())
        urls = data["endpoints"] if isinstance(data, dict) else data
        return [str(u).strip() for u in urls if str(u).strip()]

    env = os.environ.get("MULTITALKBENCH_JUDGE_ENDPOINTS", "")
    if env:
        return [u.strip() for u in env.split(",") if u.strip()]

    raise RuntimeError(
        "No judge LLM endpoints configured. Either:\n"
        "  - create endpoints.json (see endpoints.example.json),\n"
        "  - export MULTITALKBENCH_JUDGE_ENDPOINTS=url1,url2,..., or\n"
        "  - pass --judge-endpoints url1,url2,..."
    )


def judge_model() -> str:
    return os.environ.get("MULTITALKBENCH_JUDGE_MODEL", "google/gemma-4-31B-it")


def judge_api_key() -> str:
    return os.environ.get("MULTITALKBENCH_JUDGE_API_KEY", "EMPTY")
