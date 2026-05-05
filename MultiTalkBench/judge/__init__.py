"""Judge LLM clients used by MultiTalkBench.

The benchmark calls a Gemma-4-it judge over an OpenAI-compatible endpoint
(typically self-hosted via vLLM). Endpoints are configured via
:func:`gemma4.configure` (called once from ``eval.py`` based on
``endpoints.json`` / env / CLI). See :mod:`config` for resolution rules.
"""

from . import gemma4  # noqa: F401
