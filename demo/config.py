"""Shared configuration — LLM and embedding client setup.

All providers use OpenAI-compatible APIs. Configure via environment variables:

  LLM_API_KEY      – API key for the chat completions provider
  LLM_BASE_URL     – Base URL (e.g. https://api.groq.com/openai/v1)
  LLM_MODEL        – Model name (e.g. openai/gpt-oss-120b, llama-3.3-70b-versatile)

  EMBED_API_KEY    – API key for the embeddings provider
  EMBED_BASE_URL   – Base URL (e.g. https://api.fireworks.ai/inference/v1)
  EMBED_MODEL      – Embedding model name (e.g. nomic-ai/nomic-embed-text-v1.5)
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from project root (one level up from demo/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── LLM (chat completions) ─────────────────────────────────────
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_BASE_URL = os.environ["LLM_BASE_URL"]
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")

# ── Embeddings ──────────────────────────────────────────────────
EMBED_API_KEY = os.environ["EMBED_API_KEY"]
EMBED_BASE_URL = os.environ["EMBED_BASE_URL"]
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")


def get_llm_client() -> OpenAI:
    """Return an OpenAI-compatible client for chat completions."""
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def get_embed_client() -> OpenAI:
    """Return an OpenAI-compatible client for embeddings."""
    return OpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY)
