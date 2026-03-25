<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OpenSymbolicAI/.github/main/profile/opensymbolicai-horizontal-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/OpenSymbolicAI/.github/main/profile/opensymbolicai-horizontal.svg">
    <img alt="OpenSymbolicAI" src="https://raw.githubusercontent.com/OpenSymbolicAI/.github/main/profile/opensymbolicai-horizontal.svg" height="48">
  </picture>
</p>

# SEC 10-K Agent Comparison: Behaviour Programming vs ReAct

A side-by-side comparison of two agentic architectures for answering financial questions over SEC 10-K filings:

- **ReAct** — The LLM decides which tools to call step-by-step in a loop (classic tool-calling agent).
- **Behaviour Programming** — Decompositions teach the LLM *how* to compose primitives. The plan is generated upfront, then executed deterministically.

Both agents share the same retrieval tools (LanceDB vector search over 10 companies' 10-K filings) and the same LLM backend. The demo runs identical queries through both and compares LLM calls, token usage, and latency.

Any **OpenAI-compatible** API works — Groq, OpenAI, Together, Ollama, etc.

## Companies Covered

Apple, Microsoft, Alphabet (Google), Amazon, Tesla, Meta, NVIDIA, Netflix, Salesforce, Adobe.

## Setup (step by step)

### Step 1: Install dependencies

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cd demo
uv sync
```

### Step 2: Configure API keys

Copy the environment template:

```bash
cp ../.env.example ../.env
```

Edit `.env` with your provider details. You need two things:

1. **An LLM provider** (chat completions with tool-calling support)
2. **An embedding provider** (for ingestion and retrieval)

#### Option A: Groq + Fireworks (default, fastest)

```env
LLM_API_KEY=gsk_your_groq_key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b

EMBED_API_KEY=fw_your_fireworks_key
EMBED_BASE_URL=https://api.fireworks.ai/inference/v1
EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5
```

Get keys at [console.groq.com](https://console.groq.com) and [fireworks.ai](https://fireworks.ai).

#### Option B: OpenAI for everything

```env
LLM_API_KEY=sk-your_openai_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o

EMBED_API_KEY=sk-your_openai_key
EMBED_BASE_URL=https://api.openai.com/v1
EMBED_MODEL=text-embedding-3-small
```

#### Option C: Ollama (fully local, no API keys needed)

```bash
# Pull models first
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

```env
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=gpt-oss:20b

EMBED_API_KEY=ollama
EMBED_BASE_URL=http://localhost:11434/v1
EMBED_MODEL=nomic-embed-text
```

#### Option D: Mix and match

Any combination works — the LLM and embedding providers are independent. Use whatever OpenAI-compatible endpoint you prefer for each.

### Step 3: Ingest SEC filings

Download filings from EDGAR, chunk, embed, and store in LanceDB:

```bash
uv run python scripts/ingest.py
```

This creates the vector index at `data/lancedb/`. You only need to run this once. If you change your embedding model, re-run this step.

### Step 4: Run the demo

```bash
# Run the 3 showcase queries (recommended for live demo)
uv run python main.py --demo

# Run all 15 queries
uv run python main.py

# Run specific queries by number (1-15)
uv run python main.py -q 1 8 12

# Side-by-side TUI (both agents running concurrently)
uv run python main.py --side-by-side --demo
```

### Showcase Queries (`--demo`)

The `--demo` flag runs 3 hand-picked queries that best illustrate the differences:

| # | Type | Query | Why it's interesting |
|---|------|-------|---------------------|
| 1 | Simple Extraction | "What was Microsoft's operating income in its most recent fiscal year?" | Both agents get it right. BP uses ~5x fewer tokens. |
| 8 | Cross-Company Comparison | "Compare the R&D spending of Meta and Apple in their most recent fiscal year." | Both correct. BP uses **35x fewer tokens**. |
| 12 | Aggregation Across All 10 | "Which of the 10 companies reported the highest total revenue?" | ReAct **fails** (hits iteration limit). BP succeeds. 14x token gap. |

### All 15 Queries

The full query set lives in `demo/queries.json` with ground truth derived from the actual filing data in LanceDB. There are 5 queries per category:

- **Simple Extraction** (1-5) — Single-company, single-metric lookups
- **Cross-Company Comparison** (6-10) — 2-3 company comparisons
- **Aggregation Across All 10** (11-15) — Questions spanning all 10 companies

## Benchmark Results

Results from running all 15 queries with GPT-OSS 120B on Groq (same LLM, same retrieval index):

### Efficiency

| Metric | ReAct | Behaviour Programming | Ratio |
|--------|-------|----------------------|-------|
| **Success Rate** | 12/15 (80%) | **15/15 (100%)** | — |
| **Total LLM Calls** | 221 | **64** | 3.5x fewer |
| **Total Tokens** | 3,245,043 | **271,499** | **12x fewer** |
| **Total Latency** | 494.7s | **68.7s** | **7.2x faster** |

#### By Category

| Category | Queries | Avg ReAct Tokens | Avg BP Tokens | Token Ratio | ReAct Failures |
|----------|---------|-----------------|--------------|-------------|---------------|
| Simple Extraction | 1-5 | 19,307 | 5,801 | 3.3x | 0 |
| Cross-Company | 6-10 | 87,969 | 10,280 | 8.6x | 0 |
| Aggregation | 11-15 | 541,733 | 29,219 | 18.5x | 3 |

### Quality (LLM-as-Judge, 1-5 scale)

| Dimension | ReAct | Behaviour Programming |
|-----------|-------|----------------------|
| Correctness | 4.0 | **4.2** |
| Completeness | **3.8** | 3.4 |
| Relevance | **4.7** | 4.5 |
| Faithfulness | 4.1 | **4.7** |
| Conciseness | 4.2 | **4.7** |
| **Overall** | 4.15 | **4.32** |

**Key findings:**

- BP succeeded on **all 15 queries**; ReAct failed on 3 (hit iteration limit on complex aggregations).
- BP used **12x fewer tokens** and was **7.2x faster** overall.
- Answer quality is comparable — BP scores slightly higher overall (4.32 vs 4.15).
- BP has notably higher **faithfulness** (4.7 vs 4.1), hallucinating less due to its deterministic execution plan.
- The efficiency gap widens with complexity: 3.3x for simple lookups, **18.5x for aggregations across all 10 companies**.
- Both agents struggle on the hardest aggregation queries (Q14-15), but BP at least produces answers while ReAct fails outright.

## Project Structure

```
demo/
├── main.py              # Entry point
├── cli.py               # CLI harness — runs queries through both agents
├── tui.py               # Side-by-side Textual TUI
├── queries.json         # 15 demo queries with ground truth
├── react_agent.py       # ReAct agent (OpenAI-compatible tool calling)
├── behavior_agent.py    # Behaviour Programming agent (opensymbolicai)
├── evaluate.py          # LLM-as-judge evaluation
├── retriever.py         # LanceDB vector search
├── sec_loader.py        # SEC EDGAR downloader and chunker
├── models.py            # Shared Pydantic models
├── config.py            # API keys and client setup
├── pyproject.toml       # Dependencies
├── scripts/
│   └── ingest.py        # One-time ingestion pipeline
└── data/
    ├── filings/         # Cached 10-K text (gitignored)
    └── lancedb/         # Vector index (gitignored)
```

## License

MIT
