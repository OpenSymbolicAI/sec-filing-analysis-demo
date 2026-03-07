"""ReAct agent using raw OpenAI-compatible tool-calling (via Groq) for SEC filing analysis.

Same tools as the behavior agent, but the LLM decides how to use them
step-by-step in a tool-calling loop.
"""

from __future__ import annotations

import json
import re
import time

from config import MODEL, get_llm_client
from models import AgentResult
from retriever import Retriever

# ── Tool definitions (OpenAI function-calling format) ───────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Search SEC 10-K filings for chunks relevant to a query. "
                "Returns a list of document chunks with company, section, and text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "k": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_for_company",
            "description": (
                "Search SEC 10-K filings filtered to a specific company. "
                "Returns a list of document chunks with company, section, and text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "company": {"type": "string", "description": "Company name to filter by"},
                    "k": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
                },
                "required": ["query", "company"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_number",
            "description": (
                "Extract a specific financial metric value from text. "
                "Returns the number with unit or 'NOT_FOUND'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to extract from"},
                    "metric": {"type": "string", "description": "The metric to extract"},
                },
                "required": ["text", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "combine_texts",
            "description": "Combine text from a list of retrieved documents into a single string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "docs": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of retrieved document dicts",
                    },
                },
                "required": ["docs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_from_context",
            "description": "Answer a question given a context string. Be precise with numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "The context text"},
                    "question": {"type": "string", "description": "The question to answer"},
                },
                "required": ["context", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_ratio",
            "description": "Compute a ratio from two number strings (numerator / denominator).",
            "parameters": {
                "type": "object",
                "properties": {
                    "numerator": {"type": "string", "description": "The numerator value"},
                    "denominator": {"type": "string", "description": "The denominator value"},
                },
                "required": ["numerator", "denominator"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_count",
            "description": (
                "Extract a non-monetary count (e.g. employee headcount) from text. "
                "Unlike extract_number, this does NOT normalise to millions. "
                "Returns the count as a plain integer string, or 'NOT_FOUND'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to extract from"},
                    "metric": {"type": "string", "description": "The non-monetary metric to extract"},
                },
                "required": ["text", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_metrics",
            "description": (
                "Extract multiple financial metrics from text in a single call. "
                "Returns a JSON object mapping each metric name to its value with unit, "
                "or 'NOT_FOUND'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to extract from"},
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of metric names to extract",
                    },
                },
                "required": ["text", "metrics"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are FinanceBot, an expert financial analyst assistant specialising in SEC 10-K \
annual filings. You have access to an index of SEC 10-K filings covering fiscal years \
2022-2025 as available on SEC EDGAR. The user's query will specify which companies to \
analyse — do not assume a fixed set of companies.

## Instructions

1. **Think step by step.** Before calling any tool, reason about what information you \
need and which tools will provide it. Plan your retrieval strategy before executing.

2. **Retrieval strategy:**
   - For single-company questions, use `retrieve_for_company` with the company name \
     and a precise query targeting the relevant financial line item or section.
   - For cross-company comparisons, retrieve data for EACH company separately. Do NOT \
     attempt to retrieve all companies in a single query — this reduces relevance.
   - **CRITICAL for aggregation queries** (e.g., "which company has the highest X", \
     "rank these companies by Y"): You MUST retrieve AND extract data for EVERY company \
     mentioned in the query. Maintain a mental checklist of all companies. After each \
     retrieve+extract step, note which companies you have covered and which remain. \
     Do NOT produce a final answer until EVERY company on the list has been processed. \
     If the query lists 10 companies, you need 10 retrieve calls and 10 extract calls.
   - Use `k=5` for most queries. Increase to `k=8` only when the answer spans multiple \
     sections (e.g., revenue broken down by segment).

3. **Data extraction:**
   - After retrieving documents, use `extract_number` to pull specific metrics. \
     Do NOT attempt to read numbers directly from retrieved text — always use the \
     extraction tool for precision.
   - When computing ratios or derived metrics, use `compute_ratio`. Do NOT perform \
     arithmetic yourself — your calculations may be imprecise. Always delegate math \
     to the compute tool.
   - If `extract_number` returns "NOT_FOUND", try broadening your search query or \
     retrieving from a different section before concluding the data is unavailable.

4. **Output formatting:**
   - Always report financial figures with their units (e.g., "$394.3 billion", \
     "€12.7 million"). Never report raw numbers without context.
   - Always cite the company name and fiscal year when presenting a figure.
   - For comparisons, present results in a structured format: list each company with \
     its value, then state the conclusion.
   - Round ratios to 2 decimal places. Round currency to 1 decimal place in billions, \
     0 decimal places in millions.

5. **Edge cases and error handling:**
   - If a company's data is not found for a particular metric, explicitly state \
     "Data not available for [Company] for [metric]" rather than guessing or omitting.
   - If the question asks about a time period outside the available filings, say so.
   - If the question is ambiguous (e.g., "revenue" could mean net revenue, gross revenue, \
     or total net sales), default to total net revenue / net sales as reported on the \
     consolidated statements of operations, and note your assumption.
   - For questions about "Google", treat this as Alphabet Inc. and search accordingly. \
     Similarly, use official company names when searching (e.g., "Meta" → "Meta Platforms").

6. **Prohibited behaviours:**
   - Do NOT hallucinate or fabricate financial figures. Every number must come from a \
     retrieved document via the extraction tool.
   - Do NOT provide investment advice, forward-looking predictions, or opinions on \
     stock performance.
   - Do NOT summarise entire filings unless explicitly asked. Focus on answering the \
     specific question asked.
   - Do NOT skip companies in aggregation queries. If the user lists specific \
     companies, you must check every single one.
   - Do NOT re-derive numbers that you have already extracted. Store and reuse results \
     across your reasoning steps.

7. **Response structure:**
   - Lead with a direct answer to the question.
   - Follow with supporting data and citations.
   - If the answer required comparison or aggregation, include a brief summary table \
     or ranked list.
   - End with any caveats about data availability or assumptions made.\
"""


class ReactAgent:
    """ReAct-style agent using Groq's tool-calling API in a loop."""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        self._client = get_llm_client()
        self._llm_calls = 0
        self._total_tokens = 0

    def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool call and return the result as a string."""
        if name == "retrieve":
            docs = self._retriever.search(args["query"], k=args.get("k", 5))
            result = [
                {
                    "company": d.chunk.company,
                    "ticker": d.chunk.ticker,
                    "fiscal_year": d.chunk.fiscal_year,
                    "section": d.chunk.section,
                    "text": d.chunk.text,
                    "score": round(d.score, 3),
                }
                for d in docs
            ]
            return json.dumps(result)

        elif name == "retrieve_for_company":
            docs = self._retriever.search(
                args["query"], k=args.get("k", 5), company=args.get("company")
            )
            result = [
                {
                    "company": d.chunk.company,
                    "ticker": d.chunk.ticker,
                    "fiscal_year": d.chunk.fiscal_year,
                    "section": d.chunk.section,
                    "text": d.chunk.text,
                    "score": round(d.score, 3),
                }
                for d in docs
            ]
            return json.dumps(result)

        elif name == "extract_number":
            resp = self._client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract the exact value for the requested metric from the text. "
                            "IMPORTANT rules:\n"
                            "1. Pay attention to the unit scale stated in the filing "
                            "(e.g. 'in thousands', 'in millions', 'in billions'). "
                            "Convert the value to MILLIONS of dollars.\n"
                            "2. Always prefer the TOTAL / CONSOLIDATED figure over any "
                            "segment or partial figure. For revenue, look for 'Total net sales', "
                            "'Total revenue', or 'Total revenues' — never a segment subtotal.\n"
                            "3. If current and prior year figures are both present, extract the "
                            "MOST RECENT year's figure.\n"
                            "Return ONLY the number followed by ' million' "
                            "(e.g. '$57,372 million', '$416,161 million'). "
                            "If not found, return exactly 'NOT_FOUND'."
                        ),
                    },
                    {"role": "user", "content": f"Metric: {args['metric']}\n\nText:\n{args['text']}"},
                ],
                temperature=0,
                max_tokens=500,
            )
            self._llm_calls += 1
            if resp.usage:
                self._total_tokens += resp.usage.total_tokens
            return resp.choices[0].message.content or "NOT_FOUND"

        elif name == "combine_texts":
            docs = args.get("docs", [])
            return "\n\n---\n\n".join(
                f"[{d.get('company', '?')} | {d.get('section', '?')}]\n{d.get('text', '')}"
                for d in docs
            )

        elif name == "answer_from_context":
            resp = self._client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer the question using ONLY the provided context. "
                            "Be precise with numbers and cite the company name. "
                            "IMPORTANT: Pay close attention to the unit scale of each filing "
                            "(some report in thousands, others in millions or billions). "
                            "Normalise all figures to millions of dollars before comparing. "
                            "If you cannot find the answer, provide your best attempt with "
                            "whatever data IS available rather than saying the data is missing."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{args['context']}\n\nQuestion: {args['question']}",
                    },
                ],
                temperature=0,
                max_tokens=300,
            )
            self._llm_calls += 1
            if resp.usage:
                self._total_tokens += resp.usage.total_tokens
            return resp.choices[0].message.content or ""

        elif name == "compute_ratio":
            def _parse(s: str) -> float | None:
                s = s.replace(",", "").replace("$", "").strip()
                match = re.search(r"[\d.]+", s)
                return float(match.group()) if match else None

            n = _parse(args["numerator"])
            d = _parse(args["denominator"])
            if n is None or d is None or d == 0:
                return "UNABLE_TO_COMPUTE"
            return f"{n / d:.2f}"

        elif name == "extract_count":
            resp = self._client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract the exact count for the requested metric from the text. "
                            "This is a NON-MONETARY value (e.g. number of employees). "
                            "Return ONLY the integer number (e.g. '16000', '228000'). "
                            "If the text says 'approximately 16,000' return '16000'. "
                            "If not found, return exactly 'NOT_FOUND'."
                        ),
                    },
                    {"role": "user", "content": f"Metric: {args['metric']}\n\nText:\n{args['text']}"},
                ],
                temperature=0,
                max_tokens=100,
            )
            self._llm_calls += 1
            if resp.usage:
                self._total_tokens += resp.usage.total_tokens
            return resp.choices[0].message.content or "NOT_FOUND"

        elif name == "extract_metrics":
            metrics = args.get("metrics", [])
            metrics_list = "\n".join(f"- {m}" for m in metrics)
            resp = self._client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract the exact values for each requested metric from the text. "
                            "IMPORTANT: Pay attention to the unit scale stated in the filing "
                            "(e.g. 'in thousands', 'in millions', 'in billions'). "
                            "Convert all values to MILLIONS of dollars. "
                            "Return a JSON object mapping each metric name to its value in millions "
                            '(e.g. {"total debt": "$85,750 million", "total equity": "$73,733 million"}). '
                            "Use 'NOT_FOUND' for any metric not present. Return ONLY valid JSON."
                        ),
                    },
                    {"role": "user", "content": f"Metrics:\n{metrics_list}\n\nText:\n{args['text']}"},
                ],
                temperature=0,
                max_tokens=500,
            )
            self._llm_calls += 1
            if resp.usage:
                self._total_tokens += resp.usage.total_tokens
            return resp.choices[0].message.content or "{}"

        return f"Unknown tool: {name}"

    def run(self, query: str, max_iterations: int = 40, timeout: float = 120) -> AgentResult:
        """Run the ReAct loop until the LLM produces a final answer."""
        start = time.time()
        self._llm_calls = 0
        self._total_tokens = 0
        consecutive_errors = 0

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        for _ in range(max_iterations):
            if time.time() - start > timeout:
                return AgentResult(
                    agent_name="ReAct (Groq Tool Calling)",
                    query=query,
                    answer="[TIMEOUT]",
                    plan="(step-by-step tool calling — no upfront plan)",
                    llm_calls=self._llm_calls,
                    total_tokens=self._total_tokens,
                    latency_seconds=round(time.time() - start, 2),
                    success=False,
                    failure_reason=f"Exceeded {timeout}s timeout",
                )
            try:
                resp = self._client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0,
                )
                consecutive_errors = 0
            except Exception as e:
                self._llm_calls += 1
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    return AgentResult(
                        agent_name="ReAct (Groq Tool Calling)",
                        query=query,
                        answer="[CRASHED]",
                        plan="(step-by-step tool calling — no upfront plan)",
                        llm_calls=self._llm_calls,
                        total_tokens=self._total_tokens,
                        latency_seconds=round(time.time() - start, 2),
                        success=False,
                        failure_reason=f"3 consecutive errors, last: {type(e).__name__}: {e}",
                    )
                # Groq sometimes fails to parse model output into a tool call
                # (e.g. LLM hallucinates a tool name not in the schema).
                # Inject a retry hint listing the valid tool names so the model
                # can self-correct.
                valid_names = [t["function"]["name"] for t in TOOLS]
                messages.append(
                    {"role": "assistant", "content": "[internal error — invalid tool call]"}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous tool call failed: {e}\n"
                            f"The ONLY valid tool names are: {valid_names}\n"
                            "Please retry with a valid tool name, or provide your final answer."
                        ),
                    }
                )
                continue
            self._llm_calls += 1
            if resp.usage:
                self._total_tokens += resp.usage.total_tokens

            choice = resp.choices[0]
            message = choice.message

            if not message.tool_calls:
                return AgentResult(
                    agent_name="ReAct (Groq Tool Calling)",
                    query=query,
                    answer=message.content or "",
                    plan="(step-by-step tool calling — no upfront plan)",
                    llm_calls=self._llm_calls,
                    total_tokens=self._total_tokens,
                    latency_seconds=round(time.time() - start, 2),
                    success=True,
                )

            # Build a clean message dict with only fields Groq accepts
            msg_dict: dict = {"role": "assistant"}
            if message.content:
                msg_dict["content"] = message.content
            if message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]
            messages.append(msg_dict)

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                result = self._execute_tool(fn_name, fn_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        return AgentResult(
            agent_name="ReAct (Groq Tool Calling)",
            query=query,
            answer="[MAX ITERATIONS REACHED]",
            plan="(step-by-step tool calling — no upfront plan)",
            llm_calls=self._llm_calls,
            total_tokens=self._total_tokens,
            latency_seconds=round(time.time() - start, 2),
            success=False,
            failure_reason=f"Exhausted {max_iterations} tool-calling iterations without producing an answer",
        )


def create_react_agent(retriever: Retriever) -> ReactAgent:
    """Create a ReactAgent instance."""
    return ReactAgent(retriever=retriever)
