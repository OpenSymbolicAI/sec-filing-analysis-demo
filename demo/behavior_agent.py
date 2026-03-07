"""Behavior programming agent for SEC filing analysis."""

from __future__ import annotations

import json as _json
import re
from typing import TYPE_CHECKING

from opensymbolicai.blueprints import DesignExecute
from opensymbolicai.core import decomposition, primitive
from opensymbolicai.llm import LLMConfig

from config import LLM_API_KEY, LLM_BASE_URL, MODEL, get_llm_client
from models import ExtractedMetrics, RetrievedSnippet
from retriever import Retriever

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion


class SECAnalyst(DesignExecute):
    """Analyses SEC 10-K filings using behaviour programming."""

    def __init__(self, retriever: Retriever, llm: LLMConfig) -> None:
        super().__init__(llm=llm)
        self._retriever = retriever
        self.total_tokens = 0
        self.llm_calls = 0

    def reset_metrics(self) -> None:
        self.total_tokens = 0
        self.llm_calls = 0

    def _track(self, resp: ChatCompletion) -> None:
        self.llm_calls += 1
        if resp.usage:
            self.total_tokens += resp.usage.total_tokens

    def _ask(self, system: str, user: str, max_tokens: int = 500) -> str:
        client = get_llm_client()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=max_tokens,
        )
        self._track(resp)
        return resp.choices[0].message.content or ""

    # ── Primitives ──────────────────────────────────────────────

    @primitive(read_only=True)
    def retrieve_for_company(self, query: str, company: str, k: int = 5) -> list[RetrievedSnippet]:
        """Search SEC filings filtered to a specific company."""
        docs = self._retriever.search(query, k=k, company=company)
        return [
            RetrievedSnippet(
                company=d.chunk.company, ticker=d.chunk.ticker,
                fiscal_year=d.chunk.fiscal_year, section=d.chunk.section,
                text=d.chunk.text, score=d.score,
            )
            for d in docs
        ]

    @primitive(read_only=True)
    def combine_texts(self, docs: list[RetrievedSnippet]) -> str:
        """Combine retrieved documents into a single context string."""
        return "\n\n---\n\n".join(
            f"[{d.company} | {d.section}]\n{d.text}" for d in docs
        )

    @primitive(read_only=True)
    def extract_number(self, text: str, metric: str) -> str:
        """Extract a financial metric from text. Returns e.g. '$57,372 million' or 'NOT_FOUND'."""
        return self._ask(
            "Extract the requested metric. Normalize to millions. Use the most recent year's total/consolidated figure. "
            "Return ONLY the number with unit, e.g. '$57,372 million'. Return 'NOT_FOUND' if absent.",
            f"Metric: {metric}\n\nText:\n{text}",
        ) or "NOT_FOUND"

    @primitive(read_only=True)
    def extract_metrics(self, text: str, metrics: list[str]) -> ExtractedMetrics:
        """Extract multiple metrics in one call. Returns structured metric→value mapping."""
        metrics_list = "\n".join(f"- {m}" for m in metrics)
        raw = self._ask(
            "Extract each metric's value. Return JSON mapping metric name to value. "
            "Use 'NOT_FOUND' if absent.",
            f"Metrics:\n{metrics_list}\n\nText:\n{text}",
        )
        try:
            return ExtractedMetrics(values=_json.loads(raw))
        except _json.JSONDecodeError:
            return ExtractedMetrics(values={m: "NOT_FOUND" for m in metrics})

    @primitive(read_only=True)
    def extract_count(self, text: str, metric: str) -> str:
        """Extract a non-monetary count (e.g. headcount). Returns integer string or 'NOT_FOUND'."""
        return self._ask(
            "Extract the count as a plain integer, e.g. '16000'. Return 'NOT_FOUND' if absent.",
            f"Metric: {metric}\n\nText:\n{text}",
            max_tokens=100,
        ) or "NOT_FOUND"

    @primitive(read_only=True)
    def answer_from_context(self, context: str, question: str) -> str:
        """Answer a question using only the provided context."""
        answer = self._ask(
            "Answer using ONLY the provided context. Be precise with numbers.",
            f"Context:\n{context}\n\nQuestion: {question}",
        )
        return answer or "Unable to determine answer from the available context."

    @primitive(read_only=True)
    def compute_ratio(self, numerator: str, denominator: str) -> str:
        """Compute a ratio from two number strings."""
        def _parse(s: str) -> float | None:
            s = s.replace(",", "").replace("$", "").strip()
            match = re.search(r"[\d.]+", s)
            return float(match.group()) if match else None
        n, d = _parse(numerator), _parse(denominator)
        if n is None or d is None or d == 0:
            return "UNABLE_TO_COMPUTE"
        return f"{n / d:.2f}"

    # ── Behaviours ──────────────────────────────────────────────

    @decomposition(
        intent="What was Salesforce's operating income in its most recent fiscal year?",
        expanded_intent="Retrieve filings for the company, combine, answer.",
    )
    def _simple_extraction(self) -> str:
        docs = self.retrieve_for_company("Salesforce operating income consolidated statements of operations", company="Salesforce", k=8)
        context = self.combine_texts(docs)
        result = self.answer_from_context(context, "What was Salesforce's operating income in its most recent fiscal year?")
        return result

    # @decomposition(
    #     intent="Compare the total revenue of Amazon and Netflix in their most recent fiscal year.",
    #     expanded_intent="Per-company: retrieve, combine all, answer.",
    # )
    # def _compare_across_companies(self) -> str:
    #     companies = ["Amazon", "Netflix"]
    #     all_contexts = []
    #     for company in companies:
    #         docs = self.retrieve_for_company(
    #             f"{company} total revenue net sales consolidated statements of operations",
    #             company=company, k=8,
    #         )
    #         all_contexts.append(self.combine_texts(docs))
    #     full_context = "\n\n===\n\n".join(all_contexts)
    #     result = self.answer_from_context(full_context, "Compare the total revenue of Amazon and Netflix in their most recent fiscal year.")
    #     return result

    # @decomposition(
    #     intent="Which company had a higher operating margin in its most recent fiscal year: Tesla or NVIDIA?",
    #     expanded_intent="Per-company: retrieve financial statements, combine, let LLM compute derived metric.",
    # )
    # def _compare_derived_metric(self) -> str:
    #     companies = ["Tesla", "NVIDIA"]
    #     all_contexts = []
    #     for company in companies:
    #         docs = self.retrieve_for_company(
    #             f"{company} operating income revenue consolidated statements of operations",
    #             company=company, k=8,
    #         )
    #         all_contexts.append(self.combine_texts(docs))
    #     full_context = "\n\n===\n\n".join(all_contexts)
    #     result = self.answer_from_context(
    #         full_context,
    #         "Which company had a higher operating margin: Tesla or NVIDIA? "
    #         "Calculate operating margin as operating income divided by total revenue.",
    #     )
    #     return result

    # @decomposition(
    #     intent="Among Apple, Microsoft, Google, Amazon, Tesla, Meta, NVIDIA, Netflix, Salesforce, and Adobe, which has the highest total revenue?",
    #     expanded_intent="Per-company: retrieve, extract metric, build summary, answer.",
    # )
    # def _find_max_single_metric(self) -> str:
    #     companies = ["Apple", "Microsoft", "Google", "Amazon", "Tesla",
    #                   "Meta", "NVIDIA", "Netflix", "Salesforce", "Adobe"]
    #     results = []
    #     for company in companies:
    #         docs = self.retrieve_for_company(
    #             f"{company} total revenue net sales consolidated statements of operations", company=company, k=8,
    #         )
    #         context = self.combine_texts(docs)
    #         value = self.extract_number(context, "total revenue")
    #         results.append(f"{company}: {value}")
    #     summary = "\n".join(results)
    #     result = self.answer_from_context(summary, "Which company has the highest total revenue?")
    #     return result


def create_behavior_agent(retriever: Retriever) -> SECAnalyst:
    """Create an SECAnalyst with default config."""
    config = LLMConfig(
        provider="openai",
        model=MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )
    return SECAnalyst(retriever=retriever, llm=config)
