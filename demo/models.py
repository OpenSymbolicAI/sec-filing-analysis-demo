"""Shared Pydantic models for the SEC agent demo."""

from pydantic import BaseModel


class FilingMeta(BaseModel):
    """Metadata for a single SEC 10-K filing."""

    company: str
    ticker: str
    cik: str
    filing_date: str
    fiscal_year: str
    url: str


class TextChunk(BaseModel):
    """A chunk of text from a filing with metadata."""

    company: str
    ticker: str
    fiscal_year: str
    section: str
    text: str


class RetrievedDoc(BaseModel):
    """A document returned by retrieval with its similarity score."""

    chunk: TextChunk
    score: float


class DemoQuery(BaseModel):
    """A demo query with label, query text, and expected ground truth."""

    label: str
    query: str
    ground_truth: str


class AgentResult(BaseModel):
    """Result from running a query through an agent."""

    agent_name: str
    query: str
    answer: str
    plan: str = ""
    llm_calls: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    success: bool = True
    failure_reason: str = ""


class QueryLog(BaseModel):
    """Log entry for a single query run through both agents."""

    label: str
    query: str
    ground_truth: str
    react_result: AgentResult | None = None
    behavior_result: AgentResult | None = None
    react_eval: "EvalScore | None" = None
    behavior_eval: "EvalScore | None" = None


class RetrievedSnippet(BaseModel):
    """A flattened retrieval result for use in behaviour primitives."""

    company: str
    ticker: str
    fiscal_year: str
    section: str
    text: str
    score: float


class ExtractedMetrics(BaseModel):
    """Mapping of metric names to their extracted values."""

    values: dict[str, str]


class EvalScore(BaseModel):
    """LLM-as-judge evaluation of an answer against ground truth."""

    correctness: int  # 1-5: factual accuracy vs ground truth
    completeness: int  # 1-5: coverage of key facts from ground truth
    relevance: int  # 1-5: answer stays on-topic, no irrelevant info
    faithfulness: int  # 1-5: no hallucinated facts beyond source material
    conciseness: int  # 1-5: answer is appropriately brief without losing key info
    correctness_reason: str
    completeness_reason: str
    relevance_reason: str
    faithfulness_reason: str
    conciseness_reason: str
