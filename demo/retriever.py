"""Retriever backed by LanceDB. Instant startup — no embedding at load time."""

from __future__ import annotations

from pathlib import Path

import lancedb

from config import EMBED_MODEL, get_embed_client
from models import RetrievedDoc, TextChunk

DB_PATH = Path(__file__).parent / "data" / "lancedb"
TABLE_NAME = "sec_chunks"

# Map common short names to the exact company names/tickers stored in the DB.
_COMPANY_ALIASES: dict[str, list[str]] = {
    "google": ["alphabet"],
    "amazon": ["amazon com"],
    "meta": ["meta platforms"],
    "nvidia": ["nvidia corp"],
    "netflix": ["netflix inc"],
    "adobe": ["adobe inc"],
    "salesforce": ["salesforce"],
    "tesla": ["tesla"],
    "apple": ["apple inc"],
    "microsoft": ["microsoft corp"],
}


class Retriever:
    """Vector search over pre-embedded SEC filing chunks in LanceDB."""

    def __init__(self) -> None:
        db = lancedb.connect(str(DB_PATH))
        self._table = db.open_table(TABLE_NAME)
        self._embed_client = get_embed_client()
        self._count = self._table.count_rows()

    @property
    def chunk_count(self) -> int:
        return self._count

    def _embed_query(self, query: str) -> list[float]:
        resp = self._embed_client.embeddings.create(input=[query], model=EMBED_MODEL)
        return resp.data[0].embedding

    def search(self, query: str, k: int = 5, company: str | None = None) -> list[RetrievedDoc]:
        """Search for chunks most similar to query. Optionally filter by company."""
        q_vec = self._embed_query(query)

        search = self._table.search(q_vec).limit(k * 3 if company else k)

        if company:
            company_lower = company.lower()
            # Build a list of terms to match: the original name + any aliases
            terms = [company_lower]
            terms.extend(_COMPANY_ALIASES.get(company_lower, []))
            clauses = []
            for term in terms:
                safe_term = term.replace("'", "''")
                clauses.append(f"lower(company) LIKE '%{safe_term}%'")
                clauses.append(f"lower(ticker) LIKE '%{safe_term}%'")
            search = search.where(" OR ".join(clauses))

        results = search.limit(k).to_list()

        docs: list[RetrievedDoc] = []
        for row in results:
            docs.append(
                RetrievedDoc(
                    chunk=TextChunk(
                        company=row["company"],
                        ticker=row["ticker"],
                        fiscal_year=row["fiscal_year"],
                        section=row["section"],
                        text=row["text"],
                    ),
                    score=1.0 - row.get("_distance", 0.0),
                )
            )
        return docs
