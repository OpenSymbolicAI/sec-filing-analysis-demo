"""Download and chunk SEC 10-K filings from EDGAR."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from models import FilingMeta, TextChunk

DATA_DIR = Path(__file__).parent / "data" / "filings"
CHUNK_SIZE = 2000  # chars per chunk (~500 tokens)
CHUNK_OVERLAP = 200

# SEC EDGAR requires a User-Agent header with contact info
EDGAR_HEADERS = {
    "User-Agent": "SECAgentDemo/0.1 (demo@example.com)",
    "Accept-Encoding": "gzip, deflate",
}

# Top 10 companies: (ticker, CIK, company name)
COMPANIES = [
    ("AAPL", "0000320193", "Apple Inc."),
    ("MSFT", "0000789019", "Microsoft Corporation"),
    ("GOOGL", "0001652044", "Alphabet Inc."),
    ("AMZN", "0001018724", "Amazon.com Inc."),
    ("TSLA", "0001318605", "Tesla Inc."),
    ("META", "0001326801", "Meta Platforms Inc."),
    ("NVDA", "0001045810", "NVIDIA Corporation"),
    ("NFLX", "0001065280", "Netflix Inc."),
    ("CRM", "0001108524", "Salesforce Inc."),
    ("ADBE", "0000796343", "Adobe Inc."),
]


def _get_latest_10k_url(cik: str) -> FilingMeta | None:
    """Find the latest 10-K filing URL for a given CIK."""
    cik_padded = cik.lstrip("0").zfill(10)
    url = (
        f"https://efts.sec.gov/LATEST/search-index?q=%2210-K%22"
        f"&dateRange=custom&startdt=2024-01-01&enddt=2025-12-31"
        f"&forms=10-K&from=0&size=1"
    )
    # Use the EDGAR full-text search API
    search_url = (
        f"https://efts.sec.gov/LATEST/search-index?"
        f"q=&forms=10-K&dateRange=custom&startdt=2024-01-01&enddt=2025-12-31"
        f"&from=0&size=1"
    )
    # Simpler approach: use the submissions API
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    resp = httpx.get(submissions_url, headers=EDGAR_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    company_name = data.get("name", "")
    ticker = data.get("tickers", [""])[0] if data.get("tickers") else ""

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form == "10-K":
            accession = accessions[i].replace("-", "")
            doc = primary_docs[i]
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik.lstrip('0')}/{accession}/{doc}"
            )
            return FilingMeta(
                company=company_name,
                ticker=ticker,
                cik=cik,
                filing_date=dates[i],
                fiscal_year=dates[i][:4],
                url=filing_url,
            )
    return None


def _download_filing(meta: FilingMeta) -> str:
    """Download the 10-K filing HTML and extract text."""
    cache_path = DATA_DIR / f"{meta.ticker}_{meta.fiscal_year}.txt"
    if cache_path.exists():
        return cache_path.read_text()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    resp = httpx.get(meta.url, headers=EDGAR_HEADERS, timeout=60, follow_redirects=True)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove script/style tags
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    cache_path.write_text(text)
    meta_path = DATA_DIR / f"{meta.ticker}_{meta.fiscal_year}.meta.json"
    meta_path.write_text(meta.model_dump_json(indent=2))

    return text


def _chunk_text(text: str, meta: FilingMeta) -> list[TextChunk]:
    """Split filing text into overlapping chunks."""
    chunks: list[TextChunk] = []
    # Try to detect section headers for better chunking
    sections = _detect_sections(text)

    for section_name, section_text in sections:
        start = 0
        while start < len(section_text):
            end = start + CHUNK_SIZE
            chunk_text = section_text[start:end]
            if chunk_text.strip():
                chunks.append(
                    TextChunk(
                        company=meta.company,
                        ticker=meta.ticker,
                        fiscal_year=meta.fiscal_year,
                        section=section_name,
                        text=chunk_text.strip(),
                    )
                )
            start = end - CHUNK_OVERLAP

    return chunks


def _detect_sections(text: str) -> list[tuple[str, str]]:
    """Best-effort section detection from 10-K text."""
    # Common 10-K section headers
    section_patterns = [
        (r"(?i)item\s+1[.\s]+business", "Business"),
        (r"(?i)item\s+1a[.\s]+risk\s+factors", "Risk Factors"),
        (r"(?i)item\s+6[.\s]+selected\s+financial", "Selected Financial Data"),
        (r"(?i)item\s+7[.\s]+management.s\s+discussion", "MD&A"),
        (r"(?i)item\s+8[.\s]+financial\s+statements", "Financial Statements"),
    ]

    positions: list[tuple[int, str]] = []
    for pattern, name in section_patterns:
        match = re.search(pattern, text)
        if match:
            positions.append((match.start(), name))

    if not positions:
        return [("Full Document", text)]

    positions.sort(key=lambda x: x[0])
    sections: list[tuple[str, str]] = []

    # Text before first section
    if positions[0][0] > 0:
        sections.append(("Preamble", text[: positions[0][0]]))

    for i, (pos, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        sections.append((name, text[pos:end]))

    return sections


def load_all_filings() -> list[TextChunk]:
    """Download and chunk all 10 company filings. Returns all chunks."""
    all_chunks: list[TextChunk] = []

    for ticker, cik, company in COMPANIES:
        print(f"  Loading {ticker} ({company})...")
        meta = _get_latest_10k_url(cik)
        if meta is None:
            print(f"    [WARN] No 10-K found for {ticker}, skipping")
            continue
        text = _download_filing(meta)
        chunks = _chunk_text(text, meta)
        all_chunks.extend(chunks)
        print(f"    {len(chunks)} chunks, {len(text):,} chars")

    return all_chunks


def load_filings_metadata() -> list[FilingMeta]:
    """Load metadata for all cached filings."""
    metas: list[FilingMeta] = []
    if not DATA_DIR.exists():
        return metas
    for meta_path in sorted(DATA_DIR.glob("*.meta.json")):
        metas.append(FilingMeta.model_validate_json(meta_path.read_text()))
    return metas
