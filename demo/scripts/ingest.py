"""One-time ingestion: download SEC filings, chunk, embed via Fireworks, store in LanceDB.

Usage:
    uv run python scripts/ingest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent dir so we can import demo modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
import numpy as np
import pyarrow as pa

from config import EMBED_MODEL, get_embed_client
from sec_loader import load_all_filings

DB_PATH = Path(__file__).parent.parent / "data" / "lancedb"
TABLE_NAME = "sec_chunks"


def _embed_batch(client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via Fireworks."""
    resp = client.embeddings.create(
        input=[t[:4000] for t in texts],
        model=EMBED_MODEL,
    )
    return [d.embedding for d in resp.data]


def main() -> None:
    print("=== SEC 10-K Ingestion Pipeline ===\n")

    # 1. Download & chunk
    print("[1/3] Downloading and chunking filings...")
    chunks = load_all_filings()
    print(f"  Total chunks: {len(chunks)}\n")

    # 2. Embed
    print("[2/3] Embedding chunks via Fireworks...")
    client = get_embed_client()
    all_embeddings: list[list[float]] = []
    batch_size = 256

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        embeds = _embed_batch(client, texts)
        all_embeddings.extend(embeds)
        done = min(i + batch_size, len(chunks))
        print(f"  {done}/{len(chunks)} embedded")

    # 3. Store in LanceDB
    print(f"\n[3/3] Writing to LanceDB at {DB_PATH}...")
    DB_PATH.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(DB_PATH))

    # Build pyarrow table
    records = []
    for chunk, embedding in zip(chunks, all_embeddings):
        records.append(
            {
                "company": chunk.company,
                "ticker": chunk.ticker,
                "fiscal_year": chunk.fiscal_year,
                "section": chunk.section,
                "text": chunk.text,
                "vector": embedding,
            }
        )

    # Drop existing table if re-ingesting
    try:
        db.drop_table(TABLE_NAME)
    except Exception:
        pass

    table = db.create_table(TABLE_NAME, data=records)
    print(f"  Stored {len(records)} chunks with {len(all_embeddings[0])}-dim vectors")
    print(f"\n  Done. LanceDB path: {DB_PATH}")
    print(f"  Table: {TABLE_NAME}")
    print(f"  Run the demo with: uv run python main.py -q 1")


if __name__ == "__main__":
    main()
