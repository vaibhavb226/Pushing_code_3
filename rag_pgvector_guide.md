# Hybrid RAG over Postgres + pgvector — Guide

Production-grade Retrieval-Augmented Generation for RFP AIQ: **pluggable chunking
and embedding models**, stored in **Postgres/pgvector**, retrieved with **vector +
full-text search fused by Reciprocal Rank Fusion (RRF)**, and an optional
**reranker**. Everything is swapped via `.env` — no code changes.

Package: `backend_py/services/rag/`

| File | Role |
|---|---|
| `config.py` | `RagSettings` — every knob, read from `.env` |
| `chunking.py` | `recursive` / `token` / `semantic` chunkers (pluggable) |
| `embeddings.py` | `vertexai` / `openai` embedders (pluggable) |
| `store.py` | `PgVectorStore` — schema + hybrid RRF search |
| `reranker.py` | `none` / `cross_encoder` / `vertex` rerankers (pluggable) |
| `pipeline.py` | `RagPipeline` — ingest → retrieve → answer |

---

## 1. Why this design (the "best results" part)

- **Hybrid search, not just vectors.** Pure vector search misses exact SKUs, part
  numbers, supplier names, and codes that are common in bid/RFP text. Full-text
  (`tsvector`) catches those. Running both and fusing with **RRF** gives higher
  recall *and* precision than either alone — this is the single biggest quality win.
- **Pluggable embeddings & chunking.** Swap models/techniques from `.env` to compare
  quality without touching code (exactly what you asked for).
- **Reranking** reorders candidates by true query-passage relevance for a further
  precision lift when you need it.
- **One SQL statement** does the fusion server-side (fast; no Python-side merging).

```
query ─▶ embed ─┬─▶ vector search (HNSW/cosine) ─┐
                └─▶ full-text search (GIN/tsvector) ┴─▶ RRF fuse ─▶ rerank ─▶ top-k ─▶ LLM answer
```

---

## 2. Install

```bash
cd backend_py
pip install "pgvector>=0.2" sqlalchemy asyncpg tiktoken \
            langchain-google-vertexai langchain-openai
# optional, only if RAG_RERANKER=cross_encoder:
pip install sentence-transformers
# optional, only if RAG_RERANKER=vertex:
pip install google-cloud-discoveryengine
```

Postgres must have the **pgvector** extension available (Cloud SQL for PostgreSQL
supports it — enable it once; `ensure_schema()` runs `CREATE EXTENSION IF NOT EXISTS
vector`).

---

## 3. Configure (`.env`)

```ini
# Where the vectors live (defaults to your SQL_EXPERT_DB_* Postgres if left blank)
RAG_DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname

# Embedding model (pluggable) — model and dim MUST match
RAG_EMBED_PROVIDER=vertexai            # vertexai | openai
RAG_EMBED_MODEL=text-embedding-005
RAG_EMBED_DIM=768                       # 768 vertex; 1536/3072 openai 3-small/large

# Chunking (pluggable)
RAG_CHUNKER=recursive                   # recursive | token | semantic
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=120

# Retrieval
RAG_TOP_K=6
RAG_CANDIDATE_K=40
RAG_RRF_K=60
RAG_RERANKER=none                       # none | cross_encoder | vertex
```

> **Dimension rule:** the pgvector column width is fixed at table-creation time to
> `RAG_EMBED_DIM`. If you switch to an embedding model with a different width, change
> `RAG_EMBED_DIM` and recreate the table (or use a new table name). The embedder
> asserts the returned dim matches, so mismatches fail loudly, not silently.

---

## 4. Use

```python
from services.rag import RagPipeline
from dependencies import get_llm_service   # this repo's LLMService

rag = RagPipeline.from_settings()
await rag.setup()                          # create extension/table/indexes (idempotent)

# Ingest (e.g. extracted text from a Document Vault upload)
await rag.ingest("doc-123", full_text, metadata={"bid_id": "BID-2024-089"})

# Retrieve passages (hybrid + rerank)
hits = await rag.retrieve("Acme response rate", metadata_filter={"bid_id": "BID-2024-089"})
for h in hits:
    print(h.score, h.doc_id, h.content[:80])

# Grounded answer with citations
ans = await rag.answer("What did Acme quote for produce?", llm=get_llm_service())
print(ans.answer)                          # cites [1], [2], ...
await rag.close()
```

`metadata_filter` uses JSONB containment (`metadata @> filter`), so you can scope
retrieval to one bid, supplier, or document type.

---

## 5. Compare models/techniques (what you're after)

Because chunker, embedder, and reranker are all `.env`-selected, you can benchmark
combinations by re-ingesting into a fresh table name and measuring retrieval quality:

| Try | Change |
|---|---|
| Chunking | `RAG_CHUNKER=recursive` vs `token` vs `semantic` |
| Embeddings | `vertexai/text-embedding-005` vs `openai/text-embedding-3-large` (set dim!) |
| Rerank | `RAG_RERANKER=none` vs `cross_encoder` vs `vertex` |
| Fusion weight | `RAG_RRF_K` lower ⇒ top ranks dominate more |

Keep a small labelled question→expected-passage set and score `retrieve()` hits
(recall@k / MRR) per combination. That's how you land on "best results" for *your*
corpus rather than guessing.

---

## 6. Schema (created by `ensure_schema()`)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE rag_chunks (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id      text NOT NULL,
    chunk_index int  NOT NULL,
    content     text NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding   vector(768) NOT NULL,          -- width = RAG_EMBED_DIM
    embed_model text NOT NULL,
    metadata    jsonb NOT NULL DEFAULT '{}',
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (doc_id, chunk_index)
);
CREATE INDEX rag_chunks_tsv_idx  ON rag_chunks USING gin  (content_tsv);
CREATE INDEX rag_chunks_meta_idx ON rag_chunks USING gin  (metadata);
CREATE INDEX rag_chunks_hnsw_idx ON rag_chunks USING hnsw (embedding vector_cosine_ops);
```

---

## 7. Notes & tuning

- **HNSW vs IVFFlat:** HNSW (used here) needs no training and is fast/accurate for
  moderate corpora. For very large tables, consider IVFFlat with a tuned `lists`.
- **Recall knob:** raise `RAG_CANDIDATE_K` to feed the reranker more candidates
  (higher recall, slightly slower).
- **Cross-encoder cost:** `cross_encoder` runs locally and adds latency proportional
  to `RAG_CANDIDATE_K`; keep candidates modest (~40) or use `vertex` for a managed path.
- **Re-embedding on model change:** vectors from different models are not comparable —
  always re-ingest when you change `RAG_EMBED_MODEL`.
- **Read-only vs write:** ingestion writes to `rag_chunks`; keep that DB role separate
  from the read-only role used by the SQL Query Expert.
```
