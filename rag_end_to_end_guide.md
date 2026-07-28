# RAG End-to-End on Cloud SQL — Detailed Guide

**Notebook:** `backend_py/rag_end_to_end_psycopg2_v1.ipynb`

This guide explains, in plain language and in detail, every step of the notebook: what
it does, the code behind it, why it's there, what to change, and how to read the
results. It's written for someone new to RAG.

---

## 0. The big picture — what is RAG?

**RAG = Retrieval-Augmented Generation.** To answer a question about *your* documents,
a plain LLM isn't enough — it doesn't know your data. RAG fixes that in five moves:

1. **Chunk** — cut each document into small pieces.
2. **Embed** — turn each piece into a *vector* (a list of numbers) that captures its
   meaning. Pieces with similar meaning get vectors that are close together.
3. **Store** — save those vectors in PostgreSQL using the **pgvector** extension.
4. **Retrieve** — for a question, embed it too, then find the stored pieces whose
   vectors are closest (this is *vector search*).
5. **Generate** — hand those pieces to Gemini and let it write the answer.

The two choices that most affect quality are **which embedding model** and **which
chunking method** you use. There is no universal best — it depends on your data. So this
notebook tries several of each and **measures** them to find the best for you.

```
document ─▶ chunk ─▶ embed ─▶ store in pgvector
                                     │
question ─▶ embed ─▶ vector search ──┘─▶ top pieces ─▶ Gemini ─▶ answer
```

---

## 1. Key terms (glossary)

| Term | Plain meaning |
|---|---|
| **Embedding** | A list of numbers representing the meaning of a piece of text. Same model always makes comparable numbers. |
| **Dimension** | How long that list is (e.g. 768 or 3072). Each model has a fixed dimension. |
| **Chunk** | One small piece of a document. |
| **pgvector** | A Postgres extension that stores vectors and finds nearest ones fast. |
| **Cosine distance (`<=>`)** | How far apart two vectors are in meaning: 0 = identical, ~1 = unrelated. |
| **HNSW index** | A special index that makes nearest-vector search fast on big tables. |
| **Gold set** | Your test questions, each labelled with the document that should answer it. Used to score methods. |
| **recall@k** | Of your test questions, the fraction where the right document appears in the top *k* results. |
| **MRR** | Mean Reciprocal Rank — how *high* the right answer ranks on average (1.0 = always first). |

---

## 2. Prerequisites

- A **Cloud SQL for PostgreSQL** instance you can reach (directly, or via the Cloud SQL
  Auth Proxy running locally so `host="localhost"` works).
- The **pgvector** extension available on that instance (Cloud SQL supports it).
- **Vertex AI** access in your GCP project, authenticated with Application Default
  Credentials: run once in a terminal — `gcloud auth application-default login` — and
  make sure the Vertex AI API is enabled.
- Python packages (installed by Step 1): `google-genai`, `psycopg2-binary`, `tiktoken`,
  `pandas`.

---

## 3. Step-by-step walkthrough

### Step 1 — Install
```python
%pip install -q google-genai psycopg2-binary tiktoken pandas
```
Installs the client libraries: `google-genai` (Vertex embeddings + Gemini), `psycopg2`
(Postgres), `tiktoken` (token-based chunking), `pandas` (tables). If pip says "restart
kernel," do it before continuing.

### Step 2 — Configuration
```python
PROJECT_ID = "your-gcp-project-id"     # CHANGE
LOCATION   = "us-central1"
CHAT_MODEL = "gemini-2.5-flash"
DB_CONFIG  = { host, port, dbname, user, password }   # copy from your working notebook
EMBEDDERS  = ["text-embedding-004", "text-embedding-005", "gemini-embedding-001"]
CHUNKERS   = ["fixed_400", "sentence", "token_256"]
TOP_K      = 3
```
This is the only cell you *must* edit:
- **`PROJECT_ID`** — your GCP project.
- **`DB_CONFIG`** — copy the exact values from your `vector_search_psycopg2` /
  `sql_query_expert` notebook. If you use the Cloud SQL Auth Proxy, keep
  `host="localhost"` and make sure the proxy is running.
- **`EMBEDDERS`** — the embedding models to compare. Add/remove freely. `text-embedding-004`
  is the "text embedding 4" you mentioned.
- **`CHUNKERS`** — the chunking methods to compare (defined in Step 6).
- **`TOP_K`** — how many pieces to retrieve/score per question.

### Step 3 — Connect to Cloud SQL
```python
def get_conn():
    return psycopg2.connect(**DB_CONFIG)
```
Opens a connection using your `DB_CONFIG` (same pattern as your other notebook) and runs
a trivial query to prove the connection works. If it fails, the error tells you to check
`DB_CONFIG` / the proxy. `get_conn()` is reused everywhere else.

### Step 4 — Enable pgvector
```python
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
```
One-time setup. `IF NOT EXISTS` makes it safe to re-run. Without this, the `vector` column
type doesn't exist and later steps fail.

### Step 5 — Vertex embedding function
```python
def embed_texts(model, texts, task_type):
    # batches of 50; falls back to one-at-a-time if a model rejects batches
    ...
```
One function that embeds text with **any** Vertex model. Two important ideas:
- **Same model for documents and questions.** Vectors from different models live in
  different "coordinate systems" — mixing them gives nonsense. The notebook always uses
  the same model on both sides.
- **`task_type`** tells the model the *purpose*: `RETRIEVAL_DOCUMENT` when embedding
  stored text, `RETRIEVAL_QUERY` when embedding a search question. This improves
  retrieval quality.

It batches (50 at a time) for speed and, if a model only accepts one input per call,
automatically falls back to one-at-a-time. A probe call at the end confirms Vertex works
and prints the dimension.

### Step 6 — Chunking methods (three techniques)
```python
CHUNKER_FUNCS = {
    "fixed_400": lambda t: chunk_fixed(t, 400, 60),
    "sentence":  lambda t: chunk_sentence(t, 500),
    "token_256": lambda t: chunk_tokens(t, 256, 40),
}
```
Each function turns one document into a list of text pieces:
- **`fixed_400`** — a sliding window of ~400 characters with 60 chars of overlap.
  Simplest and most predictable; may cut mid-sentence.
- **`sentence`** — packs whole sentences together up to ~500 chars. Respects natural
  boundaries, so each piece reads cleanly.
- **`token_256`** — windows of 256 *model tokens* with 40 overlap (tokens ≈ word-pieces).
  Best when you must respect a model's token limit precisely.

**Overlap** means neighbouring pieces share some text, so an answer sitting on a boundary
isn't split away. The cell prints each method on a demo sentence so you can see the
difference.

### Step 7 — Your documents + gold set
```python
docs = { "acme-produce-2024": "...", ... }        # what to index
gold = { "What did Acme quote for romaine?": "acme-produce-2024", ... }  # test set
```
- **`docs`** — the text to index (keyed by a document id). Replace with your real RFP/bid
  text. In production you'd load this from the Document Vault uploads.
- **`gold`** — the evaluation set: each question maps to the document that *should* answer
  it. **This is the most important input for choosing a method.** With only a handful of
  questions the scores are rough; add 20–50 realistic questions for a trustworthy result.

### Step 8 — Ingest: chunk + embed + store (one table per combo)
```python
def ingest(model, chunker):
    # chunk every doc, embed all pieces, auto-detect dimension,
    # create rag_bench_<model>_<chunker>, insert, build HNSW index
    ...
```
For a given (model, chunker) pair it:
1. Chunks every document with the chosen chunker.
2. Embeds all pieces with the chosen model (`RETRIEVAL_DOCUMENT`).
3. **Auto-detects the dimension** from the first vector — so you never have to know
   whether a model is 768 or 3072; mixing models "just works," each in its own table.
4. Creates a dedicated table `rag_bench_<model>_<chunker>` and bulk-inserts the pieces
   and vectors (`execute_values` for speed).
5. Builds an **HNSW cosine index** so search is fast.

Each combo is isolated in its own table, which is what makes the later comparison fair.

### Step 9 — Vector search
```python
def search(table, model, question, top_k):
    # embed the question with the SAME model, then:
    # SELECT ... ORDER BY embedding <=> :qvec::vector LIMIT top_k
    ...
```
Embeds the question (`RETRIEVAL_QUERY`) and asks Postgres for the `top_k` closest pieces
by cosine distance (`<=>`). Returns each piece's `doc_id`, `content`, and `distance`
(smaller = more relevant).

### Step 10 — Benchmark every combo → leaderboard
```python
def score(table, model):
    # for each gold question: is the expected doc in the top-k? how high?
    # returns (recall@k, MRR)
```
The main event. For every (model × chunker) combo it ingests, then scores the gold
questions:
- **recall@k** — fraction of questions whose correct document is in the top *k*.
- **MRR** — average of `1 / rank_of_first_correct` (rewards ranking the right doc higher).

It prints a line per combo, and combos whose model isn't enabled in your project are
**skipped and reported** (they don't stop the run). The second cell sorts everything into
a **leaderboard** (best MRR first, then recall) and names the winner, capturing
`BEST_MODEL` and `BEST_TABLE` for the next step.

### Step 11 — Answer with Gemini (winning combo)
```python
def ask(question, top_k):
    hits = search(BEST_TABLE, BEST_MODEL, question, top_k)
    # build a context from the retrieved pieces, ask Gemini to answer ONLY from it
```
Uses the winning table + model to retrieve, then instructs Gemini to answer **only** from
the retrieved context (and say "I don't know" otherwise). This grounding is what keeps the
answer faithful to your documents instead of made up.

### Step 12 — Notes & cleanup
Explains how to read the leaderboard, suggests next steps (hybrid search + reranking via
the `services/rag/` package), and drops the temporary `rag_bench_*` tables. **Skip the
cleanup cell if you want to keep the winning table** for production use.

---

## 4. How to choose the best method (reading the leaderboard)

1. **Top row wins** — highest MRR then recall on your gold set.
2. **Trust the gold set, not vibes.** 5 questions is a demo; 20–50 real questions give a
   reliable ranking.
3. **Break ties by cost/speed.** A 768-dim model (`text-embedding-004/005`) is cheaper to
   store and faster to search than a 3072-dim one (`gemini-embedding-001`); larger chunks
   mean fewer embeddings to compute and store.
4. **Re-run when your data changes.** The best combo can shift with a different corpus.

Once you pick a winner, use those two choices (embedding model + chunking method) for your
real ingestion.

---

## 5. Going further (for even better results)

- **Hybrid search** — combine vector search with keyword (full-text) search and fuse the
  results (Reciprocal Rank Fusion). This catches exact SKUs, codes, and names that pure
  vector search misses. Already implemented in `backend_py/services/rag/` — see
  `docs/rag_pgvector_guide.md`.
- **Reranking** — a second model reorders the top candidates by true relevance, lifting
  precision. Also in `services/rag/` (`RAG_RERANKER=cross_encoder` or `vertex`).
- **Bigger gold set + metrics** — as you add questions, recall@k and MRR become more
  meaningful; keep the gold set in version control and re-benchmark when you change models.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `could not connect to server` | `DB_CONFIG` wrong, or the Cloud SQL Auth Proxy isn't running (Steps 2–3). |
| `403 PERMISSION_DENIED` from Vertex | `gcloud auth application-default login`, and enable the Vertex AI API. |
| `type "vector" does not exist` | Re-run Step 4 against the correct database. |
| A model is skipped in Step 10 | That model isn't enabled in your project/region — remove it from `EMBEDDERS`. |
| Search is slow on a large table | The HNSW index is created in Step 8; for very large tables, tune index parameters later. |
| Good pieces retrieved but weak answers | Raise `TOP_K`, or refine the prompt in Step 11. |
| Dimension looks wrong / mismatched | Dimension is auto-detected per model, so this usually means an embedding call failed — check the Vertex error. |

---

## 7. Files

| File | Role |
|---|---|
| `backend_py/rag_end_to_end_psycopg2_v1.ipynb` | The notebook this guide documents |
| `backend_py/services/rag/` | The production package (hybrid search + rerank) |
| `backend_py/docs/rag_pgvector_guide.md` | Guide for the production package |
| `backend_py/vector_search_psycopg2 (1).ipynb` | Your original connection reference |
