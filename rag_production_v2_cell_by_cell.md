# `rag_production_v2.ipynb` — Cell-by-Cell Beginner's Walkthrough

**Notebook explained:** `rag_production_v2.ipynb` (42 cells)
**Audience:** someone new to RAG who has to understand *and defend* every line.
**Companion:** `rag_production_v2_guide.md` explains v2 thematically. **This** document walks
the notebook **cell by cell, in execution order**, which is what you asked for.

---

## How to read this document

Every cell gets the same four-part treatment:

| Heading | What you get |
|---|---|
| **What it does** | Plain English, no jargon |
| **Why it's needed** | What breaks if you delete it |
| **Line by line** | The code, explained in order |
| **Example** | Concrete text/numbers so you can picture it |

Cell numbers are 0-indexed and match the notebook exactly.

---

# PART A — The mental model (read this before any code)

## A.1 What problem does RAG solve?

Gemini has never seen Sysco's bid documents. Ask it *"What did Acme quote for romaine?"* and it
will either say "I don't know" or — far worse — **invent a plausible-looking price.** In
procurement, an invented number is a catastrophe, not an inconvenience.

You cannot retrain the model on your documents: it costs a fortune, takes weeks, and goes stale
the moment a new bid lands. So you do the cheap, reliable thing instead:

> **Find the 3–4 most relevant paragraphs from your own documents, paste them into the prompt,
> and instruct the model: "answer using only this, and cite where each fact came from."**

That is **RAG — Retrieval-Augmented Generation.**

- **Retrieval** = the finding step. *This is the hard part, and it's where 90% of this notebook lives.*
- **Augmented Generation** = the model writes an answer augmented by what you found.

**The single most important consequence:** if retrieval doesn't find the right text, no prompt
in the world can save you. Retrieval quality is the **ceiling** on your entire system. That is
why v2 spends eight cells on retrieval and one on generation.

## A.2 The pipeline

```
  ═══════════ INGESTION (offline, once per document) ═══════════

  document ──▶ CHUNK ──▶ add HEADER ──▶ EMBED ──▶ store in Postgres
                                                   (vector + tsvector)

  ═══════════ SERVING (on every question) ═══════════

  question ──▶ EMBED ──┐
                       ├──▶ HYBRID SEARCH ──▶ RRF fuse ──▶ 20 candidates
  question ──▶ KEYWORD ┘     (vector + text)                    │
                                                                ▼
                                                          RERANK (keep 4)
                                                                │
                                                                ▼
                                                    GENERATE (Gemini, cited)
                                                                │
                                                                ▼
                                                       answer + sources + timings
```

## A.3 Why "meaning as numbers" works

An **embedding model** maps text into a high-dimensional space where **distance = difference in
meaning**. A toy 2-D version:

| Text | Vector (pretend 2-D) |
|---|---|
| "romaine lettuce price" | `[0.91, 0.12]` |
| "cost of lettuce per case" | `[0.88, 0.15]` ← **very close** |
| "payment terms Net-30" | `[0.10, 0.94]` ← far away |

Rows 1 and 2 share **zero keywords** — "romaine"/"lettuce", "price"/"cost" — yet land next to
each other. That is **semantic search**. A `LIKE '%romaine%'` would miss row 2 entirely.

Real models use 1536 numbers instead of 2. Same idea, more room.

## A.4 …and why meaning alone is not enough

Now try the opposite case:

| Query | Semantic search | Keyword search |
|---|---|---|
| *"who delivers fastest?"* | ✅ understands paraphrase | ❌ no shared words |
| *"BID-2024-089"* | ❌ **IDs embed terribly** | ✅ exact token match |
| *"4.8/5"* | ❌ | ✅ |

Embeddings compress meaning — and an identifier like `BID-2024-089` has almost no *meaning* to
compress. It's a label. Two different bid IDs embed to nearly the same place because they look
alike. This is a well-known, fundamental blind spot.

**So v2 runs both searches and merges them.** That is **hybrid retrieval**, and it is the single
biggest upgrade over v1.1.

## A.5 Vocabulary

| Term | Plain meaning |
|---|---|
| **Chunk** | A document cut into a bite-sized piece (~350 tokens here). You retrieve chunks, not whole documents. |
| **Embedding / vector** | A list of 1536 numbers representing a piece of text's meaning. |
| **pgvector** | The Postgres extension adding the `vector` column type and the `<=>` operator. |
| **Cosine distance (`<=>`)** | Meaning-distance. **0 = identical**, ~1 = unrelated. Sorted **ascending** — nearest first. |
| **ANN / HNSW** | *Approximate Nearest Neighbour.* A graph index that finds near-certain best matches by checking a few hundred rows instead of 10 million. |
| **tsvector / full-text search** | Postgres's built-in **keyword** search. Knows nothing about meaning; nails exact strings. |
| **GIN index** | The index type that makes `tsvector` and `jsonb` lookups fast. Like a book's back-index: word → rows containing it. |
| **Hybrid search** | Running semantic + keyword search and merging the two ranked lists. |
| **RRF** | *Reciprocal Rank Fusion* — the merge formula. Fuses on **position**, not score. |
| **Reranker** | A second, slower, smarter pass. Stage 1 grabs 40 candidates fast; the reranker reads them properly and keeps 4. |
| **Contextual header** | A one-line "where did this come from" banner prepended to each chunk **before embedding**. |
| **`task_type`** | Tells the embedder whether this text is a *stored document* or a *question*. Getting it wrong silently costs accuracy. |
| **recall@k** | Was a correct document in the top *k*? *"Did we find it at all?"* |
| **MRR** | *Mean Reciprocal Rank* — how **high** it ranked. 1st = 1.0, 2nd = 0.5, 3rd = 0.33. |
| **nDCG** | Like MRR, but handles questions with several correct documents. |
| **answer_support** | Did a retrieved chunk actually contain the answer **string**? The metric that catches "right document, wrong chunk". |
| **Hallucination** | The model confidently stating something the sources don't support. |
| **Matryoshka (MRL)** | An embedding trained so you can **truncate** it (3072 → 1536 → 768) and it still works. Like nesting dolls. |

---

# PART B — Cell-by-cell walkthrough

---

## 🔹 Cell 0 (markdown) — Why v2 exists

**What it does:** The cover page. It opens with a **post-mortem** rather than a feature list.

You asked v1.1: *"What was the Customer Satisfaction level for supplier performance review?"*
and got **"Overall district satisfaction was 8.5/10"** — but the document says **"Customer
satisfaction: 4.8/5 stars"** for Acme. `8.5/10` is the district-wide average: a completely
different number, for a completely different thing.

**Why it's needed:** it names **four independent defects**, each of which gets *worse* on real
data. Understanding them is what makes the rest of the notebook make sense.

| # | Defect in v1.1 | Consequence |
|---|---|---|
| 1 | **Chunks stored naked** — only `doc_id, chunk_index, content` were saved. All the metadata (supplier, category, date) was discarded before embedding. | A chunk reading *"Customer satisfaction: 4.8/5 stars"* has **no supplier attached**. Neither retrieval nor the LLM can tell whose score it is. |
| 2 | **Pure dense retrieval, no keyword leg.** | Exact tokens — `BID-2024-089`, `4.8/5`, `Net-45` — are precisely where embeddings are weakest. ID lookups fail *silently*. |
| 3 | **No disambiguation or citation rules in the prompt.** | The document holds **two** satisfaction numbers. The model picked one, unlabelled and uncited, and you had **no way to see** it chose wrong. |
| 4 | **No reranking, `TOP_K = 3`.** | Whatever the first-stage search returned **was** the answer. One weak embedding = one wrong answer. |

**Two further deploy-time problems it calls out:**

- **The v1.1 leaderboard measured noise.** Its latency column included the ~3.2 second Vertex
  embedding round-trip, which utterly swamped the actual Postgres search (sub-millisecond on 20
  rows). Every index scored ~3.2 s → no signal → the leaderboard crowned **`none`
  (brute-force sequential scan)** the winner. Ship that and every query scans the whole table.
  - *Root cause:* `gemini-embedding-001` emits 3072 dims; pgvector's HNSW/IVFFlat cap at
    **2000**. So every *indexed* combo using the best model was skipped, leaving only the
    unindexed one standing. v2 fixes it by truncating to 1536 (Cell 6).
- **A wrong gold label.** `"Which supplier offered an early payment discount?" → globex-dairy-2024`
  — but **Sysco (1%) and US Foods (2%) offer one too**. The benchmark was scoring **correct**
  retrievals as failures.

**The eight things v2 does instead** (each maps to cells you'll meet):

| # | Fix | Cells | Fixes defect |
|---|---|---|---|
| 1 | **Contextual chunk headers** | 13–14 | 1 |
| 2 | **Hybrid retrieval + RRF** | 22–23 | 2 |
| 3 | **Reranking** | 24–25 | 4 |
| 4 | **Grounded prompt with citations** | 26–27 | 3 |
| 5 | **Real schema + incremental re-index** | 9–10, 20–21 | cost |
| 6 | **1536-dim Matryoshka truncation** | 5–6 | index ceiling |
| 7 | **Honest evaluation** | 31–35 | measurement |
| 8 | **Ops hygiene** (pooling, retries, secrets) | 6, 8, 12 | reliability |

> 🔒 **Security — action required.** v1.1 carries a **live Postgres password in cleartext**. If
> that file was ever shared or committed, **rotate that credential now.** v2 reads it from
> `$PGPASSWORD`. This is not theoretical — say it out loud before someone else finds it.

---

## 🔹 Cell 1 (markdown) — "New to RAG? Read this first"

**What it does:** A 5-minute primer plus the vocabulary table (reproduced as section A.5 above).

**Why it's needed:** every later cell assumes these words. If "tsvector" or "RRF" is unfamiliar,
Cell 23's SQL is unreadable.

---

## 🔹 Cell 2 (markdown) — "Why ONE chunker and ONE model at serving time"

**This is the most important non-code cell in the notebook.** It resolves the confusion v1.1
created and it is your script for the client meeting.

### The core distinction

| | **Selection** (choosing) | **Serving** (running) |
|---|---|---|
| What it is | An experiment measuring which config wins | The product answering questions |
| How often | **Once**, offline, re-run when something changes | On **every user request** |
| How many pipelines | **As many as you can afford to test** | **Exactly one** — the winner |
| Lives in | **Cell 36–39 (Step 13)** | Cells 4–35 |

**So: yes, absolutely test many chunkers and models — in the sweep. Then deploy one.** Serving
27 pipelines is not something anyone does. It would multiply storage, cost and latency by 27 to
produce 27 different answers to the same question, with no way to choose between them at request
time.

### "Why didn't you use different chunking methods?"

**The right answer is "I did — here are the numbers."** That's exactly what Step 13 produces: a
table comparing all three of v1.1's chunkers *plus* the recursive splitter, on **your** data with
**your** questions.

Two weaker answers to avoid:
- *"I tested 27 combos"* → then why is only one deployed?
- *"I picked a sensible default"* → on what basis?

### "Why `gemini-embedding-001` and not gecko@003 / 004 / 005?"

First, **get the lineage right** so you sound informed. These are **generations, not options**:

```
textembedding-gecko@001/@002/@003  →  text-embedding-004/005  →  gemini-embedding-001
       (legacy, 768d)                      (768d)                  (3072d, current)
```

Picking `gecko@003` over `gemini-embedding-001` is like specifying an older model year — you'd
need a *reason*, such as an existing index you can't afford to re-embed. **Check the Vertex AI
model lifecycle page** for deprecation/retirement dates before quoting any of them; "it gets
retired in N months" is itself a decisive argument.

Then give three reasons **in this order** — note only the first is really about your data:

1. **Measured.** It won your own benchmark: MRR **0.90** vs **0.70** for `text-embedding-004/005`.
   *(v1.1's **model** comparison was sound. Its **index** comparison measured noise — quote the
   first half only. Knowing which half of your own result to disown is what credibility looks
   like.)*
2. **Published benchmarks.** Near the top of **MTEB**, the standard public embedding leaderboard.
   Supporting evidence, **never a substitute** for testing on your own data — MTEB is general
   web/academic text, not procurement documents.
3. **Operational.** Current generation (longest support runway), multilingual, and **Matryoshka
   (MRL)** — you can truncate 3072 → 1536 → 768 and trade accuracy for storage **without
   re-embedding**. The older models cannot do that.

### ⚠️ The question that will actually catch you out

> *"How many questions did you evaluate on?"*

Be honest about what small numbers can and cannot support:

- With **13 questions**, one question flipping moves recall by **~8 percentage points**. With 5,
  it's **20 points**. The real gap between two decent configs is often **2–5 points — smaller
  than your measurement noise.**
- That is exactly why v1.1's leaderboard was full of ties (`0.80, 0.80, 0.80…`). It wasn't
  measuring that the configs were equal; it **lacked the power to tell them apart.**
- **Rule of thumb: ~50 questions minimum to rank configurations, 100+ to trust a small gap.**
  They aren't hard to write — pull them from real user logs, or have two colleagues write 25 each.

**What to say before you have that:**

> *"Current results are directional, from a 13-question pilot set on 6 documents. I'm expanding
> to 50+ questions from real queries before we lock the configuration."*

That is a professional answer. Presenting 5-question results as conclusive is the thing that
damages credibility.

### The one hard constraint behind all of it

> **Vectors from different embedding models are not comparable.** `text-embedding-004` and
> `gemini-embedding-001` place "romaine lettuce" at completely different coordinates. Comparing
> one to the other returns a meaningless number — **not an error, just silent garbage.**

Consequences you must internalise:
- **One table = one model, permanently.** (That's why v1.1 built a separate table per model — it
  had no choice — and why the sweep does the same for its throwaway tables.)
- **Changing `EMBED_MODEL` or `EMBED_DIM` invalidates every stored vector.** You must re-embed
  the whole corpus with `ingest(force=True)`. It is a **migration, not a config tweak** — which
  is precisely why you run the sweep *before* you commit, not after.

---

## 🔹 Cells 3–4 — Step 1: Install

```python
%pip install -q google-genai "psycopg2-binary>=2.9" tiktoken
print("Installed. Restart the kernel if pip asked you to.")
```

**What it does:** installs three libraries.

| Package | Role | Used in |
|---|---|---|
| `google-genai` | Vertex AI SDK — embeddings **and** Gemini chat | Cells 12, 25, 27, 35 |
| `psycopg2-binary>=2.9` | PostgreSQL driver. `-binary` = pre-compiled (no C compiler needed on Windows). `>=2.9` because the pool + context-manager behaviour relies on it. | Cell 8 onward |
| `tiktoken` | Tokenizer, for **counting** tokens to budget chunk sizes | Cells 12, 14 |

**Note what's gone:** v1.1 also installed `pandas`. v2 doesn't — it wasn't used. Fewer
dependencies = fewer version conflicts.

**Why `%pip` and not `!pip`:** `%pip` is a Jupyter magic that installs into the **kernel that is
actually running**. `!pip` shells out and can hit a *different* Python — the classic "I installed
it but it says ModuleNotFoundError" trap.

---

## 🔹 Cell 5 (markdown) — Step 2 intro: credentials & `EMBED_DIM`

Two points, both important.

**1. Credentials come from the environment, not the file.**

```bash
export PGHOST=10.151.179.4 PGDATABASE=postgres PGUSER=postgres PGPASSWORD='...'
export GCP_PROJECT_ID=div-aais-rfpiq-usc1-uat GCP_LOCATION=us-central1
```

**2. Why `EMBED_DIM = 1536`** — this is the fix for v1.1's worst structural bug:

- `gemini-embedding-001` emits **3072** dims natively.
- pgvector's `hnsw`/`ivfflat` refuse anything over **2000** dims.
- So in v1.1, **every indexed combination using the best model was skipped**, leaving only the
  unindexed one → the leaderboard "chose" brute force.
- The model is **Matryoshka-trained**: truncating to 1536 (and **re-normalising**, which Cell 12
  does) retains essentially all retrieval quality *and* **halves storage**.
- **That buys HNSW back.**

You could go to 768 to halve it again — but **benchmark it with Step 11 first**, don't assume.

---

## 🔹 Cell 6 (code) — Step 2: Configuration

The only cell most people edit. Everything downstream reads from it.

### Block 1 — Vertex AI

```python
import os

PROJECT_ID   = os.environ.get("GCP_PROJECT_ID", "div-aais-rfpiq-usc1-uat")
LOCATION     = os.environ.get("GCP_LOCATION", "us-central1")

EMBED_MODEL  = "gemini-embedding-001"
EMBED_DIM    = 1536
CHAT_MODEL   = "gemini-2.5-flash"
RERANK_MODEL = "gemini-2.5-flash"
```

**`os.environ.get("NAME", fallback)`** reads an environment variable, falling back to the second
argument if unset. That pattern is what lets you override *anything* without editing code — the
same notebook runs against dev and prod by changing shell variables.

- **`LOCATION`** matters more than it looks: **models are enabled per-region.** If a model gets
  skipped later, wrong region is suspect #1.
- **`CHAT_MODEL = gemini-2.5-flash`** — the fast/cheap tier, and the right call for RAG: the hard
  work (finding the right text) is already done; the model only reads 4 paragraphs and reports
  them. A bigger model buys less than you'd think.
- **`RERANK_MODEL`** — same model, different job. In production you'd swap this for the **Vertex
  AI Ranking API** (see Cell 24).

### Block 2 — the password, done properly

```python
_pw = os.environ.get("PGPASSWORD")
if not _pw:
    import getpass
    _pw = getpass.getpass("Postgres password: ")
```

**Why:** v1.1 hardcoded the password, and **that credential now exists in every copy of that
file** — every laptop, every email, every git history. This version reads the environment and, if
unset, **prompts interactively** rather than crashing.

**`getpass.getpass()` not `input()`** — `getpass` hides what you type, so the password doesn't
end up echoed into the notebook output and saved to the `.ipynb` file.

### Block 3 — connection settings

```python
DB_CONFIG = {
    "host":     os.environ.get("PGHOST", "10.151.179.4"),
    "port":     int(os.environ.get("PGPORT", 5432)),
    "dbname":   os.environ.get("PGDATABASE", "postgres"),
    "user":     os.environ.get("PGUSER", "postgres"),
    "password": _pw,
    "connect_timeout": 10,
    "application_name": "rag_v2",
}
```

- **`int(...)`** — environment variables are **always strings**. Without the cast, `port` would
  be `"5432"` and psycopg2 would complain.
- **`connect_timeout: 10`** — give up after 10 seconds instead of hanging forever on a bad
  network route. v1.1 had no timeout; a misconfigured VPC route meant an infinite hang.
- **`application_name: "rag_v2"`** — shows up in `pg_stat_activity`. When the DBA asks "what's
  running this query?", your queries are labelled. Free observability.

```python
DOC_TABLE   = "rag_documents"
CHUNK_TABLE = "rag_chunks"
```

**Two tables, permanent.** Not 27 throwaway ones. (Contrast v1.1.)

### Block 4 — chunking

```python
CHUNK_TOKENS         = 350
CHUNK_OVERLAP_TOKENS = 60
MAX_EMBED_TOKENS     = 2000
```

- **`CHUNK_TOKENS = 350`** (~260 words / ~1400 characters). The trade-off:
  - **Too big** → the relevant sentence gets buried among nine irrelevant ones, and the chunk's
    single vector becomes a blurry average of everything in it.
  - **Too small** → facts get split apart and lose their context. *"3.20 USD/case"* — for what?
    From whom?
  - **250–500 is the practical sweet spot** for question-answering over prose.
- **`CHUNK_OVERLAP_TOKENS = 60`** (~15%). Each chunk repeats the last ~60 tokens of the previous
  one, so a fact sitting exactly on a boundary still appears **whole** in at least one chunk.
  Cheap insurance — 15% extra storage to avoid losing facts.
- **`MAX_EMBED_TOKENS = 2000`** — hard safety cap per embedding call (the model accepts 2048).
  One oversized input would otherwise **400 the entire batch of 16.**

### Block 5 — retrieval

```python
VECTOR_CANDIDATES   = 40
TEXT_CANDIDATES     = 40
RRF_K               = 60
W_VECTOR, W_TEXT    = 1.0, 1.0
TOP_K               = 8
RERANK_CANDIDATES   = 20
RERANK_TOP_N        = 4
USE_RERANKER        = True
MAX_COSINE_DISTANCE = 0.80
```

**The shape to understand: cast a WIDE net cheaply, then narrow it carefully.**

```
40 vector candidates  ┐
                      ├─▶ RRF fuse ─▶ 20 ─▶ rerank ─▶ 4 ─▶ Gemini
40 keyword candidates ┘
```

- **`VECTOR_CANDIDATES` / `TEXT_CANDIDATES = 40`** — stage 1 optimises for **recall** (don't miss
  it), not precision. 40 is cheap; missing the answer is not recoverable later.
- **`RRF_K = 60`** — the constant in `weight / (k + rank)`. Bigger `k` = flatter weighting (rank
  1 and rank 5 treated more equally). **60 is the published default** from the original RRF
  paper — use it unless you have a measured reason not to.
- **`W_VECTOR, W_TEXT = 1.0, 1.0`** — equal trust. **Raise `W_TEXT`** if your users search by
  exact IDs/SKUs; **raise `W_VECTOR`** for conceptual questions. A genuinely useful, cheap dial.
- **`RERANK_TOP_N = 4`** — how many the LLM finally sees. **Small on purpose:** irrelevant
  passages in the prompt **measurably make answers worse.** More context ≠ better answers.
- **`MAX_COSINE_DISTANCE = 0.80`** — the **relevance floor**. If the closest chunk is further
  than this **and** no keyword matched, refuse rather than answer from unrelated documents. v1.1
  had no such check — it always returned its top 3, however irrelevant. That is exactly how RAG
  systems end up confidently answering out of the wrong document.

### Block 6 — HNSW tuning

```python
HNSW_M                = 16
HNSW_EF_CONSTRUCTION  = 64
HNSW_EF_SEARCH        = 100
```

| Knob | What it controls | Cost of changing |
|---|---|---|
| `m = 16` | Graph connections per node. Higher = better recall, bigger index, slower build. | **Index rebuild** |
| `ef_construction = 64` | How hard the index works while **building**. | **Index rebuild** |
| `ef_search = 100` | How hard it works while **searching**. | **Free — no rebuild** |

**`ef_search` is the one you actually tune.** It trades latency directly for recall and costs
nothing to change. Start at 100, raise until recall plateaus. The other two only get touched
after you've exhausted everything else.

---

## 🔹 Cell 7 (markdown) — Step 3 intro: why pool

v1.1 opened a **brand-new TCP + TLS + auth handshake for every single query.** Under load that
alone costs tens of milliseconds per call and will exhaust Cloud SQL's connection limit.

---

## 🔹 Cell 8 (code) — Step 3: Connection pool

```python
import contextlib
import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

POOL = ThreadedConnectionPool(minconn=1, maxconn=8, **DB_CONFIG)
```

**What a pool is:** it opens a few connections up front and **lends them out**, instead of doing
a fresh handshake per query.

- **`minconn=1`** — always keep at least one alive and ready.
- **`maxconn=8`** — never open more than 8. Cloud SQL has a **global** limit; if you run 4 app
  instances at `maxconn=8` each, that's 32 connections against that limit. Size accordingly.

```python
@contextlib.contextmanager
def db(dict_rows: bool = False):
    conn = POOL.getconn()
    try:
        factory = psycopg2.extras.RealDictCursor if dict_rows else None
        with conn:
            with conn.cursor(cursor_factory=factory) as cur:
                yield cur
    finally:
        POOL.putconn(conn)
```

**`@contextlib.contextmanager`** turns this function into something usable as `with db() as cur:`.
Everything **before** `yield` runs on entering the block; everything **after** runs on exit —
**including when an exception was raised.** That's what guarantees cleanup.

**`with conn:` is psycopg2's TRANSACTION block, not a close-the-connection block.** This trips up
almost everyone:
- Leaving it **normally** → `COMMIT`
- Leaving it **via an exception** → `ROLLBACK`

So a failed multi-step write can **never** leave the database half-updated. This is what makes
Cell 21's ingest crash-safe.

**`with conn.cursor(...)`** closes the cursor and frees its server-side resources.

**`finally: POOL.putconn(conn)`** — runs no matter what: success, exception, even an early
`return`. **Without this line, a crash would permanently lose a connection from the pool** (a
"connection leak"), and after 8 crashes the app would hang forever waiting for one to come back.
This one line is the difference between a service that survives errors and one that dies slowly.

**`dict_rows`** — `RealDictCursor` returns rows as dicts (`row["doc_id"]`) instead of tuples
(`row[0]`). Readability, not performance.

```python
with db() as cur:
    cur.execute("SELECT current_database(), version()")
    dbname, version = cur.fetchone()
print(f"Connected to '{dbname}' - {version.split(',')[0]}")
```

**Fail-fast checkpoint.** Prove the database is reachable **before** any expensive step runs.
Without it you'd discover a connection problem halfway through Cell 21, after paying for hundreds
of embedding calls.

---

## 🔹 Cell 9 (markdown) — Step 4 intro: the schema design

Two tables, **not** one-table-per-experiment:

- **`rag_documents`** — raw source text + JSONB metadata + `content_hash` (what the document *is*
  now) and `indexed_hash` (what we last embedded). **Comparing the two is how Cell 21 re-embeds
  only what changed.**
- **`rag_chunks`** — one row per chunk, with the crucial `content` vs `embed_input` split.

---

## 🔹 Cell 10 (code) — Step 4: Schema (DDL)

```python
DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
...
"""
```

> **f-string gotcha:** because this is an f-string, any brace you want to keep **literal** in the
> SQL must be **doubled**: `'{{}}'::jsonb` renders as `'{}'::jsonb`. Miss that and you get a
> confusing `KeyError`.

### `CREATE EXTENSION IF NOT EXISTS vector;`

Vanilla Postgres has no idea what a vector is. This adds:
1. The **`vector(n)` column type**
2. **Distance operators** — `<=>` cosine (what we use), `<->` Euclidean, `<#>` inner product
3. **Index access methods** `hnsw` and `ivfflat`

**Why cosine specifically?** Cosine compares the **direction** of two vectors, ignoring length.
For text, direction ≈ topic and length ≈ verbosity. You want "these two passages are about the
same thing" to score well regardless of one being longer. Standard for text retrieval.

### Table 1 — `rag_documents`

```sql
CREATE TABLE IF NOT EXISTS rag_documents (
    doc_id        text PRIMARY KEY,
    title         text NOT NULL,
    content       text NOT NULL,
    metadata      jsonb NOT NULL DEFAULT '{}'::jsonb,
    content_hash  text NOT NULL,
    indexed_hash  text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);
```

**`metadata jsonb`** — one flexible column instead of ten fixed ones.
- **`jsonb` not `json`** — the binary form: indexable and faster to query.
- **Why one column:** adding a new field later (`region`, `contract_id`, `confidentiality`) needs
  **no schema migration**. On a table with millions of rows, `ALTER TABLE ADD COLUMN` is a real
  operational event; adding a JSON key is not.
- v1.1 used 10 fixed columns. Every new facet would have been a migration.

**`content_hash` vs `indexed_hash` — the incremental-indexing trick.** This is the cleverest bit
of the schema:

| Column | Meaning |
|---|---|
| `content_hash` | Fingerprint of what this document says **right now** |
| `indexed_hash` | Fingerprint of what we last **embedded**. `NULL` = never indexed. |

```
content_hash != indexed_hash  ⟹  needs re-embedding
content_hash == indexed_hash  ⟹  skip it, we're up to date
```

That one comparison is what turns "re-embed 10,000 documents every run" into "re-embed the 3 that
changed."

**`timestamptz` not `timestamp`** — with timezone. A global supply chain spans timezones; a naive
timestamp is ambiguous.

### Table 2 — `rag_chunks`

```sql
CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id     bigserial PRIMARY KEY,
    doc_id       text NOT NULL REFERENCES rag_documents(doc_id) ON DELETE CASCADE,
    chunk_index  int  NOT NULL,
    content      text NOT NULL,
    embed_input  text NOT NULL,
    token_count  int  NOT NULL,
    metadata     jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding    vector(1536) NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(embed_input, ''))) STORED,
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (doc_id, chunk_index)
);
```

#### ⭐ `content` vs `embed_input` — the single most important design decision in v2

| Column | Contains | Used for |
|---|---|---|
| `content` | **Clean text only** | Shown to the LLM and the user |
| `embed_input` | **Contextual header + clean text** | What we **embedded** and what **keyword search** reads |

**Why they must be separate:**

The header makes a chunk findable by its supplier name **even when the chunk body never mentions
it** — and simultaneously keeps the ugly metadata banner **out of the answer** the user sees.

Concretely, this is **the fix for your satisfaction bug**:

```
embed_input:  "Document: Q1 2024 Supplier Performance Review | ID: performance-review-2024 |
               Supplier Name: Riverside USD | Category: Performance Review | Bid Date: 2024-04-01

               ...Acme Foods maintained 98% on-time delivery rate, with quality score
               of 9.2/10. Customer satisfaction: 4.8/5 stars..."

content:      "...Acme Foods maintained 98% on-time delivery rate, with quality score
               of 9.2/10. Customer satisfaction: 4.8/5 stars..."
```

If you embedded only `content`, the vector for "Customer satisfaction: 4.8/5 stars" has **no
connection to "performance review"** — which is literally what you asked about. That's defect #1.

#### `ON DELETE CASCADE` — no orphaned vectors

```sql
doc_id text NOT NULL REFERENCES rag_documents(doc_id) ON DELETE CASCADE
```

- **`REFERENCES`** — this value must match a real document. The database enforces it.
- **`ON DELETE CASCADE`** — deleting a document **automatically deletes its chunks**.

**Why it matters:** without it, deleting a document leaves its vectors behind, and they keep
showing up in search results **forever**. Your system would confidently cite a document that no
longer exists. This is a genuinely nasty class of bug that the database can prevent for you.

#### `tsv` — the GENERATED column

```sql
tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(embed_input, ''))) STORED
```

**What `to_tsvector` does:** strips punctuation, lowercases, and **stems** words —
`'deliveries' → 'deliveri'` — so searching *"delivery"* also matches *"deliveries"*.

**Why GENERATED:** Postgres computes and maintains it automatically. You **never write to it**,
and it **can never drift out of sync** with the text. The alternative (a trigger, or updating it
in application code) is a bug waiting to happen — someone eventually updates `embed_input`
without updating `tsv`, and keyword search silently returns stale results.

**`coalesce(x, '')`** guards against `NULL` (which would make the whole tsvector `NULL`).

**Note it indexes `embed_input`, not `content`** — so keyword search also benefits from the
header. Searching "Riverside" finds the chunk even though the chunk body never says it.

#### `UNIQUE (doc_id, chunk_index)`

One row per (document, position). Makes re-ingest safe — you can't accidentally end up with two
"chunk 0" rows for the same document.

### The three indexes

```sql
CREATE INDEX ... USING gin (tsv);                        -- keyword search
CREATE INDEX ... USING gin (metadata jsonb_path_ops);    -- metadata filters
CREATE INDEX ... (doc_id);                               -- B-tree, for re-ingest DELETE
```

- **GIN** (*Generalised Inverted Index*) is the right type for **search-inside-a-value** columns.
  It maps each word → the rows containing it, exactly like the index at the back of a book.
- **`jsonb_path_ops`** is a **smaller, faster** GIN variant that supports the `@>` ("contains")
  operator — which is precisely what our metadata filters use. It doesn't support other jsonb
  operators, but we don't need them, so we get the speed for free.
- The **B-tree on `doc_id`** makes `DELETE ... WHERE doc_id = ...` during re-ingest fast.

### What is deliberately NOT here

```python
# NOTE: the HNSW *vector* index is deliberately NOT created here - see Step 7c.
```

**Why:** building HNSW on an **empty** table and then inserting means the index is maintained
row-by-row — markedly slower. **Bulk-load first, index once.** Standard database practice, and
it's why `ensure_vector_index()` lives in Cell 21 instead.

---

## 🔹 Cell 11 (markdown) — Step 5 intro

Four things v1.1's `embed_texts` lacked: **retry with backoff**, **L2 normalisation**, **input
truncation**, and a **query cache**. Each is explained at the code below.

---

## 🔹 Cell 12 (code) — Step 5: Embeddings, batched / retried / normalised

### Setup

```python
import math, random, time
from functools import lru_cache
import tiktoken
from google import genai
from google.genai.types import EmbedContentConfig

genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
_enc = tiktoken.get_encoding("cl100k_base")
```

**`vertexai=True`** routes through **Vertex AI** (enterprise: IAM auth, VPC controls, data
residency) rather than the consumer Gemini API (API-key auth). For a Sysco deployment this is the
correct and defensible choice. Auth comes from **Application Default Credentials** — run
`gcloud auth application-default login` once.

**`cl100k_base`** is GPT-3.5/4's tokenizer, and it does **not** match Vertex's. That's deliberate:
we only need a **consistent yardstick** for budgeting chunk sizes, not an exact match. Say this
before someone "catches" it — it's a considered choice, not an oversight.

### Helpers

```python
def n_tokens(text): return len(_enc.encode(text))

def _truncate(text, max_tokens=MAX_EMBED_TOKENS):
    toks = _enc.encode(text)
    return text if len(toks) <= max_tokens else _enc.decode(toks[:max_tokens])
```

**Why truncate:** one oversized input would otherwise **fail the entire batch of 16.** Silently
degrading one document beats losing sixteen.

```python
def _l2_normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec
```

**What it does:** scales a vector so its length is exactly 1.0, **without changing its
direction** — Pythagoras in 1536 dimensions.

**Why it's needed:** cosine distance only cares about direction (the angle), not length. But
**truncating a 3072-dim MRL embedding to 1536 leaves it slightly shorter than 1.0**, because you
threw away some of its components. Re-normalising keeps `<=>` numerically well-behaved and makes
distances comparable across models. `if norm` guards against divide-by-zero.

### Retry with exponential backoff + jitter

```python
_TRANSIENT = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL", "DEADLINE")

def _with_retry(fn, attempts=5, base=1.0):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1 or not any(s in str(e) for s in _TRANSIENT):
                raise
            time.sleep(base * (2 ** i) + random.random())
```

**`fn` is a function, not a value.** Callers pass `lambda: something()` so we can **re-invoke**
it. If you passed the *result*, there'd be nothing to retry.

**The two-condition give-up test is the important part:**

```python
if i == attempts - 1 or not any(s in str(e) for s in _TRANSIENT):
    raise
```

1. **Out of attempts** → give up.
2. **This error will never fix itself** → give up **immediately**.

A `403 PERMISSION_DENIED` or a typo'd model name is **permanent**. Retrying it 5 times wastes
15 seconds and **hides the real problem** behind a wall of retry noise. Only retry things that
might succeed on a second try.

> This is the direct fix for v1.1's bare `except Exception`, which retried *everything* 50 times.

**`time.sleep(base * (2 ** i) + random.random())`:**
- **`2 ** i`** → waits of 1s, 2s, 4s, 8s — **exponential backoff**. Gives an overloaded server
  progressively more room.
- **`+ random.random()`** → **jitter**. Without it, many parallel workers that all failed at the
  same instant would all retry at the **same instant**, hammering the server in lockstep and
  causing the very overload they're backing off from. The random fraction spreads them out.

### The main function

```python
def embed_texts(texts, task_type, model=EMBED_MODEL, dim=EMBED_DIM, batch_size=16):
    cfg = EmbedContentConfig(task_type=task_type, output_dimensionality=dim)
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [_truncate(t) for t in texts[i:i + batch_size]]
        try:
            resp = _with_retry(lambda: genai_client.models.embed_content(
                model=model, contents=batch, config=cfg))
            out.extend(e.values for e in resp.embeddings)
        except Exception:
            for t in batch:
                resp = _with_retry(lambda t=t: genai_client.models.embed_content(
                    model=model, contents=[t], config=cfg))
                out.append(resp.embeddings[0].values)
    return [_l2_normalize(v) for v in out]
```

**`output_dimensionality=dim`** — this is where the Matryoshka truncation actually happens. You
ask the API for 1536 dims and it returns them.

**`task_type` — the free accuracy upgrade:**

| Value | Use when | Passed in |
|---|---|---|
| `RETRIEVAL_DOCUMENT` | The text is a **thing to be found** | Cell 21 (ingest) |
| `RETRIEVAL_QUERY` | The text is a **question doing the finding** | Cells 12, 23 (search) |

**Why it exists:** documents and questions look nothing alike.
- Document: *"Romaine lettuce: 3.20 USD/case. Spinach: 4.10 USD/case."*
- Question: *"What did Acme quote for romaine?"*

One is a declarative price list; the other is an interrogative sentence with a question mark. A
naive embedding puts them further apart than they should be. `task_type` tells the model to
project them into a **shared space** where an answer and its question land close together.
**Costs nothing, measurably improves recall.** Getting it wrong produces **no error, just worse
results** — the most dangerous kind of bug.

**Batching (16 per call):** each API call has fixed overhead (~100–300 ms of network + auth). With
500 chunks, one-at-a-time is 500 round trips ≈ a minute of pure waiting. In batches of 16 it's 32.

> **⚠️ The `lambda t=t:` detail — a genuine Python trap worth knowing.**
> ```python
> lambda t=t: ...   # ✅ captures the CURRENT t
> lambda: ...       # ❌ every lambda shares the loop variable
> ```
> Python closures capture the **variable**, not its value. Without `t=t`, all the lambdas would
> see whatever `t` is when they're finally *called* — the last item — and you'd embed the last
> text repeatedly. A classic bug that produces plausible-looking garbage.

**The batch → single fallback:** some model/quota combinations reject batches. Rather than
failing, retry that block one text at a time. Slow but correct — **graceful degradation.**

### Query embedding + cache

```python
def embed_query(question):
    return embed_texts([question], "RETRIEVAL_QUERY")[0]

@lru_cache(maxsize=1024)
def embed_query_cached(question):
    return tuple(embed_query(question))
```

**`@lru_cache(maxsize=1024)`** remembers the last 1024 questions. Same question → **instant, no
API call.** Saves ~200 ms on any repeated question, which in a real app is most of them.

**Why it returns a `tuple`, not a `list`:** `@lru_cache` requires everything it stores to be
**hashable** (immutable), and lists are not. Callers convert back with `list(...)`.

> **Note the discipline here:** Cell 33's benchmark deliberately calls the **uncached**
> `embed_query()`, so its latency numbers reflect a real **cold** request rather than a cache hit.
> Benchmarking your own cache is how you produce numbers that look great and mean nothing.

### Smoke test

```python
_probe = embed_query("hello")
assert len(_probe) == EMBED_DIM, f"expected {EMBED_DIM} dims, got {len(_probe)}"
print(f"Vertex OK - {EMBED_MODEL} -> {len(_probe)} dims, |v| = {math.sqrt(sum(x*x for x in _probe)):.4f}")
```

The **`assert`** is a real guard: if the model silently returns a different dimension, every
subsequent `INSERT` would fail with a confusing type error hundreds of lines later. Fail here
instead. The printed `|v|` should be `1.0000` — proof that normalisation worked.

---

## 🔹 Cell 13 (markdown) — Step 6 intro: chunking + headers

**Chunking.** v1.1's three chunkers were all naive: `fixed_400` slices mid-word, `token_256`
slices mid-sentence. Production uses a **recursive** strategy — respect the largest natural
boundary that fits, and fall back to a harder cut only when you must:

> paragraph → sentence → hard token split, then pack greedily to a token budget with
> sentence-aligned overlap.

**Contextual headers — the actual fix for your bug.** Before embedding, each chunk is prefixed
with its document identity:

```
Document: Q1 2024 Supplier Performance Review | ID: performance-review-2024 |
Supplier Name: Riverside USD | Category: Performance Review | Bid Date: 2024-04-01

Q1 2024 Performance Review Summary: Acme Foods maintained 98% on-time delivery ...
```

A chunk is now retrievable by *"performance review"*, *"Riverside"* or *"Q1 2024"* **even when
the chunk body never repeats those words**, and the LLM can see which document each fact came
from.

**Anthropic's published *contextual retrieval* work measures roughly a 35% reduction in retrieval
failures from this idea alone.** (Their full version generates a per-chunk LLM summary — the
obvious upgrade when budget allows; **the header form used here is free.**)

> **📌 Honest note about this demo corpus.** All six sample documents are 70–140 tokens, so at
> `CHUNK_TOKENS = 350` **each becomes exactly one chunk and the splitter never actually fires.**
> That's deliberate and worth understanding: on *this* data the satisfaction bug is fixed by the
> **header, hybrid retrieval and the prompt rules** — not by lucky chunk boundaries. The splitter
> matters once you point Cell 19 at real multi-page RFPs. **Re-run Step 11 after you do**, because
> that's when `CHUNK_TOKENS` starts to matter.
>
> Volunteering this is exactly the kind of thing that builds credibility.

---

## 🔹 Cell 14 (code) — Step 6: The splitter + header builder

### The two regexes

```python
_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
```

**`_PARA`** — a blank line = a paragraph break. `\n` newline, `\s*` any whitespace (possibly
none), `\n` newline.

**`_SENT`** — read in three parts:

| Part | Meaning |
|---|---|
| `(?<=[.!?])` | **lookbehind** — the character *before* this point must be `.` `!` or `?` |
| `\s+` | the whitespace actually removed by the split |
| `(?=[A-Z0-9("'])` | **lookahead** — what *follows* must start like a new sentence |

**Why lookaround instead of plain matching:** lookbehind/lookahead **assert** without
**consuming**, so the punctuation stays attached to the sentence.

**The lookahead is what stops it splitting `"3.20 USD"`** — after the `.` comes `2`… wait, `2` is
in `[A-Z0-9]`. So this is a **heuristic, not a parser.** It handles the common cases and can be
fooled by decimals and abbreviations ("U.S. Foods"). Good enough here; production systems reach
for `spaCy` or `nltk`. Know the limitation.

### `_hard_split` — the last resort

```python
def _hard_split(unit, max_tokens):
    toks = _enc.encode(unit)
    return [_enc.decode(toks[i:i + max_tokens]) for i in range(0, len(toks), max_tokens)]
```

**Only reached when a single "sentence" is longer than the ENTIRE chunk budget** — e.g. a giant
table row, or a wall of text with no punctuation (very common in scraped PDFs). Ugly, but better
than emitting a chunk too big to embed.

### `split_text` — the recursive splitter

**Phase 1 — text → sentences:**

```python
units = []
for para in _PARA.split(text.strip()):
    para = para.strip()
    if not para: continue
    for sent in _SENT.split(para):
        sent = sent.strip()
        if not sent: continue
        units.extend(_hard_split(sent, max_tokens) if n_tokens(sent) > max_tokens else [sent])
```

Split on blank lines → then split each paragraph into sentences → and **only** if one sentence
still blows the budget, chop it. **That cascade is what "recursive splitting" means:** try the
biggest natural boundary first, fall back to a smaller one, cut mid-sentence only when there is
genuinely no alternative.

**Phase 2 — sentences → chunks (greedy packing with sentence-aligned overlap):**

```python
chunks, buf, buf_tokens = [], [], 0

for unit in units:
    t = n_tokens(unit)
    if buf and buf_tokens + t > max_tokens:
        chunks.append(" ".join(buf))

        keep, kept = [], 0
        for prev in reversed(buf):
            pt = n_tokens(prev)
            if kept + pt > overlap_tokens:
                break
            keep.insert(0, prev)
            kept += pt
        buf, buf_tokens = keep, kept

    buf.append(unit)
    buf_tokens += t

if buf:
    chunks.append(" ".join(buf))
```

- **`buf`** — sentences accumulated for the chunk being built.
- **`buf_tokens`** — a running count, so we don't re-tokenise the whole buffer every iteration.
- **`if buf and ...`** — the `buf and` prevents emitting an empty chunk on the very first pass.

**The overlap block is the clever part.** After flushing a chunk, we walk **backwards** through
its sentences, taking **whole sentences** until we hit the overlap budget, and those become the
**start of the next chunk**.

- **`reversed(buf)`** — walk from the end.
- **`keep.insert(0, prev)`** — inserting at position 0 rebuilds the **original order**.
- **Whole sentences only** — so the overlap never starts mid-thought.

**Why overlap at all — the concrete failure it prevents:**

```
...Customer satisfaction was │ 4.8/5 stars...
                              ↑ chunk boundary
```

Neither chunk contains the complete fact, and retrieval loses it. Repeating ~60 tokens means the
fact appears **intact in at least one chunk.**

### Contextual headers

```python
HEADER_FIELDS = ["supplier_name", "category", "document_type", "bid_date", "status"]

def contextual_header(doc_id, title, metadata):
    bits = [f"Document: {title}", f"ID: {doc_id}"]
    bits += [f"{f.replace('_', ' ').title()}: {metadata[f]}"
             for f in HEADER_FIELDS if metadata.get(f)]
    return " | ".join(bits)
```

**Why only five fields:** these are the ones users **search by**. Adding every field would
**dilute the embedding** — the header would grow to dominate the chunk and every chunk would
start looking alike. Deliberate selection, not laziness.

- **`.replace('_',' ').title()`** turns `supplier_name` → `Supplier Name`.
- **`if metadata.get(f)`** skips fields that are missing, `None`, or empty — no
  `Status: None` noise.

```python
def build_chunks(doc_id, title, content, metadata):
    header = contextual_header(doc_id, title, metadata)
    return [
        {
            "chunk_index": i,
            "content": piece,
            "embed_input": f"{header}\n\n{piece}",
            "token_count": n_tokens(piece),
        }
        for i, piece in enumerate(split_text(content))
    ]
```

**The same header goes on every chunk of a document.** And there it is again — the `content` /
`embed_input` split. **The header makes the chunk findable without ever appearing in the answer.**

### The demo

```python
_demo = build_chunks(
    "performance-review-2024", "Q1 2024 Supplier Performance Review",
    "Acme Foods maintained 98% on-time delivery. Customer satisfaction: 4.8/5 stars.",
    {"supplier_name": "Riverside USD", "category": "Performance Review", "bid_date": "2024-04-01"},
)
print(_demo[0]["embed_input"])
```

**Run this and read the output carefully.** Seeing the header physically attached to the
satisfaction sentence is the moment the whole fix clicks.

---

## 🔹 Cell 15 (markdown) — Step 7a intro

Same six documents as v1.1, but metadata now lands in **one JSONB column** instead of ten fixed
ones. JSONB means adding a facet (`region`, `contract_id`, `confidentiality`) never needs a
migration, and `metadata @> '{"category":"Dairy"}'` is served by the GIN index from Cell 10.

**`content_hash` covers title + content + metadata**, so *any* edit — **including a metadata-only
edit that changes the contextual header** — correctly marks the document for re-embedding. That
last clause is easy to get wrong and would leave stale headers embedded forever.

---

## 🔹 Cell 16 (code) — Step 7a: Upsert + delete

```python
import hashlib, json
from psycopg2.extras import execute_values

def content_hash(title, content, metadata):
    payload = json.dumps({"t": title, "c": content, "m": metadata},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

**What a hash is:** a fixed-length fingerprint of arbitrary data. Change one character anywhere
and the hash changes completely.

**`sort_keys=True` is essential.** Python dicts have no guaranteed ordering across
constructions — `{"a":1,"b":2}` and `{"b":2,"a":1}` are equal but serialise differently. Without
sorting, **the same document could hash differently on two runs** and get re-embedded forever.
That's a real, expensive bug this one keyword prevents.

**`ensure_ascii=False`** keeps non-ASCII characters as-is rather than escaping them, so the hash
is stable across encodings.

```python
def upsert_documents(doc_map):
    rows = []
    for doc_id, d in doc_map.items():
        title, content = d["title"], d["content"]
        meta = d.get("metadata", {})
        rows.append((doc_id, title, content, json.dumps(meta), content_hash(title, content, meta)))

    with db() as cur:
        execute_values(cur, f"""
            INSERT INTO {DOC_TABLE} (doc_id, title, content, metadata, content_hash) VALUES %s
            ON CONFLICT (doc_id) DO UPDATE SET
                title = EXCLUDED.title, content = EXCLUDED.content,
                metadata = EXCLUDED.metadata, content_hash = EXCLUDED.content_hash,
                updated_at = now()
        """, rows, template="(%s, %s, %s, %s::jsonb, %s)")
```

**`execute_values`** — psycopg2's bulk-insert helper. Expands one `VALUES %s` into
`VALUES (...), (...), (...)` and sends **one statement** instead of N. For 1000 rows, roughly
100× faster than a loop of `execute()`.

**`ON CONFLICT (doc_id) DO UPDATE`** — the **UPSERT**. If `doc_id` exists, update it instead of
erroring. `EXCLUDED` is Postgres's alias for "the row you tried to insert."

**Notice `indexed_hash` is NOT updated here.** That's deliberate and important: writing a document
updates `content_hash` but leaves `indexed_hash` alone, so the two now differ → Cell 21 correctly
sees it needs re-embedding. Only Cell 21, *after successfully storing the vectors*, advances
`indexed_hash`.

**`template="(%s, %s, %s, %s::jsonb, %s)"`** — the `::jsonb` cast tells Postgres to parse the JSON
string into a real jsonb value. Without it you get a type error.

```python
def delete_documents(doc_ids):
    with db() as cur:
        cur.execute(f"DELETE FROM {DOC_TABLE} WHERE doc_id = ANY(%s)", (doc_ids,))
```

**`ANY(%s)`** is the Postgres way to say `IN (list)` when the list is a **parameter**. You cannot
safely parameterise `IN (...)` directly; `ANY(array)` is the correct pattern.

**No need to delete chunks** — the `ON DELETE CASCADE` from Cell 10 handles it. That's the FK
earning its keep.

---

## 🔹 Cell 17 (code) — Step 7a: The six seed documents

Six realistic Sysco-shaped documents, each with `title`, `content`, and a `metadata` dict.

| doc_id | What it is | Why it's in the corpus |
|---|---|---|
| `acme-produce-2024` | Produce quote — romaine 3.20/case, Net-30 | Basic price lookup |
| `globex-dairy-2024` | Dairy quote — milk 1.85/gal, Net-45, **2% early-payment discount** | |
| `bid-summary-2024` | RFP **BID-2024-089** — 5/8 responded (**62.5%**), due 2024-05-25 | **Exact-ID lookup test** |
| `sysco-proteins-2024` | Proteins — salmon 12.80/lb, Net-60, **1% early-payment discount** | Multi-entity test |
| `us-foods-beverages-2024` | Beverages — OJ 2.30/gal, **2% early-payment discount** | Multi-entity test |
| `performance-review-2024` | Q1 scorecard — **customer satisfaction 4.8/5**, **district 8.5/10** | **The bug's home** |

**Three deliberate traps in this corpus** — these are not accidents:

1. **Two satisfaction numbers in one document** (`4.8/5` per-supplier, `8.5/10` district-wide).
   This is the exact ambiguity that broke v1.1. A good system surfaces **both, labelled.**
2. **Three suppliers offer an early-payment discount** (Globex 2%, Sysco 1%, US Foods 2%).
   v1.1's gold set labelled only one as correct and scored the other two **correct** retrievals
   as **failures**. Fixed in Cell 32.
3. **`bid_id: "BID-2024-089"` on every document.** An exact identifier that embeds terribly and
   keyword-matches perfectly — the hybrid-search proof case.

**`upsert_documents(SOURCE_DOCS)`** at the bottom writes them. **Safe to re-run** — it upserts,
and unchanged documents hash identically so Cell 21 skips them entirely.

> **This is where your real data goes.** Everything downstream is generic; this dict is the only
> place the demo corpus lives.

---

## 🔹 Cells 18–19 — Step 7b: Import from an existing table

```python
def import_from_existing_table(table, id_col, text_col, title_col=None,
                               metadata_cols=None, where=None, limit=None):
    metadata_cols = metadata_cols or []
    cols = [id_col, text_col] + ([title_col] if title_col else []) + metadata_cols
    sql = f"SELECT {', '.join(cols)} FROM {table}"
    if where:  sql += f" WHERE {where}"
    if limit:  sql += f" LIMIT {int(limit)}"
    ...
```

**Why this exists:** at Sysco, bid/RFP text almost certainly **already lives in some table.**
Retyping it would create a second copy that drifts out of sync. This is a small **adapter** —
one per source system — that maps an existing table into `rag_documents`.

**Note the design difference from v1.1's 7b:** v1.1 read the other table *directly* at query time.
v2 **copies it into `rag_documents`.** That's better because `rag_documents` carries the hash
columns needed for incremental indexing, and because you get one uniform shape regardless of how
many source systems you onboard.

**`metadata_cols` become JSONB keys**, so they're usable as **retrieval filters** *and* they
**appear in the contextual header** when they match `HEADER_FIELDS`. One parameter, two benefits.

- **`if not r[text_col]: continue`** — skip `NULL`/empty rows. Embedding an empty string wastes
  an API call and produces a meaningless vector that pollutes results.
- **`str(r[id_col])`** — coerce integer PKs to strings so ids compare consistently.
- **`int(limit)`** — the cast makes the f-string interpolation safe by guaranteeing a number.

**The cell only *defines* the function.** The example call is commented out — uncomment and point
it at your real table.

> ⚠️ **Caveat to flag before someone else does:** `table`, `where`, and the column names are
> f-string-interpolated into SQL. These are **your own config values, not user input**, so it's
> not a live injection hole — but if they ever become user-supplied, switch to
> `psycopg2.sql.Identifier`. Note the contrast with Cell 23, where the actual **query text** is
> properly parameterised with `%(name)s`.

---

## 🔹 Cell 20 (markdown) — Step 7c intro: incremental ingest

v1.1 **dropped and rebuilt every table on every run** — 27 combinations, every chunk re-embedded,
every time. Fine for six demo documents; **ruinous on a real corpus** (cost, wall-clock, rate
limits).

Here a document is re-chunked and re-embedded **only when its `content_hash` differs from the
`indexed_hash` we last embedded.** The chunk rewrite is **one transaction per document**, so a
crash mid-ingest can never leave a document half-indexed.

---

## 🔹 Cell 21 (code) — Step 7c: `ingest()`

### Building the vector index

```python
def ensure_vector_index():
    with db() as cur:
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {CHUNK_TABLE}_embedding_hnsw
            ON {CHUNK_TABLE} USING hnsw (embedding vector_cosine_ops)
            WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})
        """)
        cur.execute(f"ANALYZE {CHUNK_TABLE}")
```

**What HNSW is** (*Hierarchical Navigable Small World*): a multi-layer graph where each vector
links to its neighbours. Search enters at a coarse top layer, greedily walks toward the query, and
descends. **Analogy:** finding a house by taking the highway to the right city, then the main road
to the right neighbourhood, then the street. You never look at every house.

**`vector_cosine_ops`** — the **operator class**. It tells the index to organise vectors for
**cosine** distance, matching the `<=>` used in Cell 23. **Mismatching these means Postgres
silently ignores your index and does a full scan.** A classic, invisible performance bug — the
query still returns correct results, just 1000× slower.

**`ANALYZE`** refreshes Postgres's table statistics so the query planner will actually **choose**
the new index. Without it the planner may not believe the index is worth using.

**Requires `EMBED_DIM <= 2000`** — hence 1536.

### The incremental check

```python
clause = "TRUE" if force else "indexed_hash IS DISTINCT FROM content_hash"
```

> **⚠️ `IS DISTINCT FROM`, not `!=` — this is a genuine SQL trap.**
>
> In SQL, **any comparison with `NULL` returns `NULL`, not `TRUE`.** A never-indexed document has
> `indexed_hash = NULL`, so:
>
> ```sql
> NULL != 'abc123'   →  NULL   (not TRUE!)  →  WHERE clause fails  →  row skipped
> ```
>
> Using `!=` would **silently skip every new document you add.** You'd write documents, run
> ingest, see "nothing to ingest", and have no idea why. `IS DISTINCT FROM` is `!=` with
> NULL-aware semantics: `NULL IS DISTINCT FROM 'abc'` → `TRUE`.

```python
if only:
    clause += " AND doc_id = ANY(%s)"
    params.append(only)
```

**`force=True`** re-embeds everything. **You MUST do this after changing `EMBED_MODEL`,
`EMBED_DIM`, or the chunking settings** — the stored vectors were produced by different rules and
are now invalid. **`only=[ids]`** restricts to specific documents.

### The per-document loop

```python
for doc in pending:
    # 1. Chunk, with contextual headers
    chunks = build_chunks(doc["doc_id"], doc["title"], doc["content"], doc["metadata"])

    # 2. Embed the HEADER+TEXT version
    vectors = embed_texts([c["embed_input"] for c in chunks], "RETRIEVAL_DOCUMENT")

    # 3. Pair chunks with vectors
    rows = [
        (doc["doc_id"], c["chunk_index"], c["content"], c["embed_input"],
         c["token_count"], json.dumps(doc["metadata"]), str(v))
        for c, v in zip(chunks, vectors)
    ]
```

**Step 2 is the fix for your bug, in one line:** we embed `embed_input` (header + text), **not**
`content`. The supplier, category and date become **part of what the vector represents.**

**`str(v)`** — a Python list `[0.1, 0.2, ...]` stringifies to `"[0.1, 0.2, ...]"`, which is
**exactly pgvector's text input format.** Convenient coincidence, used throughout.

**`zip(chunks, vectors)`** walks both lists together — they're guaranteed the same length and
order because `embed_texts` preserves input order.

### The transaction — why this is crash-safe

```python
    with db() as cur:
        cur.execute(f"DELETE FROM {CHUNK_TABLE} WHERE doc_id = %s", (doc["doc_id"],))
        execute_values(cur, f"""
            INSERT INTO {CHUNK_TABLE}
                (doc_id, chunk_index, content, embed_input, token_count, metadata, embedding)
            VALUES %s
        """, rows, template="(%s, %s, %s, %s, %s, %s::jsonb, %s::vector)", page_size=200)
        cur.execute(f"UPDATE {DOC_TABLE} SET indexed_hash = %s WHERE doc_id = %s",
                    (doc["content_hash"], doc["doc_id"]))
```

**All three statements are inside ONE `with db()` block = ONE transaction.** Remember from Cell 8:
leaving the block normally commits, leaving via an exception rolls back. So:

> **Either all of it lands, or none of it does.** A crash here can **never** leave a document with
> its old chunks deleted and no new ones. This is the property that makes it safe to run ingest
> against production.

**Delete-then-insert rather than UPDATE:** the new chunk **count** may differ from the old one
(you changed `CHUNK_TOKENS`, or the document got longer), so there's no row-by-row correspondence
to update.

**Advancing the watermark inside the same transaction** is the subtle bit:

```python
cur.execute(f"UPDATE {DOC_TABLE} SET indexed_hash = %s ...")
```

Because it's in the same transaction, `indexed_hash` **can only be recorded if the chunks
actually landed.** The two can never disagree. If it were a separate transaction, a crash between
them would mark a document as indexed when its vectors were never written — and it would be
skipped forever.

**`page_size=200`** — how many rows `execute_values` sends per round trip. Batching within the
batching.

### Running it + the health check

```python
stats = ingest()

with db(dict_rows=True) as cur:
    cur.execute(f"""
        SELECT count(*) AS chunks, count(DISTINCT doc_id) AS docs,
               round(avg(token_count)) AS avg_tokens, max(token_count) AS max_tokens,
               pg_size_pretty(pg_total_relation_size('{CHUNK_TABLE}')) AS on_disk
        FROM {CHUNK_TABLE}
    """)
    print("\nIndex state:", dict(cur.fetchone()))
```

**`pg_size_pretty(pg_total_relation_size(...))`** — how much disk the table + its indexes actually
use, in human units ("2.4 MB"). Run this after loading real documents: it's how you project
storage cost, and how you decide whether 768 dims is worth benchmarking.

**Run the cell twice.** The second time it prints *"Nothing to ingest — every document is already
up to date."* **That's the whole feature working**, and it's the most satisfying demo in the
notebook.

---

## 🔹 Cell 22 (markdown) — Step 8 intro: hybrid retrieval

**The core upgrade.** Two independent retrievers run in **one** SQL statement, and their ranked
lists are merged with **Reciprocal Rank Fusion**:

```
score(chunk) = Σ_retrievers  weight_r / (k + rank_r(chunk)),   k = 60
```

**RRF fuses on *rank*, never on raw score.** That's the key insight: a cosine **distance** (0 =
best, lower is better) and a `ts_rank_cd` **score** (higher is better, unbounded) live on
completely incompatible scales. Trying to blend them numerically requires normalising both, which
is fragile and data-dependent. **Fusing on position sidesteps the problem entirely** — 1st place
is 1st place regardless of scale.

**Why both legs earn their keep:**

| Query | Dense alone | Lexical alone | Hybrid |
|---|---|---|---|
| *"who delivers fastest?"* | ✅ semantic paraphrase | ❌ no term overlap | ✅ |
| *"BID-2024-089"* | ⚠️ IDs embed poorly | ✅ exact token | ✅ |
| *"Acme customer satisfaction"* | ⚠️ | ✅ header carries "Acme" | ✅ |

**Three details that matter and are easy to get wrong** (all explained at the code):
- The KNN must sit in an **inner subquery**, or you lose the index.
- **`hnsw.ef_search`** is set **per transaction**.
- **JSONB metadata pre-filtering** is evaluated **inside** the ANN scan.

---

## 🔹 Cell 23 (code) — Step 8: The hybrid SQL

This is the densest cell in the notebook. Take it CTE by CTE.

> **Reading guide:**
> - `WITH name AS (...)` = a **CTE** (Common Table Expression), a named temporary result you can
>   refer to below. Think of it as a variable holding a table.
> - `%(name)s` = a **placeholder**. psycopg2 substitutes the value safely — **this is what
>   prevents SQL injection.** Never build SQL with f-strings around user input; table names only.

### CTE 1 — `vec`: the meaning search

```sql
WITH vec AS (
    SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY distance) AS rank
    FROM (
        SELECT chunk_id, embedding <=> %(qv)s::vector AS distance
        FROM rag_chunks
        WHERE (%(filter)s::jsonb IS NULL OR metadata @> %(filter)s::jsonb)
        ORDER BY embedding <=> %(qv)s::vector
        LIMIT %(vec_limit)s
    ) v
),
```

> ### 🔑 Why the nested SELECT — the most important performance detail in the notebook
>
> A window function like `ROW_NUMBER()` at the **same query level** as
> `ORDER BY embedding <=> ...` forces Postgres to **rank every row in the table**, which throws
> away the HNSW index and turns this into a **full sequential scan.**
>
> So: the **inner** query does the indexed nearest-neighbour lookup and takes the top 40. The
> **outer** query then numbers only those 40 surviving rows.
>
> **This one detail is the difference between a millisecond and a full table scan** — and the
> query returns identical results either way, so you'd never notice from the output. You'd only
> notice when the table hits a million rows and everything falls over.

**The metadata filter:**

```sql
WHERE (%(filter)s::jsonb IS NULL OR metadata @> %(filter)s::jsonb)
```

- If **no filter** was passed we send `NULL`, and `NULL IS NULL` is `TRUE`, so the whole condition
  passes and nothing is excluded. One SQL statement handles both cases — no string-building.
- **`@>` means "contains"**: `metadata @> '{"category":"Dairy"}'` matches rows whose metadata
  includes that key/value. Backed by the `jsonb_path_ops` GIN index from Cell 10.
- **This is a *pre*-filter** — it runs **inside** the ANN scan, not after it. Filtering afterwards
  would mean retrieving 40 candidates and possibly discarding 39, leaving you with 1 result.

### CTE 2 — `txt`: the keyword search

```sql
txt AS (
    SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY score DESC) AS rank
    FROM (
        SELECT c.chunk_id, ts_rank_cd(c.tsv, q) AS score
        FROM rag_chunks c, websearch_to_tsquery('english', %(question)s) q
        WHERE c.tsv @@ q
          AND (%(filter)s::jsonb IS NULL OR c.metadata @> %(filter)s::jsonb)
        ORDER BY score DESC
        LIMIT %(txt_limit)s
    ) t
),
```

- **`websearch_to_tsquery`** parses a normal user query **the way a search engine would** — it
  understands quotes and `OR` — so you can pass **raw user input straight in** without sanitising
  it into query syntax. (Contrast `to_tsquery`, which throws a syntax error on ordinary prose.)
- **The comma** between the table and the function is an **implicit CROSS JOIN**. It just names
  the parsed query as `q` so the rest of the query can refer to it.
- **`@@` means "this text matches this query."**
- **`ts_rank_cd`** scores how well a row matches, factoring in **cover density** — how *close* the
  matched words are to each other, not merely how often they appear. A chunk with "customer" and
  "satisfaction" adjacent beats one with them in different paragraphs.
- **`ORDER BY score DESC`** — note keyword scores are **higher is better**, the opposite of
  distance. That inversion is exactly the scale mismatch RRF avoids having to reconcile.

### CTE 3 — `fused`: Reciprocal Rank Fusion

```sql
fused AS (
    SELECT
        COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
        COALESCE(%(w_vec)s / (%(rrf_k)s + v.rank), 0)
      + COALESCE(%(w_txt)s / (%(rrf_k)s + t.rank), 0) AS rrf_score,
        v.rank AS vec_rank,
        t.rank AS txt_rank
    FROM vec v FULL OUTER JOIN txt t ON v.chunk_id = t.chunk_id
)
```

**`FULL OUTER JOIN`** keeps chunks found by **either** search. That's the whole point — a chunk
only the keyword search found must not be discarded. But that means `v.chunk_id` is `NULL` for
those rows, hence **`COALESCE`** (= "first non-NULL value") to get a usable id.

**The RRF formula, with real numbers:**

```
weight / (k + rank),  k = 60

rank  1  →  1/61  = 0.0164
rank  2  →  1/62  = 0.0161
rank 40  →  1/100 = 0.0100
```

**What this buys you:** a chunk that **both** searches ranked highly scores
`0.0164 + 0.0164 = 0.0328` and beats a chunk only one of them loved (`0.0164`). **Agreement
between two independent methods is strong evidence.**

**The second `COALESCE(..., 0)`** means "found by only one search" contributes 0 from the other,
rather than making the **whole sum NULL** (any arithmetic with NULL is NULL). Miss this and every
single-leg hit scores NULL and sorts unpredictably.

**`vec_rank` / `txt_rank` are kept for debugging** — you can see **which** search found each chunk
and where. `txt_rank = None` means the keyword search never found it. When retrieval misbehaves,
this is the first thing you look at.

### Final SELECT

```sql
SELECT c.chunk_id, c.doc_id, c.chunk_index, c.content, c.metadata,
       d.title, f.rrf_score, f.vec_rank, f.txt_rank,
       (c.embedding <=> %(qv)s::vector) AS cosine_distance
FROM fused f
JOIN rag_chunks c    ON c.chunk_id = f.chunk_id
JOIN rag_documents d ON d.doc_id   = c.doc_id
ORDER BY f.rrf_score DESC, cosine_distance ASC
LIMIT %(limit)s
```

**Why join back only now:** the CTEs carried around only **ids and scores** (cheap). Fetching the
actual `content` text for **hundreds** of candidates would be wasteful — we fetch it for the
handful that actually won.

**`cosine_distance` is recomputed here** so the relevance floor in Cell 27 has a number to check.

**`ORDER BY rrf_score DESC, cosine_distance ASC`** — best fused score first; distance breaks ties.

### The Python wrapper

```python
def retrieve(question, qv=None, mode="hybrid", top_k=TOP_K, filters=None):
    if qv is None:
        qv = list(embed_query_cached(question))

    params = {
        "qv": str(list(qv)),
        "question": question,
        "filter": json.dumps(filters) if filters else None,
        "vec_limit": VECTOR_CANDIDATES if mode in ("hybrid", "vector") else 0,
        "txt_limit": TEXT_CANDIDATES   if mode in ("hybrid", "text")   else 0,
        "w_vec":     W_VECTOR if mode in ("hybrid", "vector") else 0.0,
        "w_txt":     W_TEXT   if mode in ("hybrid", "text")   else 0.0,
        "rrf_k": RRF_K,
        "limit": top_k,
    }
    with db(dict_rows=True) as cur:
        cur.execute(f"SET LOCAL hnsw.ef_search = {int(HNSW_EF_SEARCH)}")
        cur.execute(HYBRID_SQL, params)
        return [dict(r) for r in cur.fetchall()]
```

**`qv` can be passed in.** Why: it lets Cell 33 **time the embedding and the search separately**
instead of lumping them together — **the exact mistake that made v1.1's benchmark meaningless.**

**`mode` — one switchboard drives all three modes.** Disabling a search leg is just setting its
`LIMIT` to 0 and its weight to 0. **Fewer code paths = fewer places for a bug to hide** — and,
critically, "vector only" here is genuinely the *same code* as hybrid, so the comparison in Cell
33 is honest.

**The single-leg modes exist so Step 11 can PROVE hybrid earns its complexity rather than
asserting it.** That's the difference between engineering and decoration.

**`SET LOCAL hnsw.ef_search`:**
- **`SET LOCAL`** confines it to **this transaction**, so one expensive query can't slow everyone
  else down.
- **Postgres does not allow placeholders in `SET`**, so the value must be interpolated.
  **`int()` makes that safe** by guaranteeing it's a number, not attacker-controlled text.

### The demo

```python
q = "What was the Customer Satisfaction level for supplier performance review?"
for mode in ("vector", "text", "hybrid"):
    print(f"\n--- {mode} ---")
    for i, h in enumerate(retrieve(q, mode=mode, top_k=3), 1):
        print(f"{i}. {h['doc_id']:<26} vec={h['vec_rank']} txt={h['txt_rank']} "
              f"dist={h['cosine_distance']:.3f}\n   {h['content'][:110]}...")
```

**Run this and study the output.** It's the same question v1.1 got wrong, shown three ways. You
can see exactly which leg found which chunk and at what rank. This printout is your evidence.

---

## 🔹 Cell 24 (markdown) — Step 9 intro: reranking

**The two-stage idea:**

| Stage | Optimises for | How | Cost |
|---|---|---|---|
| 1. Retrieval | **Recall** — don't miss it | Wide net: 40 candidates, 20 kept | Cheap |
| 2. Reranking | **Precision** — get the order right | A slower, smarter model reads query vs. each candidate | Expensive per item |

**Why two stages instead of one:** the accurate method is too slow to run against a million rows;
the fast method is too crude to order the final four. So you use the fast one to get from 1,000,000
→ 20, and the accurate one to get from 20 → 4.

**This is the highest-leverage single component in most production RAG systems**, and v1.1 had
**none** — its 3 vector hits went straight to the LLM.

**In production, prefer a dedicated cross-encoder.** On Google Cloud that's the **Vertex AI
Ranking API** (`semantic-ranker-default@latest`) — purpose-built, and far cheaper and
lower-latency than an LLM call. The LLM reranker here is the **portable equivalent** so the
notebook runs with no extra service enabled. **`rerank()` is a drop-in seam — replace the body,
keep the signature.**

---

## 🔹 Cell 25 (code) — Step 9: `rerank()`

```python
RERANK_PROMPT = """You are a search relevance rater for a procurement document system.

Rate how well each passage answers the QUESTION, on a 0-10 scale:
  10 = contains the exact answer
   7 = directly about the question's subject, partial answer
   4 = same topic, does not answer the question
   0 = irrelevant

Judge each passage independently. Return ONLY a JSON array of objects with keys
"id" (the passage number) and "score" (integer 0-10). No prose.

QUESTION: {question}

PASSAGES:
{passages}"""
```

**Why an explicit rubric with anchors:** "rate 0–10" alone produces inconsistent, drifting scores.
Naming what 10, 7, 4 and 0 *mean* makes the scores comparable across calls. **The 7-vs-4
distinction is the crucial one** — "about the subject" vs "actually answers it" is exactly the
difference between the right chunk and a plausible decoy.

**"Judge each passage independently"** stops the model from grading on a curve within one batch.

```python
def rerank(question, hits, top_n=RERANK_TOP_N):
    if not hits:
        return hits
    passages = "\n\n".join(
        f"[{i}] ({h['title']}) {h['content'][:1200]}" for i, h in enumerate(hits)
    )
```

**`[:1200]`** caps each passage so 20 candidates can't blow the reranker's context window.

```python
    try:
        resp = _with_retry(lambda: genai_client.models.generate_content(
            model=RERANK_MODEL,
            contents=RERANK_PROMPT.format(question=question, passages=passages),
            config=GenerateContentConfig(temperature=0, response_mime_type="application/json"),
        ))
        scores = {int(r["id"]): float(r["score"]) for r in json.loads(resp.text)}
    except Exception as e:
        print(f"  [rerank unavailable, keeping RRF order: {type(e).__name__}]")
        return hits[:top_n]
```

- **`temperature=0`** — as deterministic as possible. You want a **rating**, not creativity.
- **`response_mime_type="application/json"`** — constrains the model to emit valid JSON. Without
  it you get markdown code fences around the JSON and `json.loads` fails.

> **The `except` block is a deliberate reliability decision, not laziness.** If the reranker errors
> or returns malformed JSON, we **keep the RRF order** rather than failing the request.
>
> **A reranker outage should degrade quality, not take retrieval down.** The user gets a slightly
> worse answer instead of an error page. Point this out — it's the kind of thinking that separates
> a demo from a service.

```python
    for i, h in enumerate(hits):
        h["rerank_score"] = scores.get(i, 0.0)
    ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
    return [h for h in ranked if h["rerank_score"] >= 4][:top_n] or ranked[:1]
```

**The `>= 4` threshold is doing real work.** It **drops candidates the reranker judged off-topic
even when there is room for them.** Why: **padding the context with irrelevant passages measurably
degrades the answer.** If only 2 of 20 candidates are relevant, send 2 — not 4.

**`or ranked[:1]`** — if *everything* scored below 4, still return the single best one rather than
nothing. The relevance floor in Cell 27 handles genuine no-match cases; this just avoids an empty
list breaking downstream code.

**`scores.get(i, 0.0)`** — if the model skipped a passage id, default to 0 rather than `KeyError`.

---

## 🔹 Cell 26 (markdown) — Step 10 intro: the prompt

**The prompt is where your wrong answer was actually produced**, so it carries explicit rules:

1. **Cite every fact** with a source number — an unverifiable answer becomes impossible.
2. **List *all* matching entities** — the direct fix for `4.8/5` vs `8.5/10`. The review holds one
   per-supplier figure and one district-wide figure; a good answer surfaces **both, labelled**,
   instead of silently choosing.
3. **Answer for the entity that was asked about** — if you ask about Acme and the context only
   covers the district, the model must **say so** rather than substituting a different number.
4. **Quote figures exactly** — no rounding, no unit conversion, no "approximately".
5. **A fixed refusal string** — **machine-detectable**, so you can alarm on refusal rate in
   production.

Plus a **relevance floor**: if nothing clears `MAX_COSINE_DISTANCE` and no lexical leg fired, we
refuse **before spending a generation call.**

---

## 🔹 Cell 27 (code) — Step 10: `ask()`

### The rules

```python
REFUSAL = "I don't have that in the indexed documents."

SYSTEM_RULES = f"""You are a procurement analyst assistant. Answer using ONLY the numbered SOURCES below.

Rules:
1. Cite every fact with its source number in square brackets, e.g. [2]. No citation, no claim.
2. If several entities or several values legitimately match the question, list ALL of them with
   attribution. Never collapse distinct figures into one number, and never present an aggregate as
   though it were an individual entity's value (or the reverse).
3. If the question names a specific entity (supplier, bid, category), answer for THAT entity. If the
   sources only cover a different entity, say so explicitly instead of substituting it.
4. Quote figures exactly as written in the sources - same units, scale, currency and precision.
5. If the sources do not contain the answer, reply with exactly: "{REFUSAL}"
   Do not guess, and do not use knowledge from outside the sources.
6. Be concise. Lead with the answer, then the supporting detail."""
```

Map each rule to the failure it prevents:

| Rule | Prevents |
|---|---|
| 1 — cite everything | Unverifiable claims. If you can't cite it, you can't say it. |
| 2 — list all matches | **Your exact bug.** *"never present an aggregate as though it were an individual entity's value"* is `8.5/10` vs `4.8/5` in one sentence. |
| 3 — answer for the entity asked | Substituting a different supplier's number |
| 4 — quote exactly | Silent rounding — `4.8/5` becoming "about 5 stars" |
| 5 — fixed refusal string | Hallucination, **and** unmeasurable refusals |
| 6 — be concise | Burying the answer in preamble |

**Why a *fixed* refusal string:** because `result["refused"]` can then be computed with
`answer.startswith(REFUSAL)`. **Machine-detectable** means you can chart refusal rate in
production, and a sudden spike tells you retrieval broke **before users complain.** A free-form
"I'm not sure I can help with that" is invisible to monitoring.

### Formatting the sources

```python
def format_sources(hits):
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h.get("metadata") or {}
        facets = " | ".join(f"{k}: {m[k]}" for k in HEADER_FIELDS if m.get(k))
        head = f"[{i}] {h['title']} (doc_id: {h['doc_id']}" + (f" | {facets}" if facets else "") + ")"
        blocks.append(f"{head}\n{h['content']}")
    return "\n\n".join(blocks)
```

**Each block carries the title AND metadata**, so the model can attribute a bare number like
`4.8/5` to the right supplier. **Without this the model sees floating facts with no owner —
which is precisely how v1.1 produced the wrong answer.**

Note the metadata appears here **again**, having already been used in the embedding header. Same
information, two different jobs: the header helps **find** the chunk; this helps **attribute** it.

**`enumerate(hits, 1)`** numbers from 1 — matching the `[1] [2] [3]` the citation rule refers to.
**`or {}`** guards against `metadata` being `None`.

### The four stages

```python
def ask(question, filters=None, mode="hybrid", use_reranker=USE_RERANKER, verbose=False):
    timings = {}

    # Stage 1: embed
    t0 = time.perf_counter()
    qv = list(embed_query_cached(question))
    timings["embed_ms"] = (time.perf_counter() - t0) * 1000

    # Stage 2: search
    first_stage_k = RERANK_CANDIDATES if use_reranker else TOP_K
    t0 = time.perf_counter()
    hits = retrieve(question, qv=qv, mode=mode, top_k=first_stage_k, filters=filters)
    timings["search_ms"] = (time.perf_counter() - t0) * 1000
```

**`time.perf_counter()`** is the high-resolution **monotonic** clock — the right one for timing
short operations (unlike `time.time()`, which can jump if the system clock is adjusted).

**`first_stage_k`** — retrieve **more** candidates when reranking, because the reranker's job is
to sift a wide net. Without one, take only what will fit in the prompt. Adaptive, not fixed.

### The guardrail

```python
    if not hits or (min(h["cosine_distance"] for h in hits) > MAX_COSINE_DISTANCE
                    and all(h["txt_rank"] is None for h in hits)):
        return {"answer": REFUSAL, "sources": [], "timings": timings, "refused": True}
```

**Two conditions must BOTH hold to refuse:**
1. Nothing was semantically close (`min distance > 0.80`), **AND**
2. No keyword matched either (`txt_rank is None` for everything).

**Why require both:** on a valid **exact-ID lookup**, cosine distance can look poor (IDs embed
badly — that's the whole premise of hybrid search) but the **keyword hit is decisive.** Refusing
on distance alone would break exactly the queries hybrid search was added to fix.

**This runs BEFORE the generation call** — you don't pay for an LLM call you're going to discard.

> v1.1 had no such check: it always returned its top 3, however irrelevant. **That is how RAG
> systems end up confidently answering from unrelated documents.**

### Rerank + generate

```python
    t0 = time.perf_counter()
    hits = rerank(question, hits) if use_reranker else hits[:RERANK_TOP_N]
    timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

    prompt = f"{SYSTEM_RULES}\n\nSOURCES:\n{format_sources(hits)}\n\nQUESTION: {question}"
    if verbose:
        print(prompt[:2000], "\n...\n")

    t0 = time.perf_counter()
    resp = _with_retry(lambda: genai_client.models.generate_content(
        model=CHAT_MODEL, contents=prompt,
        config=GenerateContentConfig(temperature=0, max_output_tokens=1024),
    ))
    timings["generate_ms"] = (time.perf_counter() - t0) * 1000
```

**Prompt structure: rules first, then the evidence, then the question.** That ordering is
deliberate — the model reads the constraints before it reads anything it might be tempted to
answer from.

**`temperature=0`** — creativity is exactly what you do **NOT** want here. You want faithful
reporting of the sources.

> **`verbose=True` prints the assembled prompt. Do this once.** Seeing exactly what the model was
> given is the **fastest way to understand any wrong answer.** Nine times out of ten the answer is
> obviously wrong *given that context*, and the real bug is in retrieval.

### The return value

```python
    return {
        "answer": answer,
        "sources": [{"n": i, "doc_id": h["doc_id"], "title": h["title"],
                     "content": h["content"], "metadata": h.get("metadata") or {},
                     "cosine_distance": round(h["cosine_distance"], 4),
                     "rerank_score": h.get("rerank_score")}
                    for i, h in enumerate(hits, 1)],
        "timings": timings,
        "refused": answer.startswith(REFUSAL),
    }
```

**Not just a string — a full audit record.** Every answer comes with **the sources behind it**
and **per-stage timings.** That means:
- **Every answer is auditable** — you can always show a user *why* the system said what it said.
- **Every slow request is diagnosable** — you know instantly whether it was embed, search, rerank
  or generate.

v1.1's `ask()` returned `resp.text` and nothing else. You could not debug it.

**`show()`** just pretty-prints all of that.

---

## 🔹 Cells 28–29 — The regression test

```python
show(ask("What was the Customer Satisfaction level for supplier performance review?"))
show(ask("What was Acme Foods' customer satisfaction rating?"))
show(ask("What is Acme's organic produce certification policy?"))   # not in corpus -> refuse
```

**Three deliberate tests:**

| Query | Expected behaviour | Tests |
|---|---|---|
| 1. The original failing question | Surfaces **both** satisfaction figures, **labelled and cited** | Rule 2 + headers |
| 2. Acme's rating specifically | Gives **4.8/5** attributed to Acme, not the district average | Rule 3 |
| 3. Organic certification policy | **Refuses** — the fixed refusal string | Rule 5 + relevance floor |

**Query 3 is the one people forget to test, and it's the most important.** A system that answers
everything is a system that hallucinates. **Knowing when to *not* answer is a feature** — demo it.

---

## 🔹 Cell 30 — Two more capability demos

```python
show(ask("What are the evaluation criteria weightings for BID-2024-089?"))
show(ask("What are the payment terms?", filters={"category": "Dairy"}))
```

**Query 1 — exact-identifier lookup.** `BID-2024-089` is the case **pure dense retrieval loses
and hybrid wins.** Check `txt_rank` in the output: the keyword leg found it. **This is your
proof** that hybrid search earns its complexity.

**Query 2 — metadata pre-filtering.** *"What are the payment terms?"* is hopelessly ambiguous —
five suppliers have different terms. `filters={"category": "Dairy"}` constrains the ANN scan
**before it runs**, so only Globex's chunks are candidates.

**Why this matters architecturally:** pure vector search **cannot** reliably enforce hard
constraints. "Only dairy" is not a semantic property; it's a fact. **Filter with SQL, rank with
vectors.** That combination is exactly what a relational vector store gives you and a
special-purpose vector service makes awkward — a genuinely good answer to *"why Postgres?"*

---

## 🔹 Cell 31 (markdown) — Step 11 intro: honest evaluation

**Three flaws made v1.1's leaderboard unusable:**

1. **Latency included the ~3.2 s embedding call**, drowning the sub-millisecond Postgres search it
   was meant to measure. Every index scored ~3.2 s → no signal → it crowned a brute-force scan.
2. **Doc-level gold only.** *"The right document was in the top 3"* does not tell you whether the
   chunk holding the **answer** was retrieved. **Your satisfaction question retrieved the right
   document and still produced a wrong answer — v1.1's metrics would have scored that a perfect
   1.0.** That is a devastating indictment of the metric.
3. **No unanswerable questions**, so **hallucination was never measured at all.**

**v2 addresses all three:**
- **`embed_ms` / `search_ms` / `rerank_ms` reported separately** — `search_ms` is now the number
  you tune.
- **`answer_support@k`** — did a retrieved chunk actually contain the answer string? **This is the
  metric that would have caught your bug.**
- **nDCG@k** alongside recall/MRR, and a **refusal-correctness** check in Step 12.
- **Corrected gold labels** — *"Which suppliers offered an early payment discount?"* now lists all
  three. v1.1 was marking two correct retrievals as failures.

All configurations are scored on the **same final context size** (`RERANK_TOP_N`), so reranked and
non-reranked runs are actually comparable.

> **13 questions is still a demo. Before you deploy, write 50–100 questions from real user logs.
> That is the highest-value hour you can spend on a RAG system** — nothing else tells you whether
> a change helped.

---

## 🔹 Cell 32 (code) — Step 11: The gold set

```python
GOLD = [
    {"q": "What did Acme quote for romaine lettuce?",
     "doc_ids": ["acme-produce-2024"], "must_contain": ["3.20"]},
    ...
]
```

**Two fields per question, and the second is the innovation:**

| Field | Meaning |
|---|---|
| `doc_ids` | **Every** document that legitimately answers the question (a **list**, not one id) |
| `must_contain` | The **literal string** that must appear in a retrieved chunk for the answer to be derivable |

**`must_contain` is what catches "right document, wrong chunk."** Doc-level recall says "we found
`performance-review-2024`" — but did we retrieve the chunk containing `4.8/5`, or the one
containing `8.5/10`? Only `must_contain` can tell you. **This single field is the metric v1.1
lacked.**

**The 13 questions, and what each is for:**

| # | Question | Tests |
|---|---|---|
| 1 | Acme romaine price | Basic semantic lookup |
| 2 | **Acme's Q1 satisfaction rating** (`4.8/5`) | **Your original bug** |
| 3 | **District satisfaction** (`8.5/10`) | The **decoy** — proves the system distinguishes them |
| 4 | **Which supplier*s* offered early payment discount** — 3 doc_ids | **Corrected label**; multi-entity |
| 5 | Globex payment terms (`Net-45`) | Entity-specific fact |
| 6 | Supplier response rate (`62.5%`) | Percentage extraction |
| 7 | Customer due date (`2024-05-25`) | Date extraction |
| 8 | Sysco salmon price (`12.80`) | Specific line item among many |
| 9 | **`BID-2024-089` evaluation criteria** | **Exact-ID lookup — hybrid's proof case** |
| 10 | Lowest on-time delivery (`92%`) | Comparison across entities |
| 11 | Minimum order for beverages (`$250`) | Category-scoped fact |
| 12 | Cold chain on dairy (`Cold chain`) | Attribute lookup |
| 13 | **Organic certification policy** — `unanswerable: True` | **Refusal / hallucination** |

**Questions 2 and 3 as a pair are the sharpest test in the set.** They're near-identical in
wording and point at **different numbers in the same document.** A system that can't tell them
apart will pass one and fail the other — which is exactly what v1.1 did.

**Question 13 is the hallucination detector.** With `doc_ids: []` and `unanswerable: True`, the
only correct behaviour is to **refuse.**

---

## 🔹 Cell 33 (code) — Step 11: `evaluate()`

### nDCG

```python
def _ndcg(ranked_docs, gold_docs, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked_docs[:k]) if d in gold_docs)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_docs), k)))
    return dcg / ideal if ideal else 0.0
```

**What it measures:** like MRR, but handles questions with **several** correct documents (such as
question 4, which has three).

**How to read the formula:**
- **`1.0 / log2(i + 2)`** is the positional discount: position 0 → `1/log2(2)` = 1.00, position 1
  → `1/log2(3)` = 0.63, position 2 → `1/log2(4)` = 0.50. **Being found higher is worth more.**
- **`dcg`** sums that discount for every correct document actually found.
- **`ideal`** is the score you'd get if all correct documents were at the very top.
- **`dcg / ideal`** normalises to 0–1, so questions with 1 correct doc and 3 correct docs are
  comparable.

### The evaluation loop

```python
def evaluate(mode="hybrid", use_reranker=False, k=RERANK_TOP_N):
    answerable = [g for g in GOLD if not g.get("unanswerable")]
    ...
    for g in answerable:
        t0 = time.perf_counter()
        qv = embed_query(g["q"])                    # ← UNCACHED, deliberately
        t_embed += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        hits = retrieve(g["q"], qv=qv, mode=mode,
                        top_k=RERANK_CANDIDATES if use_reranker else k)
        t_search += (time.perf_counter() - t0) * 1000
```

> **⭐ `embed_query` (uncached), not `embed_query_cached` — and the embedding is timed
> *separately* from the search.** This is the direct fix for v1.1's fatal measurement error.
>
> - **Uncached** → `embed_ms` reflects a real **cold** request, not a cache hit.
> - **Separate timers** → `search_ms` is finally the number you can tune. It's a few milliseconds
>   of Postgres work that v1.1 was reporting as ~3200 ms because it timed the API call alongside
>   it.

**`hits = hits[:k]`** — **every config is scored on the same final list size `k`**, so reranked
(20 → 4) and non-reranked (4) runs are **fairly comparable.** Without this, the reranked config
would be scored on 20 candidates and win trivially.

### The four metrics

```python
        ranked = [h["doc_id"] for h in hits]
        gold_docs = set(g["doc_ids"])
        if gold_docs & set(ranked):                             # recall
            recall += 1
            first = next(i for i, d in enumerate(ranked) if d in gold_docs)
            mrr += 1.0 / (first + 1)                            # MRR
        ndcg += _ndcg(ranked, gold_docs, k)                     # nDCG
        blob = " ".join(h["content"] for h in hits).lower()
        if all(s.lower() in blob for s in g["must_contain"]):   # answer_support
            support += 1
```

| Metric | Question it answers | Formula |
|---|---|---|
| **recall@k** | Did we find a correct doc at all? | `hits / total` |
| **MRR** | How **high** did the first correct doc rank? | `1 / (position + 1)` |
| **nDCG@k** | When several docs are correct, did we rank them all well? | see above |
| **answer_support** | Was the **answer text** actually retrieved? | substring check |

**`gold_docs & set(ranked)`** — set **intersection**. Non-empty means at least one correct
document was retrieved.

**`answer_support` is the important one.** It joins all retrieved chunks into one blob and checks
whether every required string appears. **Recall says "we found the document"; support says "we
found the sentence."** The gap between them is exactly where your bug lived.

### The comparison table

```python
rows = []
for mode, rr in [("text", False), ("vector", False), ("hybrid", False), ("hybrid", True)]:
    r = evaluate(mode, rr)
    rows.append(r)
```

**Four configurations, and the progression is the argument:**

| Config | What it proves |
|---|---|
| `text` | Keyword-only baseline |
| `vector` | Semantic-only baseline (**this is what v1.1 was**) |
| `hybrid` | **Does fusing them beat either alone?** |
| `hybrid + rerank` | **Does reranking add more on top?** |

**This is how you justify complexity: by measuring the alternative, not by asserting.** If hybrid
doesn't beat vector on your data, you've learned something valuable and should say so.

**Expected output on 6 documents:** most configs will tie. The notebook says so itself:

> *"On six documents most configs will tie; the spread appears at real corpus size, and the
> exact-ID and multi-entity questions are where hybrid and reranking already separate."*

**Look at questions 4 and 9 specifically** — that's where the difference shows up even on a tiny
corpus.

---

## 🔹 Cell 34 (markdown) — Step 12 intro: LLM-as-judge

**Retrieval metrics tell you whether the right text was *found*. They cannot tell you whether the
answer was *right*.**

**Your satisfaction question is the proof: retrieval was fine, the answer was not.** Cell 33's
metrics would have given it 1.0.

So a second harness scores the **generated answers** on three axes:

| Axis | What it catches |
|---|---|
| **groundedness** | Is every claim supported by the retrieved sources? → **hallucination detector** |
| **correctness** | Does it contain the expected fact? |
| **completeness** | When several entities match, are they all covered? → **the exact v1.1 failure** |

Plus a **deterministic `exact_fact_present` backstop** and a hard **refusal check**.

> **An LLM judge is noisy on any single item.** Use it to compare **configurations across a whole
> set**, and keep a small human-labelled slice as ground truth. Treating a single judge score as
> authoritative is a mistake; treating the average across 12 questions as a comparison signal is
> reasonable.

---

## 🔹 Cell 35 (code) — Step 12: `judge_answers()`

```python
JUDGE_PROMPT = """You are evaluating a retrieval-augmented answer. Be strict.

QUESTION: {question}
EXPECTED FACT(S) a correct answer must contain: {expected}

SOURCES THAT WERE RETRIEVED:
{sources}

ANSWER GIVEN:
{answer}

Score 0-5 on each axis:
- groundedness: every claim is supported by the sources (5 = fully grounded, 0 = fabricated)
- correctness: the answer contains the expected fact(s), stated accurately
- completeness: if several entities or values legitimately match, all are covered and attributed

Return ONLY JSON: {{"groundedness": int, "correctness": int, "completeness": int, "why": "one sentence"}}"""
```

**The judge sees the sources too**, so it can verify **groundedness** — whether each claim is
actually supported — rather than just judging plausibility. A judge without the sources can only
assess whether an answer *sounds* right, which is precisely the failure mode you're hunting.

**`"why": "one sentence"`** forces a brief justification. When a score looks wrong, the `why`
tells you whether the judge or the system was at fault.

**Doubled braces `{{...}}`** because this is a `.format()` template — single braces are
placeholders.

```python
def judge_answers(mode="hybrid", use_reranker=True):
    scored, refusals_ok, refusals_total = [], 0, 0

    for g in GOLD:
        res = ask(g["q"], mode=mode, use_reranker=use_reranker)

        if g.get("unanswerable"):
            refusals_total += 1
            ok = res["refused"]
            refusals_ok += ok
            print(f"{'PASS' if ok else 'FAIL'}  (refusal) {g['q'][:60]}")
            continue
```

**Unanswerable questions are handled separately** with a **deterministic pass/fail**, not an LLM
score. Refusal is binary — either it refused or it didn't, and `res["refused"]` knows because the
refusal string is fixed (Cell 27, rule 5). **No judgement call needed, so don't introduce one.**

```python
        verdict["exact"] = all(s.lower() in res["answer"].lower() for s in g["must_contain"])
```

**The deterministic backstop.** Does the expected literal actually appear in the **answer**?

**Why have both a judge and a substring check:** the judge can be talked into generosity; a
substring check cannot. **If `correctness` is 5 but `exact` is False, look hard** — either the
answer paraphrased a number it should have quoted verbatim (violating rule 4), or your
`must_contain` string is too strict. Either way you learned something.

```python
    if scored:
        avg = lambda key: sum(s[key] for s in scored) / len(scored)
        print(f"\nGroundedness {avg('groundedness'):.2f}/5 | ...")
    if refusals_total:
        print(f"Correct refusals: {refusals_ok}/{refusals_total}")
```

**The four numbers you take to the client meeting:** groundedness, correctness, completeness, and
correct refusals.

---

## 🔹 Cell 36 (markdown) — Step 13 intro: the configuration sweep

> **This is the cell you run before a client meeting.** It answers *"why this chunker?"* and
> *"why this embedding model?"* **with a table instead of an opinion.**

It compares, on **your** documents and **your** questions:

- **5 chunking strategies** — all three of v1.1's (`fixed_400`, `sentence_500`, `token_256`) plus
  the recursive splitter at two sizes, **so the comparison is honest rather than rigged toward the
  new one.**
- **Several embedding models** — including the older generation, so *"why not 004?"* has a
  **measured** answer.
- **Several dimensions** — 3072 vs 1536 vs 768 via Matryoshka truncation. Directly answers *"can
  we halve storage without hurting accuracy?"*
- **Header on vs off** — an **ablation**: same everything, contextual header removed. **This
  isolates how much the header actually contributes**, which is how you defend it as a design
  decision rather than a hunch.

**Each configuration gets its own throwaway table** (`rag_sweep_*`), because vectors from
different models and dimensions cannot share one. They're **dropped at the end** — unlike v1.1,
which dropped the table it had just crowned the winner.

### ⭐ Read the output honestly

The report prints a **95% confidence interval** next to each score. **With 12 questions that
interval is roughly ±25 points** — so most configurations will be **statistically
indistinguishable**, and the table will say so.

> **That is the correct result, not a broken one.** It is the same reason v1.1's leaderboard was
> full of `0.80, 0.80, 0.80` ties.

**What to do with that:** treat the sweep as a **screening tool.** It reliably catches
configurations that are *badly* wrong (a chunker that shreds your tables, a dimension too small
for your domain). **It cannot resolve a 3-point gap until you have ~50–100 questions.**

> **Report it that way and you will be credible. Report a 3-point win from 12 questions as
> decisive and a sharp reviewer will take the whole analysis apart.**

**💰 Cost warning.** This re-embeds your whole corpus **once per configuration.** Six demo
documents is pennies; **10,000 real documents × 12 configs is a real bill.** Start with
`SWEEP_CONFIGS[:4]`, and sweep on a representative **sample** of a large corpus.

---

## 🔹 Cell 37 (code) — Step 13: The candidates

```python
def chunk_fixed(text, size=400, overlap=60): ...     # v1.1's fixed_400
def chunk_sentence(text, size=500): ...              # v1.1's sentence
def chunk_tokens(text, size=256, overlap=40): ...    # v1.1's token_256

CHUNKERS = {
    "fixed_400":     lambda t: chunk_fixed(t, 400, 60),      # v1.1
    "sentence_500":  lambda t: chunk_sentence(t, 500),       # v1.1
    "token_256":     lambda t: chunk_tokens(t, 256, 40),     # v1.1
    "recursive_350": lambda t: split_text(t, 350, 60),       # v2 default
    "recursive_600": lambda t: split_text(t, 600, 100),      # v2, larger
}
```

**v1.1's three chunkers are reproduced verbatim** — and that's a deliberate integrity choice.
**Keeping them means you can show a client the exact methods that were considered**, and the
comparison is **fair rather than rigged** toward the new one. If `recursive_350` wins, it wins
against real competition.

Each chunker's docstring names its own weakness:
- `chunk_fixed` — *"happily cuts mid-word and mid-number"*
- `chunk_sentence` — *"a character budget does not map cleanly to the model's token limit"*
- `chunk_tokens` — *"guarantees the token limit is respected, but cuts mid-sentence"*

**The `lambda`s freeze the parameters**, so every caller gets a uniform `f(text) -> list[str]`
interface.

```python
SWEEP_CONFIGS = [
    # --- vary the CHUNKER, model fixed ("why this chunking method?") ---
    {"name": "gem001-1536 / fixed_400",     ..., "chunker": "fixed_400",     "header": True},
    {"name": "gem001-1536 / sentence_500",  ..., "chunker": "sentence_500",  "header": True},
    {"name": "gem001-1536 / token_256",     ..., "chunker": "token_256",     "header": True},
    {"name": "gem001-1536 / recursive_350", ..., "chunker": "recursive_350", "header": True},
    {"name": "gem001-1536 / recursive_600", ..., "chunker": "recursive_600", "header": True},

    # --- vary the MODEL, chunker fixed ("why not 004 / 005?") ---
    {"name": "text-emb-004 / recursive_350", "model": "text-embedding-004", "dim": 768,  ...},
    {"name": "text-emb-005 / recursive_350", "model": "text-embedding-005", "dim": 768,  ...},

    # --- vary the DIMENSION ("can we halve storage?") ---
    {"name": "gem001-3072 / recursive_350", "dim": 3072, ...},
    {"name": "gem001-768  / recursive_350", "dim": 768,  ...},

    # --- ABLATION: identical to v2 default, header removed ---
    {"name": "gem001-1536 / recursive_350 / NO HEADER", ..., "header": False},
]
```

**Notice the structure: change ONE thing at a time.** That's what makes the results
interpretable — this is a **controlled experiment**, not a random search. Each group answers a
specific question a client will ask:

| Group | Answers |
|---|---|
| Vary chunker | *"Why this chunking method?"* |
| Vary model | *"Why not 004/005?"* |
| Vary dimension | *"Can we halve storage?"* |
| **Ablation** | *"Does the header actually help?"* |

**The ablation is the most sophisticated entry.** It's identical to the v2 default with **one
thing removed.** If it scores the same, the header isn't earning its tokens and you should say so.
**Being willing to measure whether your own idea works is exactly what a reviewer is checking
for.**

---

## 🔹 Cell 38 (code) — Step 13: The sweep machinery

```python
_sweep_qcache = {}

def _sweep_embed_query(cfg, question):
    key = (cfg["model"], cfg["dim"], question)
    if key not in _sweep_qcache:
        _sweep_qcache[key] = embed_texts([question], "RETRIEVAL_QUERY",
                                         model=cfg["model"], dim=cfg["dim"])[0]
    return _sweep_qcache[key]
```

**Cost optimisation.** The five chunker configs all use the same model and dimension, so the same
question embeds identically for all of them. Caching by `(model, dim, question)` avoids
re-embedding **12 questions × 10 configs**. Note the key **includes model and dim** — because
vectors from different models are not comparable, sharing across them would be a correctness bug,
not just an optimisation.

```python
def sweep_ingest(cfg):
    table = _sweep_table_name(cfg)
    with db() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cur.execute(f"CREATE TABLE {table} (chunk_id bigserial PRIMARY KEY, doc_id text NOT NULL,
                     content text NOT NULL, embedding vector({cfg['dim']}) NOT NULL)")
```

**`DROP TABLE IF EXISTS` is safe here** — these really are throwaway tables, unlike production's
`rag_chunks`. **`_sweep_table_name`** slugs the config name and truncates to 60 chars (Postgres
identifiers max at 63; **silent** truncation could make two configs collide into one table and
corrupt the whole sweep).

```python
    chunk_fn = CHUNKERS[cfg["chunker"]]
    rows = []
    for d in docs:
        header = contextual_header(...) if cfg["header"] else ""
        for piece in chunk_fn(d["content"]):
            rows.append((d["doc_id"], piece, f"{header}\n\n{piece}" if header else piece))
```

**The `if cfg["header"]` is the ablation switch** — one boolean turns the header on or off,
everything else identical.

```python
        if cfg["dim"] <= 2000:
            cur.execute(f"CREATE INDEX ON {table} USING hnsw (embedding vector_cosine_ops)")
```

**The 3072 config falls back to an exact scan** — HNSW can't index it. That's fine for a sweep
(we're measuring retrieval **quality**, not speed), **but it is exactly the constraint that makes
3072 impractical to actually deploy.** The sweep can tell you 3072 is more accurate; it can't make
it indexable.

```python
def sweep_eval(cfg, table, k=RERANK_TOP_N):
    """Score one configuration using dense retrieval only."""
```

> **Vector-only on purpose.** The keyword leg is **identical for every config** (it doesn't use
> embeddings), so including it would **dilute exactly the difference we're trying to measure.**
> This is careful experimental design — isolating the variable under test.

### Wilson confidence interval

```python
def wilson_ci(p, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))
```

**Plain-English version: "the true score is somewhere in this range."**

**Why Wilson and not the textbook formula:** the standard normal-approximation interval breaks
down for **small n** and for scores near **0 or 1** — which is **exactly our situation** (12
questions, scores often 0.9+). It can even produce intervals extending below 0 or above 1, which
is nonsense. Wilson stays sensible.

**`z=1.96`** is the 95% confidence multiplier.

**This function is what makes the sweep honest.** Without it you'd read `0.83` vs `0.75` as a
win. With it you read `[0.55, 0.95]` vs `[0.47, 0.91]` and see they **overlap heavily** — you
cannot distinguish them on this evidence.

---

## 🔹 Cell 39 (code) — Step 13: Run the sweep + report

```python
CONFIGS_TO_RUN = SWEEP_CONFIGS       # ← set to SWEEP_CONFIGS[:4] for a cheap trial

for cfg in CONFIGS_TO_RUN:
    try:
        table, n_chunks = sweep_ingest(cfg)
        scores = sweep_eval(cfg, table)
        mb = n_chunks * (4 * cfg["dim"] + 8) / 1024 / 1024
        ...
    except Exception as e:
        sweep_results.append({**cfg, "ok": False, "error": f"{type(e).__name__}: {e}"})
```

**`try/except` per config** — a config can legitimately fail (a model not enabled in your project,
an unsupported dimension). **Record it and keep going** rather than losing the whole run. Without
this, one predictable failure at config #7 would kill the remaining three and waste everything
already computed.

**Storage estimate:** `n_chunks * (4 * dim + 8) / 1024 / 1024` — pgvector stores **4 bytes per
dimension** plus ~8 bytes of row overhead. This is how you answer *"what will this cost to
store?"* before you commit.

### The report

```python
ok.sort(key=lambda r: (r["support"], r["mrr"], r["recall"]), reverse=True)
```

**Sorted by `support` first**, not recall. **Answer-support is the metric closest to what users
actually experience** — did we retrieve the text containing the answer? Recall (did we get the
right document) is a weaker proxy. **The sort order is a statement about what you consider
"better", so be ready to defend it.**

### ⭐ The honesty check

```python
if ok:
    best = ok[0]
    b_lo, b_hi = wilson_ci(best["support"], best["n"])
    tied = [r for r in ok[1:] if wilson_ci(r["support"], r["n"])[1] >= b_lo]
    print(f"\nHighest score: {best['name']} (answer-support {best['support']:.2f})")
    print(f"Statistically indistinguishable from it on {best['n']} questions: {len(tied)} config(s)")
```

**`tied`** = every config whose confidence-interval **upper bound** reaches the winner's **lower
bound.** **Anything whose interval overlaps the winner's cannot be called worse on this
evidence.**

```python
        print(f"\n  With only {best['n']} questions the confidence interval is ~"
              f"±{(b_hi - b_lo) / 2 * 100:.0f} points, so this sweep CANNOT separate those.")
        print("  Report it as directional. To actually rank them, grow GOLD to 50-100 questions.")
```

> **This is the most professionally valuable output in the entire notebook.** It doesn't just
> print a winner — **it tells you when you're not entitled to claim one.**
>
> Note the wording: `"Highest score"`, **not** `"Best"`. That distinction is deliberate and
> correct. Highest score ≠ best config when the intervals overlap.

### Cleanup

```python
with db() as cur:
    for t in sweep_tables:
        cur.execute(f"DROP TABLE IF EXISTS {t}")
print(f"\nDropped {len(sweep_tables)} temporary sweep table(s). "
      f"Production tables ({DOC_TABLE}, {CHUNK_TABLE}) untouched.")
```

**Only the throwaway `rag_sweep_*` tables are dropped.** **`rag_documents` and `rag_chunks` are
explicitly untouched** — contrast v1.1's Step 12, which dropped the very table it had just crowned
the winner, so `ask()` broke immediately afterwards.

**Comment this out** if you want to inspect a sweep table by hand afterwards.

---

## 🔹 Cell 40 (markdown) — Step 14: Production checklist

What changes between this notebook and a deployed service.

### Data & indexing
- **Re-index on write, not on a schedule.** Call `upsert_documents()` from whatever writes your
  source data, then `ingest()`. The content-hash check makes it **cheap and safe to call often.**
- **Bulk backfill:** load rows first, build HNSW last. Raise `maintenance_work_mem`
  (e.g. `SET maintenance_work_mem = '2GB'`) for the build — **an HNSW build that spills to disk is
  dramatically slower.**
- **`ANALYZE` after large loads** so the planner keeps choosing the index.
- Watch re-ingest churn: `pg_stat_user_tables.n_dead_tup`, and tune autovacuum on `rag_chunks`.
- **⚠️ Metadata filters + ANN interact badly at scale.** HNSW filters **after** the scan, so a
  narrow filter can return **fewer rows than you asked for.** On pgvector ≥ 0.8 set
  `hnsw.iterative_scan = relaxed_order`; otherwise raise `hnsw.ef_search` when filtering, or use a
  partial index per high-traffic filter value. *This one surprises people in production.*

### Retrieval tuning, in the order that pays

| # | Knob | Cost to change |
|---|---|---|
| 1 | **`hnsw.ef_search`** — the recall/latency dial. Start at 100, raise until recall plateaus. | **Free** |
| 2 | `CHUNK_TOKENS` / `CHUNK_OVERLAP_TOKENS` | Re-ingest; **measure with Step 11 before and after** |
| 3 | `RRF_K`, `W_VECTOR`/`W_TEXT` — push lexical weight up if users search by ID/SKU | Free |
| 4 | `m` / `ef_construction` | **Index rebuild.** Only after 1–3 are exhausted |

**The ordering is the advice.** Try the free dial before the expensive one.

### Reliability
- Swap the LLM reranker for the **Vertex AI Ranking API** (cheaper, faster, purpose-built).
  **`rerank()` is the seam.**
- **Vertex embedding quota is the usual first bottleneck.** `_with_retry` absorbs bursts; for
  backfills add a bounded worker pool and a rate limiter.
- Set request timeouts and a **circuit breaker** around Vertex. **A retrieval-only degraded mode
  (return the sources, skip generation) beats a 30-second hang.**
- Keep the pool smaller than Cloud SQL's `max_connections` ÷ your instance count.

### Observability — log every request

`ask()` already returns per-stage timings and the retrieved `doc_id`s. **Persist them.** Four
signals tell you retrieval is degrading **before users complain**:

| Signal | What a change means |
|---|---|
| `refusal rate` | Spike = retrieval broke, or the corpus lost documents |
| `p95 search_ms` | Rising = index degrading or table outgrowing its tuning |
| `mean top-1 cosine_distance` | Rising = queries drifting away from your corpus |
| `% of answers with zero citations` | Rising = the model is going off-script |

### Correctness
- **Grow the gold set to 50–100 questions from real user logs**, and **run Steps 11 + 12 in CI.
  Treat a regression in `answer_support` as a build failure.**
- **Re-run the eval after *any* change** to chunking, embedding model, `EMBED_DIM`, or the prompt.
- **Embedding models are not interchangeable.** Changing `EMBED_MODEL` or `EMBED_DIM` invalidates
  every stored vector. Re-embed with `ingest(force=True)` — **a mixed-model table returns
  nonsense.**

### Security
- **Rotate the Postgres password that was hardcoded in v1.1** if that file ever left your machine.
- Use **IAM database authentication or Secret Manager**; never a literal in a notebook.
- **If documents are access-controlled, filter by permission IN THE SQL** (`metadata @> ...` on a
  tenant/ACL key) — **never by post-filtering the LLM's output.** By then the model has already
  read the restricted text, and you're one prompt-injection away from it leaking.

### Worth adding next
- **Query rewriting** — resolve pronouns and follow-ups against conversation history *before*
  retrieval. ("What about theirs?" is unembeddable as-is.)
- **Full contextual retrieval** — an LLM-written 1–2 sentence situating summary per chunk (a paid
  upgrade over the free header used here).
- **Parent-document retrieval** — search small chunks, feed the surrounding **section** to the LLM.
  Best of both: precise retrieval, complete context.
- **Semantic caching** on the query embedding for repeated questions.

---

## 🔹 Cell 41 (markdown) — Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `column cannot have more than 2000 dimensions` | `EMBED_DIM > 2000`. Keep 1536 (or 768). **This is what forced v1.1 into brute-force scans.** |
| `403 PERMISSION_DENIED` from Vertex | `gcloud auth application-default login`, and enable the Vertex AI API on the project |
| `429 RESOURCE_EXHAUSTED` during ingest | `_with_retry` backs off automatically; for large backfills lower `batch_size` and add a rate limiter |
| `type "vector" does not exist` | Cell 10 ran against a **different database** than `retrieve()` uses. Check `PGDATABASE` |
| `unrecognized configuration parameter "hnsw.ef_search"` | pgvector too old. Check `SELECT extversion FROM pg_extension WHERE extname='vector'` |
| Search returns nothing for an exact ID | The lexical leg needs that term in `embed_input`. Check with `SELECT embed_input FROM rag_chunks WHERE doc_id = '...'` |
| Answers right but slow | Look at `ask()["timings"]` — it is **almost always `generate_ms` or `rerank_ms`, never `search_ms`** |
| **Right document cited, wrong number stated** | Rules 2/3 in `SYSTEM_RULES` — **the exact v1.1 failure.** Confirm the header is in `embed_input`, and raise `RERANK_TOP_N` so competing values are both in context |
| **`recall` high but `support` low** | Chunks too small, or the answer straddles a boundary. Raise `CHUNK_TOKENS` / `CHUNK_OVERLAP_TOKENS` and re-ingest |
| Everything got worse after a model change | Stored vectors are stale. `ingest(force=True)` |

### Teardown

```python
# with db() as cur:
#     cur.execute(f"DROP TABLE IF EXISTS {CHUNK_TABLE}, {DOC_TABLE} CASCADE")
```

**Commented out by default** — unlike v1.1's Step 12, **which dropped the very table it had just
crowned the winner.** A small detail that says a lot about the difference between the two
notebooks.

---

# PART C — Defending the design

Likely questions and honest answers.

**"Why Postgres/pgvector instead of Pinecone / Weaviate / a dedicated vector DB?"**
The vectors live next to the relational data — same transactions, same backups, same access
control, same SQL. No second system to operate, no sync pipeline to drift. **Cell 30's
`filters={"category": "Dairy"}` demo is the concrete argument**: you can filter on business data
*inside* the ANN scan. Dedicated vector DBs win at very large scale (hundreds of millions of
vectors) and offer richer ANN tuning; for a bid/RFP corpus, Postgres is the lower-risk choice.

**"Why one chunker at serving time when you tested five?"**
Because selection and serving are different activities. Serving 27 pipelines would multiply
storage, cost and latency by 27 to produce 27 different answers with no way to choose at request
time. **Step 13 is the selection experiment; its winner is what's deployed.** (See Cell 2.)

**"Why `gemini-embedding-001`?"**
Three reasons, in order: (1) **measured** — it won the benchmark, MRR 0.90 vs 0.70; (2) **MTEB** —
supporting public evidence, not a substitute for your own data; (3) **operational** — current
generation, multilingual, and Matryoshka lets you re-trade accuracy for storage without
re-embedding.

**"Why 1536 and not 3072?"**
pgvector's HNSW caps at 2000 dimensions. At 3072 you get **no index at all** — every query becomes
a full table scan. **1536 buys the index back at essentially no quality cost**, and halves
storage. Step 13 measures exactly that trade-off on your data.

**"How confident are you in this ranking?"**
With 13 questions: **directional, not statistical.** The sweep prints its own confidence intervals
and will tell you which configs it cannot separate. Expanding to 50–100 questions from real user
logs is the next step.

**"Why hybrid search — isn't semantic search enough?"**
No, and Cell 33's `text` / `vector` / `hybrid` comparison measures it rather than asserting it.
Embeddings are structurally weak on exact identifiers — `BID-2024-089` has almost no *meaning* to
compress. Cell 30's exact-ID query is the demonstration.

**"How do you know the contextual header helps?"**
The **ablation** in Step 13: identical config, header removed. If it doesn't help, the table will
say so.

**"What happens when a document is updated?"**
`upsert_documents()` updates `content_hash`; `ingest()` sees it differs from `indexed_hash` and
re-embeds **only that document**, in **one transaction**. Everything else is skipped.

**"What are this system's current limitations?"** *(Say these before you're asked.)*
1. Gold set is 13 questions — needs 50–100.
2. Demo corpus is 6 short documents; the chunker never actually fires on it.
3. LLM reranker should be the Vertex Ranking API in production.
4. No query rewriting for follow-up/pronoun questions.
5. Metadata filter + HNSW interaction needs `iterative_scan` at scale (Cell 40).
6. Table/column names are f-string-interpolated in Cell 19's importer — fine for config, not for
   user input.
7. The sentence regex is a heuristic and can be fooled by decimals and abbreviations.
8. No access-control filtering yet — the hook exists (`metadata @>`), the policy doesn't.

---

# PART D — Quick reference

## Execution order & dependencies

```
Cell  4  install
Cell  6  config       ──▶ PROJECT_ID, EMBED_MODEL, EMBED_DIM, DB_CONFIG, all tuning constants
Cell  8  pool         ──▶ POOL, db()
Cell 10  schema       ──▶ rag_documents, rag_chunks, GIN indexes    [needs: db]
Cell 12  embeddings   ──▶ genai_client, embed_texts(), embed_query(), _with_retry()
Cell 14  chunking     ──▶ split_text(), contextual_header(), build_chunks()
Cell 16  doc I/O      ──▶ upsert_documents(), delete_documents(), content_hash()
Cell 17  seed data    ──▶ SOURCE_DOCS written to rag_documents      ← YOUR DATA GOES HERE
Cell 19  importer     ──▶ import_from_existing_table()   (optional)
Cell 21  ingest       ──▶ ingest(), ensure_vector_index()  ⏱ SLOW / COSTS $  [needs: 12,14,16]
Cell 23  retrieval    ──▶ HYBRID_SQL, retrieve()                    [needs: 8,12]
Cell 25  rerank       ──▶ rerank()                                  [needs: 12]
Cell 27  generation   ──▶ ask(), show(), SYSTEM_RULES, REFUSAL      [needs: 23,25]
Cell 29  regression test                                            [needs: 27]
Cell 30  capability demos                                           [needs: 27]
Cell 32  gold set     ──▶ GOLD (13 questions)                       ← YOUR QUESTIONS GO HERE
Cell 33  retrieval eval ──▶ recall / MRR / nDCG / support + timings [needs: 23,25,32]
Cell 35  answer eval  ──▶ LLM-as-judge + refusal check   ⏱ COSTS $  [needs: 27,32]
Cell 37  sweep configs ──▶ CHUNKERS, SWEEP_CONFIGS
Cell 38  sweep machinery ──▶ sweep_ingest(), sweep_eval(), wilson_ci()
Cell 39  run sweep    ──▶ leaderboard + confidence intervals  ⏱⏱ SLOWEST / COSTS $$$
```

## Tables

| Table | Created by | Contents | Lifetime |
|---|---|---|---|
| `rag_documents` | Cell 10 | Raw documents + JSONB metadata + hashes, 1 row/doc | **Permanent** |
| `rag_chunks` | Cell 10 | Chunks + `embedding` + `tsv`, many rows/doc | **Permanent** |
| `rag_sweep_*` | Cell 38 | One per sweep config | Dropped in Cell 39 |

## The knobs

| Setting | Cell | Effect | Change cost |
|---|---|---|---|
| `EMBED_MODEL`, `EMBED_DIM` | 6 | Which embedder, what vector size | **`ingest(force=True)`** |
| `CHUNK_TOKENS`, `CHUNK_OVERLAP_TOKENS` | 6 | Chunk granularity | **`ingest(force=True)`** |
| `HNSW_M`, `HNSW_EF_CONSTRUCTION` | 6 | Index build quality | **Index rebuild** |
| `HNSW_EF_SEARCH` | 6 | **Recall/latency dial** | **Free** |
| `RRF_K`, `W_VECTOR`, `W_TEXT` | 6 | Fusion balance | Free |
| `VECTOR_CANDIDATES`, `TEXT_CANDIDATES` | 6 | First-stage net width | Free |
| `RERANK_CANDIDATES`, `RERANK_TOP_N` | 6 | Second-stage narrowing | Free |
| `MAX_COSINE_DISTANCE` | 6 | Refusal threshold | Free |
| `HEADER_FIELDS` | 14 | What goes in the header | **`ingest(force=True)`** |
| `SOURCE_DOCS` | 17 | **Your documents** | `ingest()` |
| `SYSTEM_RULES` | 27 | Answer style & strictness | Free |
| `GOLD` | 32 | **Your evaluation questions** | Free |
| `SWEEP_CONFIGS` | 37 | What the sweep compares | — |

## pgvector cheat sheet

```sql
CREATE EXTENSION IF NOT EXISTS vector;

embedding vector(1536)                       -- column type

a <=> b   -- cosine distance      ← this notebook
a <-> b   -- Euclidean / L2
a <#> b   -- negative inner product

CREATE INDEX ON t USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
SET LOCAL hnsw.ef_search = 100;              -- per-transaction recall/latency dial

SELECT content, embedding <=> %s::vector AS d
FROM t ORDER BY d LIMIT 4;                   -- k-NN search
```

**Dimension limit: HNSW and IVFFlat cap at 2000.** Above that you get no index — an exact scan.

## Postgres full-text cheat sheet

```sql
to_tsvector('english', text)              -- text -> searchable tokens (stemmed, lowercased)
websearch_to_tsquery('english', query)    -- user input -> query (handles quotes, OR)
tsv @@ q                                  -- "does this text match this query?"
ts_rank_cd(tsv, q)                        -- relevance score, cover-density aware
CREATE INDEX ON t USING gin (tsv);        -- makes it fast
```

## Your first-week checklist

1. ✅ Set `PGPASSWORD` and the other env vars; **run the notebook as-is** end to end.
2. ✅ Read Cell 14's demo output — see the contextual header physically attached.
3. ✅ Read Cell 23's three-mode output — see which leg found what.
4. ✅ **Run Cell 21 twice.** The second run says "nothing to ingest". That's incremental indexing.
5. ✅ Call `ask(..., verbose=True)` once and read the assembled prompt.
6. ✅ Replace `SOURCE_DOCS` (Cell 17) with 10–20 **real** bid/RFP documents.
7. ✅ **Write 50+ real gold questions** in Cell 32. *This is the highest-value work available to
   you* — the entire benchmark's credibility rests on it. Ask the people who actually search these
   documents what they ask.
8. ✅ Re-run Cells 33 and 35. **Now the numbers mean something.**
9. ✅ Run Cell 39 (start with `SWEEP_CONFIGS[:4]` to control cost). Keep the output for the deck.
10. ✅ Try a question the documents can't answer; confirm it refuses.

---

*Cell-by-cell guide for `rag_production_v2.ipynb`. Companion to `rag_production_v2_guide.md`.*
