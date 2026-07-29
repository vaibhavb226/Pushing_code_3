# Production RAG on Cloud SQL — Detailed Guide (v2)

**Notebook:** `rag_production_v2.ipynb`
**Replaces:** `rag_end_to_end_psycopg2_new_v1.1.ipynb` and `rag_end_to_end_guide.md`

This guide explains, in plain language, every step of the v2 notebook: what it does, the code
behind it, why it's there, what to change, and how to read the results. It is written for
someone new to RAG, and it includes the material you need to **defend the design decisions**
to a client team.

---

## 0. Why there is a v2

While testing v1.1 you asked:

> *"What was the Customer Satisfaction level for supplier performance review?"*

and got back **"Overall district satisfaction was 8.5/10"** — when the document actually says
**"Customer satisfaction: 4.8/5 stars"** for Acme Foods. The `8.5/10` figure is the
district-wide average, a completely different number.

That was not bad luck. Four separate design gaps caused it, and every one gets worse on real
data.

| # | Defect in v1.1 | Consequence |
|---|---|---|
| 1 | **Chunks were stored naked.** Step 8 saved only `doc_id`, `chunk_index`, `content`. All the metadata written in Step 7a — supplier, category, date, document type — was discarded before embedding. | A chunk reading *"Customer satisfaction: 4.8/5 stars"* has no supplier attached. Neither the search nor the LLM can tell whose score it is. |
| 2 | **Pure meaning-based retrieval, no keyword search.** | Exact strings — `BID-2024-089`, `4.8/5`, `Net-45`, a SKU — are exactly where embeddings are weakest. ID lookups fail silently. |
| 3 | **The prompt had no disambiguation or citation rules.** | The document holds *two* satisfaction figures. The model picked one, unlabelled and uncited, and you had no way to see it chose wrong. |
| 4 | **No reranking, `TOP_K = 3`.** | Whatever the vector search returned *was* the answer. One weak embedding equals one wrong answer. |

### Two more issues that would have hurt at deployment

**The benchmark was measuring noise.** v1.1's latency column included the ~3.2 second Vertex
embedding call, which completely swamped the actual Postgres search (well under a millisecond
on 20 rows). Every index type therefore scored ~3.2 s, the comparison carried no signal, and
the leaderboard crowned **`none` — a brute-force sequential scan** — the winner. Deploying that
means every query scans the entire table.

The root cause is worth understanding: `gemini-embedding-001` produces 3072-dimension vectors,
but pgvector's `hnsw` and `ivfflat` indexes refuse anything above **2000 dimensions**. So every
indexed combination using that model was skipped, leaving only the unindexed one standing.
v2 fixes this by truncating to 1536 dimensions (see section 5, Step 2).

**One gold-set label was wrong.** `"Which supplier offered an early payment discount?"` was
labelled as `globex-dairy-2024` only — but Sysco (1%) and US Foods (2%) also offer one. The
benchmark was scoring correct retrievals as failures.

**Credit where due:** v1.1's *embedding model* comparison was sound — `gemini-embedding-001`
genuinely won on MRR (0.90 versus 0.70). Only the *index* comparison was broken. v2 simply
uses the model that won.

> **Security — act on this.** v1.1 contains a live Postgres password in cleartext in cell 2.
> If that file has been emailed, shared, or committed anywhere, **rotate that credential.**
> v2 reads it from the `PGPASSWORD` environment variable instead.

---

## 1. The big picture — what is RAG?

**RAG = Retrieval-Augmented Generation.** An LLM cannot know what is in *your* private
documents, and if you simply ask, it will invent something plausible. So instead of asking the
model to *remember*, you **find** the relevant text yourself, **paste it into the prompt**, and
ask the model to answer *only from that*. Retrieval does the knowing; the LLM does the writing.

```
document -> chunk -> embed -> store in Postgres (pgvector)
                                       |
question -> embed -> search -----------+-> best chunks -> Gemini -> answer + citations
```

v2 adds three things to that basic loop:

```
                    +-> vector search (meaning) --+
question -> embed --|                             +-> RRF merge -> rerank -> Gemini
                    +-> full-text search (words)--+
```

---

## 2. Glossary

| Term | Plain meaning |
|---|---|
| **Chunk** | A document cut into a bite-sized piece (a few hundred words). You retrieve chunks, not whole documents — pasting a 50-page PDF into every prompt is slow, expensive, and *makes answers worse* by burying the relevant sentence. |
| **Embedding** | A list of numbers (here 1536 of them) representing a piece of text's *meaning*. Texts that mean similar things get similar numbers. |
| **Dimension** | How long that list is. Fixed per model, and fixed per database column. |
| **Vector** | The embedding once stored in the database. `vector(1536)` is a pgvector column type. |
| **Cosine distance (`<=>`)** | How far apart two embeddings are. **0 = identical meaning**, larger = less related. This is why searches sort *ascending* — nearest first. |
| **pgvector** | The Postgres extension providing the `vector` type and the `<=>` operator. |
| **ANN / HNSW** | *Approximate Nearest Neighbour.* Comparing a question against 10 million rows is slow, so HNSW builds a graph that finds near-certain best matches while checking only a few hundred rows. "Approximate" means it occasionally misses a borderline match, in exchange for being roughly 1000× faster. |
| **Full-text search / `tsvector`** | Classic **keyword** matching, built into Postgres. Knows nothing about meaning, but nails exact strings like `BID-2024-089`. |
| **Hybrid search** | Running the meaning search *and* the keyword search, then merging the two ranked lists. Each covers the other's blind spot. |
| **RRF** | *Reciprocal Rank Fusion* — the merging formula. Combines lists by **position** (1st, 2nd, 3rd) rather than by score, so two incompatible scoring scales never have to be reconciled. |
| **Reranker** | A second, smarter, slower pass. Stage 1 grabs ~40 roughly-relevant candidates fast; the reranker reads them properly and keeps the best 4. |
| **`task_type`** | Tells the embedding model whether text is a *document being stored* (`RETRIEVAL_DOCUMENT`) or a *question doing the searching* (`RETRIEVAL_QUERY`). Getting this wrong silently costs accuracy — no error, just worse results. |
| **Contextual header** | A short "where this came from" banner (title, supplier, category, date) prepended to each chunk *before embedding*. The main fix for the satisfaction bug. |
| **Gold set** | Your test questions, each labelled with the document(s) that should answer it. |
| **recall@k** | Of your test questions, how often a correct document appeared in the top *k*. "Did we find it at all?" |
| **MRR** | *Mean Reciprocal Rank* — how **high** it ranked. 1st = 1.0, 2nd = 0.5, 3rd = 0.33. "Did we find it *first*?" |
| **nDCG** | Like MRR, but handles questions with several correct documents. |
| **answer-support** | Did a retrieved chunk actually contain the answer *text*? Stricter than recall — see section 7. |
| **Ablation** | Re-running an experiment with one component removed, to measure that component's contribution. |
| **Hallucination** | The model stating something the sources do not support. The Step 10 prompt exists to prevent this. |

---

## 3. Prerequisites

- A **Cloud SQL for PostgreSQL** instance you can reach (directly, or via the Cloud SQL Auth
  Proxy running locally so `host="localhost"` works).
- **pgvector** available on that instance. For `hnsw.iterative_scan` (mentioned in section 8)
  you need pgvector 0.8 or later; everything else works on older versions.
- **Vertex AI** access, authenticated with Application Default Credentials:
  run `gcloud auth application-default login` once, and enable the Vertex AI API.
- Python packages: `google-genai`, `psycopg2-binary`, `tiktoken` (installed by Step 1).
- Environment variables set before launching the notebook:

```bash
export PGHOST=10.151.179.4 PGDATABASE=postgres PGUSER=postgres PGPASSWORD='...'
export GCP_PROJECT_ID=div-aais-rfpiq-usc1-uat GCP_LOCATION=us-central1
```

---

## 4. What changed from v1.1

| Area | v1.1 | v2 |
|---|---|---|
| Purpose | Method bake-off (27 pipelines) | Production pipeline (1), plus an offline sweep |
| Chunk storage | `doc_id`, `chunk_index`, `content` | Adds `embed_input` (with header), `metadata`, `token_count`, generated `tsvector` |
| Retrieval | Vector only | **Hybrid** vector + full-text, merged with RRF |
| Reranking | None | Second-stage precision pass |
| Prompt | "Answer from the context" | Citations, multi-entity rule, entity-match rule, fixed refusal string |
| Index | Compared; brute force "won" | HNSW at 1536 dims, built after bulk load |
| Re-indexing | Drop and rebuild everything, every run | Content-hash incremental — only changed documents |
| Connections | New connection per query | Pooled |
| Secrets | Password hardcoded | From environment |
| Metrics | recall, MRR, misleading latency | recall, MRR, nDCG, **answer-support**, refusal correctness, **per-stage** latency, confidence intervals |
| Cleanup | Dropped the winning table | Teardown is commented out by default |

---

## 5. Step-by-step walkthrough

### Step 1 — Install

```python
%pip install -q google-genai "psycopg2-binary>=2.9" tiktoken
```

`google-genai` talks to Vertex AI (embeddings and Gemini), `psycopg2-binary` is the Postgres
driver, `tiktoken` counts tokens so chunks can be budgeted. If pip says "restart kernel," do it.

### Step 2 — Configuration

The only cell you must edit. Key settings:

```python
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM   = 1536
CHUNK_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 60
VECTOR_CANDIDATES = 40
TEXT_CANDIDATES   = 40
RERANK_TOP_N = 4
HNSW_EF_SEARCH = 100
```

**Why `EMBED_DIM = 1536`** — this is the fix for v1.1's brute-force problem.
`gemini-embedding-001` emits 3072 dimensions natively, which exceeds pgvector's 2000-dimension
ceiling for `hnsw` and `ivfflat`. The model is **Matryoshka (MRL)** trained, meaning you can cut
the vector short and it still works. Truncating to 1536 keeps essentially all retrieval quality,
halves storage, and **allows the index**. The notebook re-normalises after truncating (a
truncated vector is no longer unit length).

**The password** is read from `PGPASSWORD`, with an interactive `getpass` prompt as fallback.
Never hardcode it.

### Step 3 — Connection pool

v1.1 opened a fresh TCP + TLS + authentication handshake for *every query*. Under load that
costs tens of milliseconds per call and exhausts Cloud SQL's connection limit. v2 uses a
`ThreadedConnectionPool` (1–8 connections) wrapped in a `db()` helper:

```python
with db(dict_rows=True) as cur:
    cur.execute("SELECT ...")
```

The helper commits on success, rolls back on exception, and always returns the connection to
the pool — even on a crash. Keep `maxconn` below Cloud SQL's limit divided by your number of
app instances.

### Step 4 — Schema

Two tables instead of one-table-per-experiment.

**`rag_documents`** — your raw text plus:

- `metadata jsonb` — supplier, category, dates, and anything else. One JSONB column instead of
  ten fixed columns means adding a field later needs no migration, and it is index-backed.
- `content_hash` — a fingerprint of what the document says *now*.
- `indexed_hash` — a fingerprint of what was last *embedded*. When these differ, the document
  needs re-indexing. This pair is what makes Step 7c incremental.

**`rag_chunks`** — one row per chunk:

- `content` — clean text, shown to the LLM and the user.
- `embed_input` — content **plus its contextual header**; this is what was embedded and what
  keyword search reads. Storing both separately is what lets the header improve retrieval
  without polluting the answer.
- `metadata jsonb` — copied down from the document so you can filter *during* the search.
- `tsv` — a **generated** column. Postgres computes and maintains it, so it can never drift out
  of sync with the text.
- `ON DELETE CASCADE` — deleting a document automatically deletes its chunks, so you can never
  leave orphaned vectors that still appear in results.

Indexes: **GIN** on `tsv` (keyword search), **GIN** on `metadata` (filters), **B-tree** on
`doc_id` (fast re-ingest). The **HNSW vector index is created later**, in Step 7c — building it
on an empty table and then inserting is much slower than loading first and indexing once.

### Step 5 — Embeddings

`embed_texts(texts, task_type)` with four things v1.1 lacked:

1. **Retry with exponential backoff and jitter.** Vertex returns `429 RESOURCE_EXHAUSTED` under
   load. Waits 1s, 2s, 4s, 8s plus a random fraction so parallel workers don't retry in lockstep.
   Permanent errors (403, bad model name) re-raise immediately instead of wasting five attempts.
2. **L2 normalisation** — rescales each vector to length 1.0 without changing its direction.
   Necessary because truncating an MRL embedding leaves it slightly short.
3. **Input truncation** — one oversized chunk would otherwise fail an entire batch.
4. **A query cache** (`lru_cache`) — saves ~200 ms on repeated questions.

**The golden rule, unchanged from v1.1:** documents and questions must be embedded by the
*same* model, with the correct `task_type` for each role.

### Step 6 — Chunking and contextual headers

v2 uses **one** chunker (see section 6 for why). It is *recursive*: respect the largest natural
boundary that fits, and only cut harder when you must.

> paragraph -> sentence -> hard token split, then pack greedily to a token budget with
> sentence-aligned overlap.

This is strictly better than all three of v1.1's: `fixed_400` cuts mid-word (and mid-number —
`2024-05-01` becomes `202` + `4-05-01`), `sentence` uses a character budget that doesn't map to
token limits, and `token_256` cuts mid-sentence.

**Overlap** repeats ~60 tokens between neighbours, as *whole sentences*, so a fact sitting on a
boundary still appears intact in at least one chunk.

**The contextual header** is the main fix for your bug. Before embedding, each chunk is prefixed:

```
Document: Q1 2024 Supplier Performance Review | ID: performance-review-2024 |
Supplier Name: Riverside USD | Category: Performance Review | Bid Date: 2024-04-01

Q1 2024 Performance Review Summary: Acme Foods maintained 98% on-time delivery ...
```

The chunk is now findable by *"performance review"*, *"Riverside"* or *"Q1 2024"* even when the
chunk body never repeats those words, and the LLM can see which document each fact came from.
Anthropic's published *contextual retrieval* work measures roughly a **35% reduction in
retrieval failures** from this idea alone. (Their full version generates a per-chunk LLM
summary — the obvious upgrade when budget allows; the header form here is free.)

> **Note on the demo corpus.** All six sample documents are 70–140 tokens, so at
> `CHUNK_TOKENS = 350` each becomes exactly **one** chunk and the splitter never fires. On this
> data the bug is fixed by the header, hybrid search, and the prompt — **not** by lucky chunk
> boundaries. The splitter matters once you point Step 7b at real multi-page RFPs.

### Step 7a / 7b — Loading documents

**7a** seeds the six sample documents with rich metadata. **7b** provides
`import_from_existing_table(...)` to map a table you already have into `rag_documents` — the
normal production path, one small adapter per source system.

`content_hash` covers title + content + metadata, so *any* edit — including a metadata-only edit
that changes the header — correctly marks the document for re-embedding.

### Step 7c — Incremental ingest

v1.1 dropped and rebuilt every table on every run: 27 combinations, every chunk re-embedded,
every time. Fine for six demo documents; ruinous on a real corpus.

v2 re-embeds a document **only when `content_hash` differs from `indexed_hash`**. Note the SQL
uses `IS DISTINCT FROM`, not `!=` — a plain `!=` evaluates to NULL (not true) for
never-indexed documents, which would silently skip every new document you add.

Each document's rewrite is **one transaction**: delete old chunks, insert new chunks, advance
the watermark. A crash can never leave a document half-indexed.

`ingest(force=True)` re-embeds everything. **You must do this after changing `EMBED_MODEL`,
`EMBED_DIM`, or the chunking settings** — the stored vectors were produced by different rules
and are no longer valid.

### Step 8 — Hybrid retrieval with RRF

The core upgrade. Two independent searches run in **one** SQL statement, merged by Reciprocal
Rank Fusion:

```
score(chunk) = sum over searches of  weight / (k + rank),   k = 60
```

RRF fuses on **rank**, never on raw score, so a cosine distance and a `ts_rank_cd` score never
have to be put on a common scale. That scale mismatch is what makes naive score-blending
fragile.

Why both legs earn their keep:

| Query | Meaning search alone | Keyword search alone | Hybrid |
|---|---|---|---|
| "who delivers fastest?" | Works — understands paraphrase | Fails — no shared words | Works |
| "BID-2024-089" | Weak — IDs embed poorly | Works — exact token | Works |
| "Acme customer satisfaction" | Weak | Works — header carries "Acme" | Works |

Three details that are easy to get wrong and are handled in the notebook:

- **The nearest-neighbour lookup sits in an inner subquery.** A window function such as
  `ROW_NUMBER()` at the same level as `ORDER BY embedding <=> ...` forces Postgres to rank
  *every* row, discarding the HNSW index. Ranking the already-limited 40 rows in an outer query
  keeps the index. This single detail is the difference between a millisecond and a full scan.
- **`hnsw.ef_search` is set per transaction** with `SET LOCAL`, so one expensive query cannot
  slow everyone else down.
- **Metadata filters run inside the search**, via `metadata @> '{"category":"Dairy"}'`.

Both `vec_rank` and `txt_rank` are returned, so when debugging you can see *which* search found
each chunk.

### Step 9 — Reranking

Stage 1 optimises for **recall** (cast a wide net, 40 candidates, 20 kept). Stage 2 optimises
for **precision** — a slower, smarter model reads the query against each candidate and reorders,
keeping the best 4. This is the highest-leverage single component in most production RAG
systems, and v1.1 had none.

**In production, prefer a dedicated cross-encoder** — on Google Cloud that is the
**Vertex AI Ranking API** (`semantic-ranker-default@latest`), which is purpose-built and far
cheaper and faster than an LLM call. The notebook's LLM-based reranker is the portable
equivalent so it runs with no extra service enabled; `rerank()` is a drop-in seam — replace the
body, keep the signature.

If the reranker fails or returns malformed JSON, the code keeps the RRF order rather than
failing the request. A reranker outage should degrade quality, not take retrieval down.

### Step 10 — Grounded generation

The prompt is where the wrong answer was actually produced, so it carries explicit rules:

1. **Cite every fact** with a source number — an unverifiable answer becomes impossible.
2. **List *all* matching entities** — the direct fix for `4.8/5` versus `8.5/10`. Never collapse
   distinct figures into one, and never present an aggregate as an individual's value.
3. **Answer for the entity that was asked about** — if asked about Acme and the sources only
   cover the district, say so rather than substituting.
4. **Quote figures exactly** — no rounding, no unit conversion.
5. **A fixed refusal string** — machine-detectable, so you can alarm on refusal rate.

Plus a **relevance floor**: if nothing clears `MAX_COSINE_DISTANCE` *and* no keyword matched, the
system refuses before paying for a generation call. Both conditions must hold, so a valid
exact-ID lookup (poor cosine distance, decisive keyword hit) is not wrongly refused.

`ask()` returns the answer, its sources, and per-stage timings, so every answer is auditable and
every slow request is diagnosable.

### Step 11 — Retrieval evaluation

Reports **recall@k, MRR, nDCG, answer-support**, and latency split into `embed_ms`,
`search_ms`, `rerank_ms`. It compares four configurations — text only, vector only, hybrid,
hybrid + rerank — all scored on the same final context size so the comparison is fair.

### Step 12 — Answer evaluation (LLM-as-judge)

Retrieval metrics tell you the right text was *found*. They cannot tell you the answer was
*right* — your satisfaction question is the proof. So a second harness scores generated answers
on **groundedness** (is every claim supported?), **correctness**, and **completeness** (when
several entities match, are all covered?), plus a deterministic "is the expected string actually
present" backstop and a hard refusal check on the unanswerable question.

An LLM judge is noisy on any single item. Use it to compare *configurations across a whole set*,
and keep a small human-labelled slice as ground truth.

### Step 13 — Configuration sweep

**This is the cell you run before a client meeting.** Covered in full in section 6.

### Step 14 — Production checklist

Covered in section 8.

---

## 6. Answering the client's questions

This section exists because "why did you choose X?" needs evidence, not opinion.

### 6.1 Selection versus serving

There are **two separate activities**, and conflating them is what made v1.1 confusing:

| Aspect | **Selection** (choosing) | **Serving** (running) |
|---|---|---|
| What it is | An experiment measuring which config wins | The product answering questions |
| How often | Once, offline; re-run when something changes | Every user request |
| How many pipelines | **As many as you can afford to test** | **Exactly one** — the winner |
| Lives in | Step 13 | Steps 2–12 |

So: **yes, absolutely test many chunkers and many models** — in Step 13. Then **deploy one**.
Serving 27 pipelines would multiply storage, cost, and latency by 27 to produce 27 different
answers to the same question, with no way to choose between them at request time.

### 6.2 "Why didn't you use different chunking methods?"

**The right answer is "I did — here are the numbers."**

Step 13 compares **five chunkers** on your data: all three of v1.1's (`fixed_400`,
`sentence_500`, `token_256`), reproduced verbatim so the comparison isn't rigged, plus the
recursive splitter at two sizes. Run it, keep the output, put the table in the deck.

That answer is far stronger than either *"I tested 27 combinations"* (then why is only one
deployed?) or *"I used a sensible default"* (on what basis?).

### 6.3 "Why `gemini-embedding-001` and not gecko@003 / 004 / 005?"

**First, get the lineage right.** These are **generations, not options**:

```
textembedding-gecko@001/@002/@003  ->  text-embedding-004/005  ->  gemini-embedding-001
      (legacy, 768d)                        (768d)                   (3072d, current)
```

They are successive generations of the same product line. Choosing `gecko@003` over
`gemini-embedding-001` is like specifying an older model year — you would need a *reason*, such
as an existing index you cannot afford to re-embed.

Note there is **no Vertex `text-embedding-003`** — `text-embedding-3-small`/`large` are OpenAI's,
and Google's `@003` is `textembedding-gecko@003`. Getting this wrong in a meeting is avoidable.
Check the **Vertex AI model lifecycle page** for current deprecation and retirement dates before
quoting any of them; "it is being retired in N months" is itself a decisive argument.

**Second, give three reasons in this order** — only the first is really about your data:

1. **Measured** — it won your own benchmark. v1.1's model comparison was sound:
   `gemini-embedding-001` scored MRR **0.90** against **0.70** for `text-embedding-004`/`005`.
   (Do not quote v1.1's *index* comparison — that half was measuring noise.)
2. **Published benchmarks** — it ranks at or near the top of **MTEB**, the standard public
   embedding leaderboard. Supporting evidence only, never a substitute for testing on your own
   data: MTEB is general web and academic text, not procurement documents.
3. **Operational** — current generation (longest support runway), multilingual, and **Matryoshka
   (MRL)**, which lets you truncate 3072 → 1536 → 768 and trade accuracy for storage *without
   re-embedding*. That flexibility is a real architectural advantage the older models lack.

### 6.4 The hard constraint behind all of this

> **Vectors from different embedding models are not comparable.** `text-embedding-004` and
> `gemini-embedding-001` place "romaine lettuce" at completely different coordinates. Comparing
> one to the other returns a meaningless number — not an error, just silent garbage.

Consequences:

- **One table = one model, permanently.** That is why v1.1 built a separate table per model — it
  had no choice — and why Step 13 does the same for its throwaway tables.
- **Changing `EMBED_MODEL` or `EMBED_DIM` invalidates every stored vector.** You must re-embed
  the whole corpus. It is a migration, not a config tweak — which is precisely why you run
  Step 13 *before* you commit, not after.

### 6.5 The question that will actually catch you out

> *"How many questions did you evaluate on?"*

You tested on **2–3 documents**. Be clear about what that can and cannot support:

- With **13 questions**, one question flipping changes recall by about **8 percentage points**.
  With 5 questions, **20 points**. The real difference between two decent configurations is
  often **2–5 points** — *smaller than your measurement noise*.
- That is exactly why v1.1's leaderboard was full of ties (`0.80`, `0.80`, `0.80`). It was not
  finding the configurations equal; it **lacked the power to tell them apart**.
- **Rule of thumb: ~50 questions minimum to rank configurations, 100+ to trust a small gap.**
  They do not need to be hard to write — pull them from real user queries, or have two
  colleagues write 25 each from documents they know.

Step 13 prints a **95% confidence interval** (Wilson score) beside every result, and explicitly
lists which configurations are *statistically indistinguishable* from the winner. Verified
numbers: at **n = 12 the interval is about ±25 points**; at **n = 100 it narrows to ±10**.

**What to say if asked before you have that:**

> *"Current results are directional, from a 13-question pilot set on 6 documents. I'm expanding
> to 50+ questions drawn from real queries before we lock the configuration."*

That is a professional answer. Presenting 5-question results as conclusive is what damages
credibility.

### 6.6 What Step 13 actually sweeps

| Dimension varied | Configurations | Question it answers |
|---|---|---|
| Chunker | `fixed_400`, `sentence_500`, `token_256`, `recursive_350`, `recursive_600` | "Why this chunking method?" |
| Embedding model | `text-embedding-004`, `text-embedding-005`, `gemini-embedding-001` | "Why not 004 or 005?" |
| Dimension | 3072, 1536, 768 | "Can we halve storage?" |
| Contextual header | on / **off (ablation)** | "Does the header actually help?" |

Each configuration gets its **own throwaway table** (`rag_sweep_*`), because vectors of
different models and dimensions cannot share a column. They are dropped at the end — unlike
v1.1, which dropped the table it had just crowned the winner.

The report also prints **chunk count** and **vector storage in MB** per configuration, so you
can put cost alongside quality.

> **Cost warning.** The sweep re-embeds your whole corpus once per configuration. Six demo
> documents is pennies; 10,000 real documents × 10 configurations is a real bill. Start with
> `SWEEP_CONFIGS[:4]`, and sweep on a representative *sample* of a large corpus.

---

## 7. How to read the evaluation output

**Rank by `answer_support`, not `recall`.** This is the lesson from your bug. Recall asks "was
the right *document* retrieved?" — and for your satisfaction question, it *was*. v1.1's metrics
would have scored that wrong answer a perfect **1.0**. `answer_support` asks the stricter
question: "did a retrieved chunk actually contain the answer *text*?"

Then:

1. **Check the confidence interval before believing a gap.** If two configurations' intervals
   overlap, you have not shown one is better. Say so.
2. **Break genuine ties by cost.** Fewer dimensions means less storage and faster search; 768
   costs half of 1536.
3. **Look at `search_ms`, not total latency.** It is the only column reflecting database work —
   the mistake that invalidated v1.1's index comparison. If answers are slow, the cause is
   almost always `generate_ms` or `rerank_ms`.
4. **Re-run after any change** to chunking, embedding model, dimension, or prompt.
5. **Treat a regression in `answer_support` as a build failure** once this is in CI.

---

## 8. Production checklist

### Data and indexing

- **Re-index on write, not on a schedule.** Call `upsert_documents()` from whatever writes your
  source data, then `ingest()`. The content-hash check makes it cheap to call often.
- **Bulk backfill:** load rows first, build HNSW last. Raise `maintenance_work_mem`
  (e.g. `SET maintenance_work_mem = '2GB'`) — an HNSW build that spills to disk is dramatically
  slower.
- **`ANALYZE` after large loads** so the planner keeps choosing the index.
- Watch re-ingest churn via `pg_stat_user_tables.n_dead_tup`; tune autovacuum on `rag_chunks`.
- **Metadata filters and ANN interact badly at scale.** HNSW filters *after* the scan, so a
  narrow filter can return fewer rows than requested. On pgvector 0.8+ set
  `hnsw.iterative_scan = relaxed_order`; otherwise raise `hnsw.ef_search` when filtering, or use
  a partial index per high-traffic filter value.

### Retrieval tuning, in the order that pays

1. **`hnsw.ef_search`** — the recall/latency dial. Start at 100, raise until recall plateaus.
   Free to change, no rebuild.
2. **`CHUNK_TOKENS` / `CHUNK_OVERLAP_TOKENS`** — needs a re-ingest. Measure before and after.
3. **`RRF_K`, `W_VECTOR` / `W_TEXT`** — raise the keyword weight if users search by ID or SKU.
4. **`m` / `ef_construction`** — needs an index rebuild. Only after 1–3 are exhausted.

### Reliability

- Swap the LLM reranker for the **Vertex AI Ranking API**. `rerank()` is the seam.
- Vertex embedding quota is the usual first bottleneck. For backfills add a bounded worker pool
  and a rate limiter.
- Set request timeouts and a circuit breaker around Vertex. A retrieval-only degraded mode
  (return sources, skip generation) beats a 30-second hang.
- Keep the connection pool smaller than Cloud SQL's `max_connections` divided by instance count.

### Observability

`ask()` returns per-stage timings and the retrieved `doc_id`s. **Persist them.** Four signals
tell you retrieval is degrading before users complain:

- refusal rate
- p95 `search_ms`
- mean top-1 `cosine_distance`
- percentage of answers containing zero citations

### Security

- **Rotate the Postgres password hardcoded in v1.1** if that file ever left your machine.
- Use IAM database authentication or Secret Manager.
- If documents are access-controlled, filter by permission **in the SQL** (`metadata @> ...` on a
  tenant or ACL key) — never by post-filtering the LLM's output.

### Worth adding next

- **Query rewriting** — resolve pronouns and follow-ups against conversation history before
  retrieval.
- **Full contextual retrieval** — an LLM-written one-or-two-sentence situating summary per chunk.
- **Parent-document retrieval** — search small chunks, feed the surrounding section to the LLM.
- **Semantic caching** on the query embedding.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `column cannot have more than 2000 dimensions` | `EMBED_DIM > 2000`. Keep 1536 or 768. This is what forced v1.1 into brute-force scans. |
| `403 PERMISSION_DENIED` from Vertex | `gcloud auth application-default login`, and enable the Vertex AI API. |
| `429 RESOURCE_EXHAUSTED` during ingest | Retries back off automatically; for large backfills lower `batch_size` and add a rate limiter. |
| `type "vector" does not exist` | Step 4 ran against a different database than the searches use. Check `PGDATABASE`. |
| `unrecognized configuration parameter "hnsw.ef_search"` | pgvector too old, or the extension is not loaded. Check `SELECT extversion FROM pg_extension WHERE extname='vector'`. |
| Search returns nothing for an exact ID | The keyword leg needs that term in `embed_input`. Check with `SELECT embed_input FROM rag_chunks WHERE doc_id = '...'`. |
| Answers are right but slow | Check `ask()["timings"]` — almost always `generate_ms` or `rerank_ms`, never `search_ms`. |
| Right document cited, wrong number stated | Rules 2 and 3 in `SYSTEM_RULES` — the exact v1.1 failure. Confirm the header is in `embed_input`, and raise `RERANK_TOP_N` so competing values are both in context. |
| `recall` high but `answer_support` low | Chunks too small, or the answer straddles a boundary. Raise `CHUNK_TOKENS` or `CHUNK_OVERLAP_TOKENS` and re-ingest. |
| Everything got worse after a model change | Stored vectors are stale. Run `ingest(force=True)`. |
| Sweep configurations all tie | Expected with a small gold set — see section 6.5. Grow the gold set. |

---

## 10. Files

| File | Role |
|---|---|
| `rag_production_v2.ipynb` | The v2 notebook this guide documents |
| `rag_production_v2_guide.md` | This guide (Markdown) |
| `rag_production_v2_guide.docx` | This guide (Word) |
| `rag_end_to_end_psycopg2_new_v1.1.ipynb` | The previous notebook — **contains a cleartext password; rotate it** |
| `rag_end_to_end_guide.md` / `.docx` | The previous guide |

---

## 11. Your next step

The highest-value action is unglamorous: **get to 50 evaluation questions** drawn from real user
queries, and add them to `GOLD` in Step 11.

Nothing else in this notebook moves the needle as much. It is what turns Step 13 from
"directional" into "conclusive," it is what makes a regression in CI meaningful, and it is the
one thing the notebook cannot do for you.
