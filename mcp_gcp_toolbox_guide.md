# MCP SQL Query Expert (GCP-native) — Complete Guide

**Notebook:** `backend_py/mcp_gcp_toolbox_query_expert_v1.ipynb`
**Purpose:** Answer natural-language questions about the RFP-AIQ analytics database and return a
one-line human answer — using an **MCP (Model Context Protocol)** architecture instead of the
in-process `services/sql_expert` pipeline. Built so you can run both side by side and decide which
to adopt.

This guide explains every part of the notebook, the two operating modes, exactly how to switch
between them, and how to extend and troubleshoot it.

---

## 1. The 30-second summary

- The notebook stands up **Google's "MCP Toolbox for Databases"** — an open-source MCP *server*
  that connects to your Cloud SQL for PostgreSQL — and drives it with **Gemini on Vertex AI**
  (no Claude anywhere).
- It runs in one of **two modes**, chosen by a single variable `MODE` in Step 1:
  - `MODE = "curated"` → **you** define fixed, parameterized SQL queries in `tools.yaml`. Safe by
    construction; the model can only call queries you wrote.
  - `MODE = "prebuilt"` → **Toolbox** auto-generates generic tools (including a free-form
    `execute_sql`) directly from the database. Zero authoring, but the model can run any SQL.
- The database connection is always pulled from the **same `.env`** the backend uses
  (`SQL_EXPERT_DB_*`). You never retype credentials.

---

## 2. What MCP is (in one picture)

MCP is a standard "plug" that lets an AI application talk to external tools/data through a uniform
protocol. Three roles:

```
  HOST (the AI app)      ->  CLIENT (built in)  ->  SERVER (exposes tools)  ->  your data
  e.g. this notebook's       toolbox-langchain      Google MCP Toolbox          Cloud SQL
  Gemini agent                                       (tools.yaml OR --prebuilt)  Postgres
```

- **Host** — the thing the user talks to. Here, the Gemini agent in the notebook.
- **Client** — the connector that speaks MCP. Here, `toolbox-langchain`'s `ToolboxClient`.
- **Server** — your code/tool that exposes capabilities over MCP. Here, Google's Toolbox binary.
- The model decides *which* tool to call and *with what arguments*; the server executes it against
  the database and returns rows.

### How this differs from the `sql_expert` pipeline

```
  This notebook (MCP):                          services/sql_expert/pipeline.py:

  Gemini (Vertex)                               LLM
     | picks a tool + params                       | generates raw SQL
     v                                             v
  MCP client (toolbox-langchain)                validators.py  (read-only, allowlist)
     | MCP over HTTP                               v
     v                                          EXPLAIN dry-run
  MCP Toolbox server  <- tools.yaml / --prebuilt   v
     | runs SQL                                  read-only execute (row cap, 15s timeout)
     v                                             v
  Cloud SQL Postgres                            one-line summary
```

The pipeline lets the LLM write **arbitrary SQL** and then defends against it with validators,
`EXPLAIN`, a row cap, and a statement timeout. The MCP `curated` mode instead only ever runs
**queries you pre-authored**, so there is nothing to validate — the SQL can't be changed by the
model. Two different safety philosophies: *generate-then-validate* vs *safe-by-construction*.

---

## 3. Prerequisites

Before running the notebook end to end you need:

1. **Python deps** (installed by Step 0):
   `toolbox-langchain`, `langchain-google-vertexai`, `langgraph`, `python-dotenv`, `requests`.
2. **`.env`** in `backend_py/` with the analytics DB settings (same ones the SQL Expert uses):
   ```
   SQL_EXPERT_DB_HOST=...
   SQL_EXPERT_DB_PORT=5432
   SQL_EXPERT_DB_NAME=...
   SQL_EXPERT_DB_USER=...
   SQL_EXPERT_DB_PASSWORD=...
   SQL_EXPERT_DB_SCHEMA=public
   GCP_PROJECT=your-gcp-project-id
   GCP_LOCATION=us-central1
   VERTEX_MODEL=gemini-2.5-flash
   ```
3. **Vertex AI auth** via Application Default Credentials — run once in a terminal:
   ```bash
   gcloud auth application-default login
   ```
   (Vertex uses IAM, not an API key.)
4. **Network access** to download the Toolbox binary (Step 2) and to reach Cloud SQL.

> **Security tip:** point the Toolbox `postgres` source at a **read-only database role**. The
> notebook reuses `SQL_EXPERT_DB_USER`; if that user can write, create a read-only role and use it.

---

## 4. Switching modes — the one thing to change

Open **Step 1** (the config cell) and set the `MODE` variable:

```python
# ===================== CHOOSE YOUR MODE HERE =====================
MODE = "curated"    # Mode 1 — you author fixed queries in tools.yaml (safe)
# MODE = "prebuilt" # Mode 2 — Toolbox auto-generates tools incl. free-form execute_sql
# ================================================================
```

Change that one word, then **re-run the notebook from Step 1 downward.** Everything else keys off
`MODE`:

| Behaviour | `MODE = "curated"` | `MODE = "prebuilt"` |
|---|---|---|
| Who writes the SQL | You (in `tools.yaml`) | Toolbox, generated from the DB |
| `tools.yaml` (Step 3) | Used | Ignored (you can skip authoring) |
| Server launch (Step 4) | `toolbox --config tools.yaml` | `toolbox --prebuilt postgres` (uses `POSTGRES_*` env) |
| Tools the agent gets | Your 5 named tools | Generic tools incl. `list_tables`, `get_table_schema`, `execute_sql` |
| Schema tuning needed | Yes — match column names to your DB | No — runs against live schema as-is |
| Safety | Safe-by-construction (no arbitrary SQL) | Unguarded (model can run any SQL) |
| `sql_expert` validators apply | N/A (SQL is fixed) | No — `execute_sql` bypasses them |

**Which to pick:**
- Choose **`curated`** for a locked-down, production-style surface where only approved questions
  can be asked.
- Choose **`prebuilt`** for quick exploration against your real schema with zero authoring — but
  understand the model can then run any read/query it wants (use a read-only DB role).

---

## 5. Cell-by-cell walkthrough

### Step 0 — Install dependencies
```python
%pip install -q toolbox-langchain langchain-google-vertexai langgraph python-dotenv requests
```
Installs the Python **client** libraries. The Toolbox **server** itself is a standalone binary,
downloaded in Step 2.

### Step 1 — Config (and the MODE switch)
```python
from config import get_settings
settings = get_settings()

MODE = "curated"          # <-- the switch (see Section 4)

DB = { host/port/database/user/password from settings.sql_expert_db_* }
GCP_PROJECT  = settings.gcp_project
GCP_LOCATION = settings.gcp_location or "us-central1"
GEMINI_MODEL = settings.vertex_model or "gemini-2.5-flash"
MAX_ROWS     = settings.sql_expert_max_rows
```
- Reuses the project's `Settings` (from `config.py`), so the DB + Vertex config come from the same
  `.env` the backend uses — no new secrets.
- `BACKEND` is resolved so the notebook can `import config` whether you launch it from the repo
  root or from `backend_py/`.
- Asserts fail fast with a clear message if `SQL_EXPERT_DB_HOST` or `GCP_PROJECT` are missing.

### Step 2 — Download the Toolbox server
```python
TOOLBOX_VERSION = "1.7.0"
# picks windows/darwin/linux + amd64/arm64 automatically
url = f"https://storage.googleapis.com/mcp-toolbox-for-databases/v{TOOLBOX_VERSION}/{os}/{name}"
urllib.request.urlretrieve(url, TOOLBOX_BIN)
```
Downloads the official Toolbox binary for your OS/arch into `backend_py/` (once; skipped if already
present). On macOS/Linux it also sets the executable bit. Alternatives noted in the cell: Docker
image, Homebrew (`brew install mcp-toolbox`), or `npx @toolbox-sdk/server`.

### Step 3 — Author `tools.yaml` (curated mode only)
Builds a Python dict and writes it to `tools.yaml`. Only used when `MODE = "curated"`; in prebuilt
mode the launch cell doesn't pass `--config`, so this file is ignored. Structure:

```yaml
sources:
  rfp-analytics:            # a named DB connection
    kind: postgres
    host: ...               # filled from SQL_EXPERT_DB_*
    port: 5432
    database: ...
    user: ...
    password: ...           # (never printed — the cell redacts it)

tools:
  list_tables:              # a named tool = one fixed SQL statement
    kind: postgres-sql
    source: rfp-analytics
    description: "..."       # the model reads this to decide when to call it
    statement: "SELECT table_name FROM information_schema.tables WHERE ..."
  describe_table:
    kind: postgres-sql
    source: rfp-analytics
    description: "..."
    parameters:             # typed, named params the model must supply
      - name: table_name
        type: string
        description: "Exact table name, e.g. 'bids'."
    statement: "... WHERE table_name = $1 ..."   # $1 binds to the first param
  total_bids: ...
  bids_by_status: ...       # WHERE t.status = $1
  total_suppliers: ...

toolsets:
  rfp-aiq:                  # a named group of tools the client loads together
    - list_tables
    - describe_table
    - total_bids
    - bids_by_status
    - total_suppliers
```

Key points:
- **`statement` is fixed.** Only the bound `$1`, `$2`, ... parameters vary at call time, so there is
  no SQL-injection surface and the model cannot change the query.
- `description` fields are how Gemini decides which tool fits a question — write them clearly.
- The analytics tools use the split-table convention from `services/sql_expert/table_notes.py`
  (`bids` combined with `bids_old` via `UNION ALL`). **Adjust column names** (`status`, etc.) to your
  real schema — run `list_tables` / `describe_table` first if unsure.

### Step 4 — Launch the Toolbox MCP server
Branches on `MODE`:
```python
if MODE == "curated":
    args = [toolbox, "--address", "127.0.0.1", "--port", "5000", "--config", "tools.yaml"]
    TOOLSET_NAME = "rfp-aiq"
else:  # prebuilt
    args = [toolbox, "--address", "127.0.0.1", "--port", "5000", "--prebuilt", "postgres"]
    env += POSTGRES_HOST/PORT/DATABASE/USER/PASSWORD   # prebuilt reads connection from env
    TOOLSET_NAME = None                                 # load all auto-generated tools
proc = subprocess.Popen(args, env=env, ...)
```
- Starts the server as a **background process** and polls port 5000 until it's listening (or raises
  with the server's own log if it exits early).
- The server exposes tools over HTTP on `http://127.0.0.1:5000`, with the MCP endpoint at `/mcp`.
- `proc` is kept so the Cleanup cell can stop it. Re-running the cell first terminates any previous
  instance.

### Step 5 — Load tools into a Gemini agent
```python
tb = ToolboxClient("http://127.0.0.1:5000")
tools = await _load(TOOLSET_NAME)     # named toolset (curated) or all tools (prebuilt)
llm = ChatVertexAI(model=GEMINI_MODEL, project=GCP_PROJECT, location=GCP_LOCATION, temperature=0)
agent = create_react_agent(llm, tools)
```
- `_load` is a small helper that tries `aload_toolset` then `load_toolset` (the method name has
  shifted across `toolbox-langchain` versions) and awaits if needed — robust to version drift.
- `create_react_agent` (LangGraph) wires the Gemini model to the MCP tools in a standard ReAct loop.
- `SYSTEM` pins the output contract: **one short human sentence, never SQL / tool names / JSON**,
  matching the SQL Expert's behaviour.

### Step 6 — Ask questions
```python
async def ask(question):
    result = await agent.ainvoke({"messages": [SystemMessage(SYSTEM), HumanMessage(question)]})
    # prints each tool call Gemini made, then the final one-line answer
```
Runs a few example questions. For each, it prints the MCP tool calls (name + args) so you can see
the round-trips, then the final answer.

### Step 7 — The two modes, recap
Markdown only. Reiterates the curated-vs-prebuilt trade-off table and reminds you it's controlled by
`MODE` in Step 1.

### Step 8 — Comparison and recommendation
Markdown only. Full comparison of MCP Toolbox vs the `sql_expert` pipeline, plus a recommendation
(below in Section 7).

### Cleanup — stop the server
```python
proc.terminate()  # then wait / kill
```
Always run this when done so the Toolbox process doesn't linger on port 5000.

---

## 6. How to add a new curated tool

In `curated` mode, extend the `config["tools"]` dict in Step 3, then add the tool's name to the
`rfp-aiq` toolset, and re-run from Step 3:

```python
config["tools"]["top_suppliers_for_segment"] = {
    "kind": "postgres-sql",
    "source": "rfp-analytics",
    "description": "List the top suppliers contacted for a given segment.",
    "parameters": [
        {"name": "segment", "type": "string", "description": "Business segment, e.g. 'Healthcare'."},
        {"name": "limit", "type": "integer", "description": "How many suppliers to return."},
    ],
    "statement": (
        "SELECT supplier_name, count(*) AS bids "
        "FROM suppliers WHERE segment = $1 "
        "GROUP BY supplier_name ORDER BY bids DESC LIMIT $2;"
    ),
}
config["toolsets"]["rfp-aiq"].append("top_suppliers_for_segment")
```

Parameters are positional: `$1` = first parameter, `$2` = second, etc. Keep `description` fields
specific — that's what the model uses to choose the tool and fill the arguments.

---

## 7. MCP Toolbox vs the `sql_expert` pipeline — and the recommendation

| Dimension | `services/sql_expert` pipeline | MCP Toolbox + Gemini |
|---|---|---|
| SQL origin | LLM generates raw SQL per question | Pre-authored parameterized queries (curated) or generic tools (prebuilt) |
| Safety model | Generate → validate → EXPLAIN → read-only → row cap → timeout | Curated: safe-by-construction. Prebuilt: unguarded |
| New question support | Automatic — any answerable question | Curated: add a tool. Prebuilt: automatic |
| Infrastructure | In-process, no extra service | A separate server process to run/monitor |
| Reuse across hosts | App-internal only | Any MCP host — Gemini, Claude Desktop, Cursor, IDEs |
| Driver LLM | Any (`LLM_PROVIDER`) | Gemini on Vertex here; server is model-agnostic |

**Recommendation for RFP-AIQ:**
- **Keep the `sql_expert` pipeline** as the in-app AIQ chat backend. It already answers open-ended
  questions safely, needs no extra service, and its validators cover LLM-generated SQL. MCP adds a
  moving part without changing that outcome.
- **Reach for Google's MCP Toolbox** when you want the *same database* usable by **external MCP
  hosts** — letting Claude Desktop, Cursor, or a Gemini agent query live bid data during
  development, or exposing a curated, safe-by-construction tool surface to other agents. Its
  parameterized-tool model shines there because it never allows arbitrary SQL.

Short version: **pipeline for the product's chat; MCP Toolbox for cross-host / agent access.**

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `SQL_EXPERT_DB_HOST is empty` | Set the `SQL_EXPERT_*` vars in `backend_py/.env`. |
| `GCP_PROJECT is empty` | Set `GCP_PROJECT` in `.env`. |
| Vertex auth error in Step 5/6 | Run `gcloud auth application-default login`; confirm the account can use Vertex AI in `GCP_LOCATION`. |
| `Toolbox exited early` | The printed server log usually says why — often a bad DB credential/host or an unreachable Cloud SQL instance. Verify the same creds work for the SQL Expert. |
| Download fails in Step 2 | Network/proxy blocked. Use the Docker image or `npx @toolbox-sdk/server` instead, or bump `TOOLBOX_VERSION`. |
| `ToolboxClient has no (a)load_toolset method` | `toolbox-langchain` version drift — upgrade the package, or check its README for the current load method. |
| A curated tool returns an error about a missing column | The seeded statements assume columns like `status`; run `describe_table` and edit the statement to match your schema. |
| Port 5000 already in use | Change `TOOLBOX_PORT` in Step 4 (and it flows through to the client URL). |

---

## 9. Files involved

| File | Role |
|---|---|
| `backend_py/mcp_gcp_toolbox_query_expert_v1.ipynb` | The notebook this guide documents |
| `backend_py/tools.yaml` | Generated by Step 3 (curated mode); the pre-authored query definitions |
| `backend_py/toolbox[.exe]` | The Toolbox server binary, downloaded by Step 2 |
| `backend_py/config.py` | Source of DB + Vertex settings (`Settings`) |
| `backend_py/.env` | Where the actual credentials live |
| `backend_py/services/sql_expert/` | The existing pipeline this is compared against |
