# Enterprise RAG Platform — Handover / Continuation Guide

Written for whoever (human or AI assistant) picks this project up next,
with no prior context. Read this before making changes.

---

## 1. What this is

A backend RAG (Retrieval-Augmented Generation) platform: upload PDF/DOCX/
XLSX/CSV documents, they get parsed/chunked/embedded/indexed into
PostgreSQL (Supabase) with pgvector, and you can query them via hybrid
(lexical + vector) search, with a grounded LLM-generation layer sitting
on top (not yet exposed via HTTP — see §10).

Stack: Python 3.11, FastAPI, SQLAlchemy 2.x + psycopg3, pgvector,
Supabase-hosted Postgres, BGE (local) and Gemini (hosted) for embeddings,
Gemini for generation. No LangChain/LlamaIndex anywhere — every piece is
hand-rolled and deliberately minimal.

Working directory for all commands below: `backend/`.

---

## 2. Status at a glance

| Layer | Status | HTTP exposed? |
|---|---|---|
| Loaders (PDF/DOCX/XLSX/CSV) | ✅ done, tested | via `/upload` |
| Normalization (Text/Table/Image elements) | ✅ done, tested | — |
| Chunking | ✅ done, tested | via `/upload` |
| Embeddings (BGE / Gemini / deterministic-test) | ✅ done, tested | via `/upload`, `/query` |
| Indexing (Postgres + pgvector) | ✅ done, tested against real Supabase | via `/upload` |
| Retrieval (lexical + vector + RRF) | ✅ done, tested against real Supabase | via `/query` |
| RAG generation (`RAGService`, `LLMProvider`) | ✅ built, unit/smoke-tested with fakes | **not yet — no `/ask` route** |
| Real Gemini generation call | ❌ never verified live — `GEMINI_API_KEY` in `.env` is empty | — |
| Tenant isolation | ❌ does not exist anywhere in the schema or code | — |
| Auth | ❌ does not exist | — |

Full test suite as of the last run: **488 passed, 7 skipped, 0 failed**
(`python -m unittest discover -s tests -p "test_*.py" -v`). The 7 skips
are all Gemini-real-API tests, skipped because `GEMINI_API_KEY` is empty.

---

## 3. Architecture

```
Ingestion:
  file (PDF/DOCX/XLSX/CSV)
    -> loader (app/ingestion/loaders/*)
    -> NormalizedDocument (app/models/document.py)
    -> ChunkingService (app/ingestion/chunkers/)
    -> Chunk[] (app/models/chunk.py)
    -> ChunkEmbeddingService + EmbeddingProvider (app/embeddings/)
    -> IndexingService (app/indexing/service.py)
    -> PostgreSQL `chunks` table + pgvector
  Orchestrated by: app/services/ingestion_service.py (IngestionService)
  Exposed via: POST /upload (app/api/routes/upload.py)

Retrieval:
  query
    -> embed_query() (app/embeddings/query.py)
    -> RetrievalRequest (app/retrieval/models.py)
    -> HybridSearchService (app/retrieval/hybrid.py)
         -> LexicalRetriever -> LexicalIndexer.search()  (Postgres FTS, NOT BM25)
         -> VectorRetriever  -> VectorIndexer.search()   (pgvector cosine distance)
         -> reciprocal_rank_fusion() (app/retrieval/rrf.py)
    -> list[RetrievalResult]
  Orchestrated by: app/services/retrieval_service.py (RetrievalService)
  Exposed via: POST /query (app/api/routes/query.py)

Generation (built, NOT yet exposed via HTTP):
  question
    -> RetrievalService.search()
    -> build_user_prompt() (app/rag/prompt.py) + SYSTEM_INSTRUCTION
    -> LLMProvider.generate() (app/rag/llm.py; GeminiLLMProvider implemented)
    -> RAGAnswer{question, answer, sources, retrieved_count}
  Orchestrated by: app/services/rag_service.py (RAGService)
  Exposed via: nothing yet -- next step is POST /ask
```

**Composition root:** `app/api/dependencies.py`. Every service is built
lazily via a `@lru_cache`-decorated `get_*()` function, injected into
routes via FastAPI's `Depends()`. Read this file top to bottom to see
exactly how everything wires together — it's short and it's the map of
the whole system's dependency graph.

---

## 4. Repository map (the files that matter)

```
backend/
├── .env / .env.example        # env config; .env is gitignored, .env.example documents every var
├── requirements.txt            # pinned direct deps only (no transitive deps listed)
├── app/
│   ├── main.py                 # FastAPI app; registers health/upload/query routers
│   ├── api/
│   │   ├── dependencies.py     # <-- START HERE. Composition root for all services.
│   │   └── routes/
│   │       ├── health.py       # GET /health
│   │       ├── upload.py       # POST /upload
│   │       └── query.py        # POST /query
│   ├── models/
│   │   ├── document.py         # NormalizedDocument, TextElement/TableElement/ImageElement
│   │   └── chunk.py             # Chunk (the retrieval unit; deliberately storage-agnostic)
│   ├── ingestion/
│   │   ├── loaders/             # CSVLoader, DocxLoader, ExcelLoader, PDFLoader
│   │   └── chunkers/             # TextChunker, TableChunker, ImageChunker, ChunkingService
│   ├── embeddings/
│   │   ├── base.py               # EmbeddingProvider ABC
│   │   ├── bge.py                 # BGEEmbeddingProvider (local, no API key)
│   │   ├── gemini.py               # GeminiEmbeddingProvider (hosted, needs GEMINI_API_KEY)
│   │   ├── local.py                 # DeterministicTestEmbeddingProvider (test/dev only)
│   │   ├── factory.py                # picks a provider by EMBEDDING_PROVIDER env var
│   │   └── query.py                   # embed_query() -- same provider for docs and queries
│   ├── indexing/
│   │   ├── config.py             # DatabaseConfig.from_env() -- also where .env auto-loads (see §7)
│   │   ├── database.py            # create_db_engine(), init_schema()
│   │   ├── models.py               # build_chunks_table() -- the ONE Table() definition
│   │   ├── repository.py            # ChunkRepository (core row upsert/delete)
│   │   ├── lexical.py                # LexicalIndexer (Postgres FTS)
│   │   ├── vector.py                  # VectorIndexer (pgvector)
│   │   └── service.py                  # IndexingService (write-path coordinator)
│   ├── retrieval/
│   │   ├── models.py             # RetrievalRequest, RetrievalResult
│   │   ├── lexical.py             # LexicalRetriever
│   │   ├── vector.py                # VectorRetriever
│   │   ├── hybrid.py                 # HybridSearchService
│   │   └── rrf.py                     # reciprocal_rank_fusion()
│   ├── rag/
│   │   ├── llm.py                # LLMProvider ABC + GeminiLLMProvider
│   │   └── prompt.py               # SYSTEM_INSTRUCTION + build_user_prompt()
│   └── services/
│       ├── ingestion_service.py   # IngestionService (write-side orchestrator)
│       ├── retrieval_service.py    # RetrievalService (read-side orchestrator)
│       └── rag_service.py           # RAGService (generation orchestrator)
└── tests/                        # 35 test files, stdlib unittest only, see §7
```

---

## 5. Running it locally

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate   # or source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
cp .env.example .env     # then fill in DATABASE_URL and (if using Gemini) GEMINI_API_KEY
uvicorn app.main:app --reload
```
Then: `GET http://127.0.0.1:8000/health`, `POST /upload` (multipart file), `POST /query`
(JSON `{"query": "...", "top_k": 10}`), and `http://127.0.0.1:8000/docs` for interactive Swagger.

**Windows note:** stale `uvicorn` processes sometimes hold a port even
after the shell that started them is gone. If `bash`'s `kill <pid>`
doesn't release the port, use PowerShell: `Get-NetTCPConnection
-LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess
-Force }`. Simplest fix is usually just picking a different port.

---

## 6. Configuration reference (`.env`)

| Variable | Required? | Purpose |
|---|---|---|
| `DATABASE_URL` | **yes** | Postgres connection string. **Must use `postgresql+psycopg://`**, not bare `postgresql://` — SQLAlchemy defaults the latter to psycopg2, which isn't installed (only psycopg3 is). This bit us once already. |
| `EMBEDDING_DIMENSION` | no | Overrides the pgvector column width used when building the `Table` object. Currently unset (falls back to whichever embedding provider is active — 768 for BGE). |
| `EMBEDDING_PROVIDER` | no | `bge` (default, local, no key needed) / `gemini` / `deterministic` (test only). |
| `GEMINI_API_KEY` | only if using Gemini | **Currently empty in `.env`.** Needed for `GeminiEmbeddingProvider` and `GeminiLLMProvider`. Get one at https://ai.google.dev/gemini-api/docs/api-key. |
| `GEMINI_LLM_MODEL` | no | Overrides the Gemini generation model. Default: `gemini-3.7-flash` (verified against official docs + SDK at implementation time — don't assume this is still current without re-checking; Google renames/deprecates models). |

`.env` is loaded automatically by `python-dotenv`, triggered as an
**import-time side effect** inside `app/indexing/config.py`. This means:
any code that reads `os.environ` for one of these vars must (directly or
transitively) import `app.indexing.config` first, or the var won't be
there yet. This already caused one bug (a test file that skipped even
though `DATABASE_URL` was configured, because it never imported that
module) — fixed, but be aware of the pattern if you add new entry points.

---

## 7. Testing

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

**Conventions — follow these, don't introduce pytest or change the style:**
- stdlib `unittest` only, everywhere. No pytest, ever (deliberate project rule).
- Ordinary tests must never require network, a real database, or a downloaded
  model. Use fakes/test doubles (there's a consistent `FakeEngine`/
  `FakeConnection` pattern used across ~6 test files for faking SQLAlchemy
  without a real Postgres connection — copy that pattern, don't reinvent it).
- Tests that *do* need something real are explicitly gated and skip cleanly
  with a clear reason if the resource isn't available:
  - `tests/test_indexing_integration.py` — real Postgres, gated on `DATABASE_URL` (currently **runs**, not skipped, since `DATABASE_URL` is set).
  - `tests/test_bge_provider_integration.py`, `tests/test_bge_pipeline_smoke.py` — real BGE model load (currently **runs**, model is cached locally).
  - `tests/test_gemini_embedding_integration.py` — real Gemini API, gated on `GEMINI_API_KEY` (currently **skips** — key is empty).
- Every service (`IngestionService`, `RetrievalService`, `RAGService`) has a
  "realistic smoke test" alongside its unit tests: real internal classes
  wired together, only the actual network/DB call faked out. If you add a
  new service, add one of these too.
- `tests/test_indexing_integration.py` writes real rows to the real Supabase
  `chunks` table during its run and cleans them up in `tearDown`. This is
  intentional, existing, accepted behavior — see §9 for the risk this carries.

---

## 8. Design conventions to preserve

- **Constructor-based dependency injection everywhere.** Nothing reaches
  into global state or constructs its own dependencies internally. New
  services should take their dependencies as `__init__` params and get
  wired up in `app/api/dependencies.py`.
- **Provider abstraction pattern**: an ABC (`EmbeddingProvider`,
  `LLMProvider`) with a concrete public method that does shared validation,
  delegating to an abstract `_impl`-style hook the concrete subclass
  implements. Follow this shape for any new provider type.
- **Never put `str(exc)` in a client-facing or wrapped-exception message.**
  Every exception wrapper in this codebase uses `type(exc).__name__` only,
  and always chains via `raise ... from exc`. This is a deliberate,
  consistently-applied rule to avoid leaking connection strings/API keys
  through error messages. Don't break it.
- **Each orchestrator service defines its own small exception hierarchy**
  (e.g. `IngestionError`/`InvalidInputError`/... , `RAGError`/
  `InvalidQuestionError`/...), each dual-inheriting from both a
  service-specific base and the natural builtin (`ValueError`/
  `RuntimeError`) so callers can catch broadly or specifically.
- **"Smallest coherent change" philosophy.** Every phase of this project
  was scoped tightly and stopped before the next; no speculative
  abstraction, no rewriting working code without a proven defect. Keep
  doing that.
- **Model names and dimensions are never hardcoded without being
  overridable** via env var, and are verified against current official
  docs/SDK source before being written — don't assume a model name from
  training data without checking (things move fast; this project already
  had to correct a wrong assumption about Gemini's embedding API once).

---

## 9. Known issues / gotchas

1. **`GEMINI_API_KEY` is empty.** Nothing Gemini-dependent has been
   verified against the real API — only against fakes. Fill it in and
   re-run the gated tests before trusting `GeminiLLMProvider` or
   `GeminiEmbeddingProvider` in production.

2. **Test suite and the real app share one physical Supabase database.**
   This already caused a real incident: `test_indexing_integration.py`'s
   own schema-creation call locked the `chunks.embedding` column at
   `vector(4)` (matching its own small test vectors) the first time it
   ran against this database, which then broke every real BGE upload
   (768-dim) until diagnosed and fixed (`ALTER TABLE ... TYPE vector(768)`,
   plus updating that test's vectors to 768-dim via zero-padding so it
   still passes). **The underlying risk is not fixed, only the symptom
   was.** A proper fix (separate test database/schema, discussed but never
   implemented) is the single highest-value infrastructure improvement
   left. See the git history around the "align pgvector schema" commit
   for the full incident writeup if you want the details.

3. **No cross-check between configured embedding dimension and the actual
   physical Postgres column width.** `VectorIndexer` only validates against
   whatever dimension *it was told* to expect, never against what the
   database actually has. This is exactly what let issue #2 happen silently
   until a real write failed. A cheap fix was scoped but never
   implemented: read the column's actual width via a one-time introspection
   query and fail loudly at startup if it disagrees with the configured
   provider's dimension.

4. **`Chunk.embedding_model` is never populated.** The field exists on the
   model but `ChunkEmbeddingService` doesn't set it. Harmless today, but
   if you ever need to know "which model embedded this chunk" for a
   migration, it won't be there.

5. **PDF image assets are stored in a local temp directory**
   (`tempfile.gettempdir()/enterprise_rag_pdf_assets`), referenced by
   `ImageElement.image_uri`. This does not survive redeploys and won't work
   across multiple app instances. Needs real object storage (S3/MinIO/etc.)
   before this matters in production — deliberately out of scope so far.

6. **No tenant/organization isolation anywhere.** Confirmed absent from the
   `Chunk` model, the `chunks` table schema, and every retrieval path.
   Anyone who can call `/query` can search across all indexed documents,
   full stop. This has been flagged repeatedly since the indexing phase
   and intentionally not invented ad hoc — it needs a real design decision
   (what is a "tenant" here — an org? a workspace? a client?) before being
   built.

7. **No authentication on any endpoint.** `/upload`, `/query` — all open.

---

## 10. What's NOT implemented, roughly in priority order

1. **`POST /ask`** — expose `RAGService` via HTTP. This is the natural
   next step; the pattern is already established twice (`/upload` wraps
   `IngestionService`, `/query` wraps `RetrievalService` — `/ask` should
   wrap `RAGService` the same way: a route-local request/response schema,
   `Depends(get_rag_service)`, map `RAGError` subclasses to HTTP status
   codes the same way `upload.py`/`query.py` already do).
2. Verify real Gemini generation end-to-end once a key is available.
3. Tenant isolation (needs a design decision first, then schema + code).
4. Authentication.
5. Test/prod database isolation (plan exists, not executed — see §9.2).
6. Object storage for PDF image assets.
7. Reranking (a cross-encoder pass after RRF) — explicitly deferred every
   phase so far, never started.
8. True BM25 if Postgres full-text search ranking proves insufficient
   (the architecture was deliberately kept swappable for this — see
   `app/indexing/lexical.py`'s docstring).
9. Observability (structured logging, metrics, tracing) — today it's just
   scattered `logger.exception()` calls in the route layer.
10. Streaming responses, chat memory/multi-turn, agents — explicitly out
    of scope for this project's current milestone, not accidentally
    omitted.
11. Deployment story (Docker, CI/CD) — nothing exists yet.

---

## 11. Uncommitted work — check this first

As of this document, `git log` shows the project committed up through
`POST /query`. **The RAG generation phase (this session's latest work) is
sitting uncommitted in the working tree:**
```
 M backend/.env.example
 M backend/app/api/dependencies.py
?? backend/app/rag/llm.py
?? backend/app/rag/prompt.py
?? backend/app/services/rag_service.py
?? backend/tests/test_gemini_llm_provider.py
?? backend/tests/test_rag_service.py
```
Review and commit this before doing anything else, or it's at risk of
being lost. (No commits were made automatically during this project's
development — every commit was a deliberate, explicit step, and this
last batch hasn't had that step yet.)

---

## 12. Quick sanity check for a fresh start

```bash
cd backend
python -m unittest discover -s tests -p "test_*.py" -v   # expect 488 passed, 7 skipped
uvicorn app.main:app --reload                              # then hit /health, /docs
```
If either of those doesn't match this document, something's drifted —
trust the code and the live test output over this file, and update this
file to match once you've confirmed what changed.
