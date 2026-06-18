# Enterprise RAG Assistant — Case Study

📺 [Watch Demo](https://youtu.be/Rvdz9DKtz5o?si=GcMIR7nWABJQYCR1) | 🐳 [Docker Hub](https://hub.docker.com/repository/docker/shivamrajput130/enterprise-rag-assistant/general) | 💻 [GitHub](https://github.com/shivamrajput130/enterprise-rag-assistant)

This is the real development story: what I planned to build, what broke, how I debugged it, and what I'd do differently. The project was not built in a single iteration — it went through nine versions over approximately six weeks. Every bug in this document was real.

> **Note on scope:** The term "enterprise" refers to the document types and retrieval challenges — HR policies, employee records, CSV datasets, FAQ files — not to production-scale deployment. The system runs locally and in Docker, has no authentication, and was built on a personal laptop (Windows 11, i5-12450H, 16 GB RAM, GTX 1650).

---

## Development Timeline

| Phase | Work done |
|---|---|
| Week 1 | Base utilities (logger, exception), config setup, PDF-only vector search |
| Week 2 | Multi-format loading (CSV, DOCX, JSON, TXT, Excel), hallucination fix, grounding prompt |
| Week 3 | Query expansion, BM25 hybrid retrieval, CrossEncoder reranking |
| Week 4 | RAGAS integration, E2E tests, Streamlit UI polish, feedback analytics |
| Week 5 | Pandas Analytics Engine, Query Router, tabular query debugging |
| Week 6 | Supabase migration (SQL Server → PostgreSQL), Docker, Docker Hub push |

---

## Project Evolution

| Version | What changed | Why |
|---|---|---|
| V1 | PDF-only, pure vector search | Starting point |
| V2 | Multi-format loading (CSV, DOCX, JSON, TXT, Excel) | PDF-only is not a portfolio piece |
| V3 | Grounding prompt + fallback | LLM answered from training data |
| V4 | Query expansion | Short queries missed correct chunks |
| V5 | CrossEncoder reranking | Irrelevant chunks ranked too high |
| V6 | Hybrid retrieval (Vector + BM25) | ID and name queries failed completely |
| V7 | Pandas Analytics Engine + Query Router | Aggregation queries hallucinated in RAG |
| V8 | Supabase PostgreSQL | SQL Server not portable, broke in Docker |
| V9 | Docker + Docker Hub deployment | Reproducible deployment |
| V10 | Docker Hub publish + Supabase cloud deployment | Production-ready, zero local dependency |

---

## 1. Where It Started

The original plan was a PDF chatbot. After building the basic version and showing it to my professor and peers, the feedback was consistent: a PDF-only chatbot is not a portfolio piece. Every RAG tutorial builds one.

The more interesting question was what breaks when you extend to a real document set — HR policies, student records, FAQ JSON files, company notes. That decision introduced every bug in this document.

I used `uv` instead of pip throughout. Faster installs, cleaner environment management.

---

## 2. Base Utilities First

Before building the pipeline, I set up logging and exception handling. This paid off throughout — every bug was traceable to an exact file and line number from day one.

```python
# logger.py — timestamped log file per run
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
file_handler = logging.FileHandler(f"logs/{LOG_FILE}")
stream_handler = logging.StreamHandler()
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.propagate = False  # prevents duplicate log entries from root logger
```

```python
# exception.py — file name and line number from traceback
class RagException(Exception):
    def __init__(self, message, error_detail):
        super().__init__(message)
        _, _, exc_tb = error_detail.exc_info()
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.line_number = exc_tb.tb_lineno
```

Most retrieval debugging was done by inspecting logs directly, not the UI. The `debug.retrieval` and `debug.reranker` flags in `config.yaml` enabled detailed chunk-level logging without adding print statements to the code.

---

## 3. Config-Driven Design

All tunable parameters live in `config.yaml`. I iterated on chunk size, retrieval top-k values, and reranker limits many times. Doing that without config would have meant grep sessions across the codebase.

I selected Groq as the LLM provider because of its low latency and free developer tier. Fast responses allowed quick iteration during debugging.

---

## 4. Document Loading — First Real Problem

My initial loader converted every file to one text string and created one large Document. That worked for a single clean PDF. It failed for everything else.

**The CSV problem:**

With text-blob loading, a CSV like:
```
Student_ID,Name,Class,Math,Science
S127,Nikhil Sharma,10,96,95
S128,Priya Mehta,10,88,91
```
became 2000 characters of comma-separated text. When chunked at 1000 characters, a record could be split mid-row. Querying `S127` retrieved a chunk containing the tail of one record and the start of the next — neither complete.

Fix — per-row document creation:
```python
def create_row_documents(file_path, file_name, file_type, df):
    for index, row in df.iterrows():
        row_text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(Document(
            page_content=row_text,
            metadata={"source": file_path, "file_name": file_name,
                      "file_type": file_type, "row": index}
        ))
```

Each row becomes a self-contained retrieval unit. The same files also remain available as DataFrames for the Pandas Analytics Engine — added in V7.

**Measured ingestion results:**

Controlled test corpus (5–6 files): approximately 80–120 chunks.

Real-world scale test — two full ML/data science textbooks (~1800 pages combined):
- **Total documents loaded: 1129**
- **Total chunks created: 2538**
- Retrieval, source citations with page numbers, and fallback all worked correctly
- Re-upload of existing files triggered "already exists. Skipping." correctly
- Average warm query response time: ~3.2 seconds

**The JSON problem:** Loading FAQ JSON as one text blob meant any FAQ question retrieved the entire file. Per-FAQ-entry documents fixed this — each entry retrieves independently.

---

## 5. Embedding Model Upgrade

Started with `all-MiniLM-L6-v2` (384 dimensions). Multi-concept queries like "compare probation period and WFH policy" missed relevant chunks. Switched to `BAAI/bge-base-en-v1.5` (768 dimensions). Retrieval quality on complex queries improved noticeably.

```python
# Singleton — loads once per process, not per query
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    _embedding_model = HuggingFaceEmbeddings(model_name=config["embedding"]["model_name"])
    return _embedding_model
```

Without the singleton, the first query of each FastAPI process took 3–4 extra seconds. Same pattern applied to the CrossEncoder reranker.

**Known limitation:** Changing the embedding model requires deleting the ChromaDB collection and re-ingesting everything — stored vectors are incompatible across models.

---

## 6. ChromaDB Over FAISS

FAISS also supports persistence via `write_index`/`read_index`, but requires explicit save and load calls on every restart. ChromaDB handles persistence automatically. For a system where documents are uploaded once and queried repeatedly across sessions, automatic persistence was more important than raw speed. This was a maintainability decision, not a performance one.

MD5-based chunk deduplication:
```python
def create_chunk_id(chunk):
    source = chunk.metadata.get("source", "")
    content = chunk.page_content or ""
    return hashlib.md5((source + content).encode("utf-8")).hexdigest()
```

Same source + same content → same ID. Re-uploading an unchanged file produces no duplicate chunks.

---

## 7. Hallucination Fix — Grounding Prompt

Before fixing this, the LLM sometimes answered from its training knowledge instead of the retrieved documents. A question about the company's WFH policy would return generic remote work advice instead of the specific policy from `hr_policy.docx`.

```python
MAX_CONTEXT_CHARS = 8000
context = context[:MAX_CONTEXT_CHARS]

prompt = f"""
Use ONLY information from the provided context.
Never use outside knowledge.
Never invent or assume information not explicitly stated.

FALLBACK RULE: If the answer cannot be found in the context, respond with EXACTLY:
I don't know based on the provided documents.

CONTEXT:
{context}

QUESTION: {query}
"""
```

`MAX_CONTEXT_CHARS = 8000` was added after hitting Groq token limit errors on long multi-document queries. Known risk: may truncate a relevant passage near position 8000.

---

## 8. Query Expansion

Short queries like "salary?" embedded poorly against full paragraphs. A 3-word query sits in a different region of the 768-dimension embedding space than the paragraph answering it.

```python
# 1 original + up to 4 LLM-generated = max 5 total, deduplicated
def clean_expanded_query(text):
    text = text.strip().lstrip("-•*").strip()
    if len(text) > 2 and text[0].isdigit() and text[1] in [".", ")"]:
        text = text[2:].strip()
    return text.strip('"').strip("'").strip()
```

`clean_expanded_query()` was needed because the LLM sometimes returned numbered lists, quoted strings, or bullet points. Without it, "1. When is salary credited?" would be sent as a search query with the numbering.

**Trade-off:** Expansion adds noise for exact-token queries like `S127`. BM25 handles those instead.

---

## 9. Hybrid Retrieval — The Biggest Single Improvement

Queries like `Student_ID S127` and `Rahul Verma` consistently failed with pure vector search. I first tried tuning vector parameters — increasing top-k, adjusting thresholds. Nothing helped. The problem was fundamental: `S127` has no semantic representation in embedding space.

The actual failure was worse than "no result returned." The system retrieved an HR policy chunk about "confidentiality of identification numbers" — the LLM answered confidently from the wrong document. No error signal.

BM25 does exact token matching. `s127` matches `s127`.

```python
def bm25_search(query, documents, top_k):
    corpus = [doc.page_content for doc in documents]
    tokenized_corpus = [tokenize(text) for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenize(query))
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k] if score > 0]
```

**Known edge case:** Hyphenated IDs like `EMP-001` tokenize as `emp` and `001`. Both match but scoring differs from treating it as one token. Not yet fixed.

**Known scale limitation:** BM25 rebuilds its index from all chunks in ChromaDB per query. At 2538 chunks this is fast. For much larger corpora, a persistent keyword index would be needed.

---

## 10. CSV Formatting Bug — Retrieval Was Fine, Prompt Was Wrong

After BM25 started retrieving the correct CSV row, the LLM still answered `S127` with just:
```
S127
```

The retrieval was correct the entire time. I spent time tuning BM25 parameters before realizing the problem was the prompt. It had no instruction for entity queries — the LLM defaulted to minimal output.

Two lines of prompt instruction fixed it:
```
If the question refers to a specific person, employee, student, ID, or entity,
return all relevant fields for that record.
```

After fix:
```
Student ID: S127 | Name: Nikhil Sharma
Class: 10 | Section: A
Math: 96 | Science: 95 | English: 98 | Hindi: 93 | Social Science: 99
Total: 481 | Percentage: 96.2 | Grade: A+
```

This was the bug that took longest to isolate — because retrieval was working correctly the whole time.

---

## 11. RAGAS Integration — Evaluation Tooling as Engineering Problem

I integrated RAGAS for automated evaluation. It worked, but with significant friction.

**Dependency conflict:** Newer RAGAS versions broke with newer LangChain. The specific failure was in LangChain's callback system — RAGAS used internal LangChain APIs that changed across versions. Additionally, `vertexai` import errors surfaced from RAGAS internal dependencies even though I wasn't using Vertex AI at all. `ragas==0.2.10` was the stable version after multiple failed attempts.

**Rate limit conflict:** I was using Groq for both query expansion in the application and as the RAGAS evaluator. Both competed for the same rate limit quota during evaluation runs. `faithfulness: nan` appeared on most runs — the evaluator LLM call was timing out.

Best stable run: Answer Relevancy 0.88, Context Precision 1.0, Context Recall 1.0.

The lesson: don't use the same rate-limited LLM provider for the application and the evaluation harness simultaneously.

---

## 12. Pandas Analytics Engine — Why RAG Alone Was Not Enough

After the RAG pipeline was stable, I tested aggregation queries:

```
"What is the average math score?"
"Who are the top 5 students?"
"How many students scored above 90?"
```

These all failed. The RAG pipeline retrieved CSV row chunks and passed them as text to the LLM. The LLM estimated averages from raw text. The estimates were wrong.

The RAG pipeline is designed for retrieval — finding the right chunk. It cannot compute. These queries need computation.

The Pandas Analytics Engine loads the CSV/Excel file directly as a DataFrame and runs pandas operations:

```python
# tabular_query.py
# average → df[col].mean()
# top N   → df.nlargest(n, col)
# count   → conditional len()
# filter  → boolean indexing
# range   → df[(df[col] >= low) & (df[col] <= high)]
# negative filter → df[df[col] != value]
```

**Negative filter bug:** The initial implementation had a boolean logic error for negative filter queries ("show students who are not in class 10"). It returned all students including class 10 students. Fixed by correcting the pandas boolean condition in `tabular_query.py`.

**Query Router bug:** The router initially sent some ambiguous queries to the wrong pipeline. Refined the LLM prompt for classification. After refinement, all test queries routed correctly.

---

## 13. Supabase Migration — SQL Server to PostgreSQL

The original feedback system used MS SQL Server via pyodbc. This required a local SQL Server installation, a machine-specific connection string in the code, and ODBC Driver 17. It worked on my machine. It did not work in Docker.

When building the Docker image, the SQL Server ODBC driver installation added significant complexity to the Dockerfile and still produced connection errors in the container. The feedback system broke every time the container was rebuilt.

Migrated to Supabase PostgreSQL. Supabase is cloud-native — the connection string is a URL, credentials come from `.env`, and there is nothing to install on the host machine or in the Docker image beyond `psycopg2`.

The analytics data that was most useful from the feedback table: the negative feedback list. It directly identified which query types the system answered badly — aggregation queries before the Pandas Engine, ambiguous short queries, and broad multi-hop questions. This list drove the next round of improvements more effectively than any other signal.

---

## 14. Docker and Deployment

The Dockerfile builds the environment, copies the source, and uses `start.sh` to start FastAPI and Streamlit together. Credentials come from `--env-file .env` at runtime — no secrets in the image.

Common issues during Docker development:
- `vertexai` import error from RAGAS surfacing during container build — resolved by pinning RAGAS version
- Groq API key not reaching the application — `--env-file` syntax issues
- ChromaDB volume persistence across container restarts — required explicit volume mounting
- Supabase connection from inside container — worked correctly once the connection string was in `.env`

The Docker image is published at: [hub.docker.com/r/shivamrajput130/enterprise-rag-assistant](https://hub.docker.com/repository/docker/shivamrajput130/enterprise-rag-assistant/general)

---

## 15. All Bugs and Fixes

| Bug | Root cause | Fix |
|---|---|---|
| LLM answered from training data | No grounding constraint | Strict grounding prompt + fallback |
| Token limit errors | Multi-chunk context too large | `MAX_CONTEXT_CHARS = 8000` |
| Short queries missed correct chunks | Query-paragraph embedding mismatch | Query expansion → up to 5 queries |
| LLM returned bullets in expansion output | No output cleaning | `clean_expanded_query()` |
| Irrelevant chunks ranked high | Cosine similarity is approximate | CrossEncoder reranking |
| IDs/names not retrieved | Semantic search cannot represent arbitrary tokens | BM25 + hybrid retrieval |
| CSV chunk splits mid-row | Text-blob loading + chunk boundary | Per-row Document creation |
| CSV entity query returns minimal answer | Prompt missing entity rule | Entity-aware prompt instructions |
| Multi-hop questions incomplete | Only top 1 chunk used | Pass all top reranked chunks |
| Duplicate chunks on re-ingestion | No deduplication | MD5 deterministic chunk IDs |
| Source citation pollution | No deduplication before reranking | Deduplicate by `source + content` |
| RAGAS import errors | LangChain callback API changed | Pinned `ragas==0.2.10` |
| `vertexai` import error | RAGAS internal dependency | Same pin resolved it |
| Faithfulness NaN | Groq rate limits during evaluation | Known — separate evaluator needed |
| Aggregation queries hallucinated | RAG pipeline cannot compute | Pandas Analytics Engine + Query Router |
| Negative filter returned wrong rows | Boolean logic error in `tabular_query.py` | Fixed condition |
| Router sent queries to wrong pipeline | Ambiguous prompt for classification | LLM prompt refinement |
| SQL Server broke in Docker | ODBC driver installation in container | Migrated to Supabase PostgreSQL |
| Groq key not reaching container | `--env-file` syntax | Fixed Docker run command |
| ChromaDB lost data on container restart | No volume mount | Added volume configuration |

---

## 16. What I'd Do Differently

**Set up RAGAS and a separate evaluator LLM from day one.** Using Groq for both the application and evaluation created rate limit conflicts. A local Ollama model for RAGAS would have made faithfulness measurement possible.

**Test all formats from day one.** Building PDF loading first and getting comfortable with it meant the CSV and JSON problems surfaced late. If I had tested all formats from the start, the loading layer would have been designed differently from the beginning.

**Write E2E tests before writing code.** The test cases I wrote reflect the bugs I already knew about. Tests written before implementation force you to think about edge cases — routing failures, empty results, malformed queries.

**Build the Pandas Engine earlier.** I only added it after seeing aggregation queries fail in the RAG pipeline. If aggregation was a requirement from the start, the Query Router architecture would have been part of the initial design.

**Measure response time per component, not just total.** I know total query latency but not the breakdown between router classification, query expansion, vector search, BM25, reranking, and LLM inference. That breakdown would have made latency optimization much easier.

**Plan for Docker from the start.** Adding Docker at the end meant retrofitting environment variable handling, removing hardcoded paths, and replacing the local SQL Server dependency. Designing for Docker from the start would have been less work than fixing it at the end.

Productionizing a project is often harder than building the initial prototype.

---

## 17. What I'd Build Next

**Persistent BM25 index.** The current implementation rebuilds from the full corpus per query. For large corpora this becomes a bottleneck. A pre-built index would fix this.

**Pandas Engine for PDF tables.** Currently only CSV and Excel feed the Pandas Engine. Tables embedded in PDF or DOCX go through RAG as text. Extracting tables from PDF with `camelot` or `pdfplumber` and feeding them to the Pandas Engine would improve accuracy on document-embedded data.

**Real streaming.** The current word-by-word delay is simulated. Real server-sent event streaming from Groq would reduce perceived latency.

**Separate evaluation LLM.** Running RAGAS with a non-rate-limited model — even a small local model via Ollama — would finally allow stable faithfulness measurement.

**GitHub Actions CI pipeline.** Currently in progress. The E2E test suite should run automatically on every push.

---

## 18. Final Reflection

Before this project, I thought of RAG as "a chatbot connected to a vector database." The interesting engineering was assumed to be in the LLM.

After building and debugging this system, I understand retrieval as the primary quality variable. Most failures originated in retrieval — wrong chunk retrieved, no chunk retrieved, or the wrong pipeline used entirely. The LLM produced reasonable answers when retrieval worked and failed when it didn't.

The biggest single improvement was hybrid search — adding BM25 alongside vector search. Not a model upgrade. Not prompt engineering. Fixing retrieval.

The second biggest improvement was the Pandas Analytics Engine and Query Router — the realization that aggregation queries and semantic queries are fundamentally different problems and need separate pipelines. A single retrieval system cannot serve both well.

Those two insights came from testing the system on real queries, not from reading about RAG. That is the main reason to build rather than just study.
