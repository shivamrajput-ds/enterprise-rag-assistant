# INTERVIEW_QA.md

# Enterprise RAG Assistant -- Interview Q&A

## 1. Why did you build this project?

Most RAG demos only answer semantic questions. I wanted a system that
could answer both document questions and structured analytics over
CSV/Excel.

------------------------------------------------------------------------

## 2. Why Hybrid Search instead of only vector search?

Vector search captures semantic meaning but can miss exact identifiers
(IDs, names, codes). BM25 improves keyword retrieval while dense
embeddings improve semantic retrieval. Combining both gives better
recall.

------------------------------------------------------------------------

## 3. Why use a CrossEncoder reranker?

Initial retrieval may return loosely related chunks. The CrossEncoder
scores query--document pairs jointly and reorders them, improving answer
quality.

------------------------------------------------------------------------

## 4. Why build a Pandas Analytics Engine?

LLMs and vector databases are not designed for deterministic aggregation
like COUNT, SUM, AVG, TOP-N or numeric filtering. Routing those queries
to Pandas provides exact results.

------------------------------------------------------------------------

## 5. How does query routing work?

The router detects analytical intent using query patterns. Analytics
queries go to Pandas; semantic/document questions go through the RAG
pipeline.

------------------------------------------------------------------------

## 6. Why didn't you rely on RAGAS?

I experimented with RAGAS but found evaluation slow and unstable for
repeated local testing. I instead created deterministic end-to-end test
cases with expected outputs for reliable regression testing. RAGAS can
be added later for automated benchmarking.

------------------------------------------------------------------------

## 7. What are the current limitations?

-   Large document collections increase latency.
-   OCR for scanned PDFs is limited.
-   No authentication or multi-user support.
-   Distributed vector storage is not implemented.

------------------------------------------------------------------------

## 8. Future Improvements

-   LangGraph-based agents
-   Authentication & RBAC
-   Redis caching
-   Elasticsearch/OpenSearch
-   Kubernetes deployment
-   Automated evaluation pipeline

------------------------------------------------------------------------

## 9. Biggest engineering challenge

Correctly routing analytical queries while preventing hallucinations.
The solution was separating deterministic analytics from semantic
retrieval.

------------------------------------------------------------------------

## 10. What are you most proud of?

Building an end-to-end production-style architecture instead of only a
notebook model, including ingestion, retrieval, routing, APIs, UI,
feedback collection, Docker, and CI/CD.
