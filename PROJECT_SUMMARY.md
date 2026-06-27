# Enterprise RAG Assistant — Project Summary

## 1. Project Overview

**Enterprise RAG Assistant** is a production-oriented document question-answering system designed to work with real enterprise-style files such as PDFs, DOCX files, CSVs, Excel sheets, JSON FAQs, and TXT documents.

The system is not just a basic PDF chatbot. It combines a **Hybrid RAG Pipeline** for semantic document questions with a **Pandas Analytics Engine** for structured-data queries. A **Hybrid Query Router** decides whether a user question should go to the RAG pipeline or the analytics pipeline.

---

## 2. Problem Statement

Traditional RAG systems work well for simple semantic questions, but they fail in two important real-world cases:

1. **Exact ID and name lookup**
   - Vector search may fail on values like `Student_ID S127`, employee IDs, names, and codes because these tokens do not have strong semantic meaning.

2. **Numerical and tabular analytics**
   - Questions like “What is the average math score?” or “Who are the top 5 students?” require computation.
   - A normal RAG system may retrieve text chunks and let the LLM guess the answer, which is unreliable.

This project solves these problems by separating semantic retrieval and structured computation into two different pipelines.

---

## 3. Solution

The system uses a hybrid architecture:

| Query Type | Example | Engine Used |
|---|---|---|
| Semantic document QA | What is the WFH policy? | Hybrid RAG Pipeline |
| Exact lookup | Student_ID S127 | BM25 + Vector Retrieval |
| Analytics query | Average math score | Pandas Analytics Engine |
| Ranking query | Top 5 students by percentage | Pandas Analytics Engine |
| Out-of-scope query | Who won the latest cricket match? | Grounded fallback |

---

## 4. System Architecture

```text
User Question
      |
      v
Hybrid Query Router
      |
      |-- Analytics Query --> Pandas Analytics Engine
      |
      |-- Semantic Query ----> Hybrid RAG Pipeline
                              |
                              |-- Query Expansion
                              |-- Vector Search
                              |-- BM25 Search
                              |-- Merge + Deduplicate
                              |-- CrossEncoder Reranking
                              |-- Grounded LLM Answer
```

The final answer includes:

- Answer text
- Retrieved sources
- Page/row metadata where available
- Response time
- Feedback capture through Supabase

---

## 5. Key Features

- Multi-format document ingestion: PDF, DOCX, CSV, JSON, TXT, XLS, XLSX
- Hybrid retrieval using vector search + BM25
- CrossEncoder reranking for better answer precision
- Query expansion for improved recall
- Query router for deciding RAG vs analytics path
- Pandas Analytics Engine for deterministic computation
- Strict grounding prompt to reduce hallucination
- Source attribution with page/row metadata
- FastAPI backend
- Streamlit user interface
- Supabase PostgreSQL feedback analytics
- Dockerized deployment
- Docker image published to Docker Hub
- GitHub Actions CI/CD workflows

---

## 6. Tech Stack

| Layer | Tools |
|---|---|
| Backend API | FastAPI, Pydantic |
| Frontend | Streamlit |
| LLM | Groq, Llama 3.1 8B Instant |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Vector Store | ChromaDB |
| Keyword Search | BM25 |
| Reranking | CrossEncoder MiniLM |
| Analytics | Pandas |
| Database | Supabase PostgreSQL |
| Deployment | Docker, Docker Hub |
| CI/CD | GitHub Actions |
| Language | Python 3.11 |

---

## 7. Evaluation Summary

The project was evaluated using a manual + deterministic end-to-end evaluation approach.

| Area | Result |
|---|---|
| Semantic RAG QA | Passed |
| Exact ID/name lookup | Passed |
| Structured analytics | Passed |
| Query routing | Passed |
| Fallback behavior | Passed |
| Source attribution | Passed |
| API workflow | Passed |
| Streamlit workflow | Passed |
| Docker workflow | Passed |
| Feedback logging | Passed |

A separate `EVALUATION_REPORT.md` contains detailed test cases, expected behavior, pass criteria, limitations, and final assessment.

---

## 8. Results and Impact

| Dimension | Result |
|---|---|
| Placement readiness | 8.7 / 10 |
| Retrieval quality | Strong |
| Analytics correctness | Strong |
| Architecture quality | Strong |
| Deployment readiness | Good |
| Evaluation maturity | Good |

The strongest part of the project is the separation of **semantic retrieval** and **structured analytics**, which makes the system more reliable than a normal RAG chatbot.

---

## 9. Key Engineering Decisions

| Decision | Reason |
|---|---|
| Added BM25 with vector search | To handle exact IDs, names, and codes |
| Added Pandas Analytics Engine | To avoid LLM guessing in numerical queries |
| Added Query Router | To send each query to the correct pipeline |
| Added CrossEncoder reranking | To reduce irrelevant retrieved chunks |
| Added strict grounding prompt | To reduce hallucination |
| Added Supabase feedback | To track user feedback and failed questions |
| Added Docker | To make deployment reproducible |

---

## 10. Known Limitations

- No authentication or role-based access control yet
- No OCR for scanned PDFs
- Pandas analytics currently supports CSV/Excel, not PDF tables
- BM25 index is not yet persisted separately
- No multi-user workspace isolation
- Load testing is basic, not enterprise-scale stress testing

---

## 11. Future Improvements

- Add authentication and user-level workspaces
- Add persistent BM25 or OpenSearch
- Add OCR for scanned PDFs
- Add PDF table extraction into Pandas
- Add Redis caching
- Add automated evaluation pipeline
- Add real streaming through Server-Sent Events or WebSockets
- Add Kubernetes deployment for cloud scale

---

## 12. Interview Pitch

This project started as a simple RAG chatbot, but during testing I found two major production problems. First, vector search failed on exact identifiers like student IDs and names. Second, RAG hallucinated on numerical aggregation questions because LLMs are not reliable for dataframe-style computation. I solved this by building a hybrid architecture where semantic questions go through a Hybrid RAG pipeline, while analytical questions are routed to a Pandas engine for deterministic computation.

This makes the system more practical than a basic PDF chatbot and demonstrates retrieval engineering, LLM orchestration, structured analytics, API development, UI development, feedback logging, Docker deployment, and evaluation discipline.

---

## 13. Repository Documents

| File | Purpose |
|---|---|
| `README.md` | Full project explanation and setup |
| `EVALUATION_REPORT.md` | Evaluation test cases and final assessment |
| `docs/INTERVIEW_QA.md` | Interview questions and prepared answers |
| `CASE_STUDY.md` | Development journey and engineering decisions |
| `DOCKER.md` | Docker setup and deployment guide |
| `PROJECT_SUMMARY.md` | Short one-page project overview |
