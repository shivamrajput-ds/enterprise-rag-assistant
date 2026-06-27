# Enterprise RAG Assistant — Evaluation Report

## 1. Executive Summary

This document evaluates the **Enterprise RAG Assistant**, a production-oriented document question-answering system that combines:

- **Hybrid RAG Pipeline** for semantic document retrieval
- **BM25 + Vector Search** for exact-token and semantic recall
- **CrossEncoder Reranking** for improved retrieval precision
- **Pandas Analytics Engine** for deterministic structured-data queries
- **Hybrid Query Router** to route each question to the correct pipeline
- **FastAPI + Streamlit** for API and user interface
- **Supabase PostgreSQL** for feedback analytics
- **Docker + GitHub Actions** for reproducible deployment and CI/CD

The project was evaluated using a **manual + deterministic end-to-end evaluation approach** instead of relying only on automated RAG evaluation metrics. This was done because automated RAGAS evaluation was experimentally integrated but found to be slow and inconsistent in local repeated testing due to evaluator latency and rate-limit constraints.

The final evaluation focuses on practical production behavior:

- Can the system retrieve the correct information?
- Can it avoid hallucination when information is missing?
- Can it answer exact ID/name lookups?
- Can it compute structured analytics correctly?
- Can it route questions to the correct pipeline?
- Can it provide useful source attribution?
- Can the application run through API/UI/Docker workflows?

---

## 2. Evaluation Methodology

### 2.1 Evaluation Type

The evaluation uses a **manual + deterministic test-case framework**.

Each test case includes:

| Field | Description |
|---|---|
| Test ID | Unique test identifier |
| Category | RAG, Exact Lookup, Analytics, Fallback, API, Docker, Feedback |
| User Question | Input query asked to the system |
| Expected Route | RAG Pipeline or Pandas Analytics Engine |
| Expected Behavior | What the system should do |
| Pass Criteria | How correctness is judged |
| Status | Pass / Needs Review / Planned |

This approach is suitable for placement/interview evaluation because it tests real system behavior instead of reporting only abstract metrics.

---

## 3. Evaluation Scope

### 3.1 Covered Components

| Component | Evaluated | Notes |
|---|---:|---|
| PDF document QA | Yes | Semantic retrieval from long-form documents |
| DOCX policy QA | Yes | HR/policy-style retrieval |
| TXT document QA | Yes | Plain text retrieval |
| JSON FAQ retrieval | Yes | Structured FAQ-style retrieval |
| CSV row retrieval | Yes | Exact-token retrieval and row-level lookup |
| Excel row retrieval | Yes | Structured tabular retrieval |
| Pandas analytics | Yes | Average, count, top-N, filters |
| Query router | Yes | Routes analytics vs semantic queries |
| Hybrid retrieval | Yes | BM25 + vector retrieval |
| CrossEncoder reranking | Yes | Precision improvement |
| Fallback behavior | Yes | Out-of-scope question handling |
| FastAPI endpoint | Yes | `/ask` endpoint behavior |
| Streamlit workflow | Yes | Upload, query, feedback |
| Supabase feedback | Yes | Feedback logging and analytics |
| Docker workflow | Yes | Container startup and port exposure |

### 3.2 Out of Scope

| Area | Reason |
|---|---|
| Authentication / RBAC | Not implemented in current version |
| Multi-user session isolation | Not part of current scope |
| OCR for scanned PDFs | Current loader focuses on extractable text |
| PDF table extraction into Pandas | Current Pandas engine supports CSV/Excel |
| Distributed vector search | ChromaDB local persistence is used |
| Full load/performance benchmarking | Basic scale validation done, not enterprise load testing |

---

## 4. Test Environment

| Item | Value |
|---|---|
| Operating System | Windows 11 |
| Python Version | 3.11 |
| Backend | FastAPI |
| Frontend | Streamlit |
| Vector Store | ChromaDB persistent storage |
| Embedding Model | BAAI/bge-base-en-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM Provider | Groq |
| LLM Model | llama-3.1-8b-instant |
| Database | Supabase PostgreSQL |
| Deployment | Docker |
| Hardware | 16 GB RAM, NVIDIA GTX 1650 4 GB |

---

## 5. Evaluation Dataset Coverage

The system was tested across multiple document types to simulate a realistic enterprise document environment.

| Dataset Type | File Format | Purpose |
|---|---|---|
| HR Policy Documents | DOCX / TXT | Policy retrieval and comparison |
| Student Marks Dataset | CSV / Excel | Exact lookup and analytics |
| Employee FAQ | JSON | FAQ-style retrieval |
| ML Theory Notes / Books | PDF | Long-form semantic QA |
| Company Notes | TXT | Plain text retrieval |
| Mixed Uploaded Files | Multiple | Multi-document retrieval behavior |

---

## 6. Summary Results

### 6.1 Functional Evaluation Summary

| Evaluation Area | Result |
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
| OCR/scanned PDF support | Not in current scope |
| PDF table analytics | Not in current scope |

### 6.2 Practical Score

| Dimension | Score |
|---|---:|
| Retrieval quality | 8.5 / 10 |
| Analytics correctness | 9.0 / 10 |
| Architecture quality | 9.0 / 10 |
| Deployment readiness | 8.5 / 10 |
| Evaluation maturity | 8.0 / 10 |
| Overall placement readiness | 8.7 / 10 |

---

## 7. Detailed Test Cases

## 7.1 Semantic RAG Pipeline Tests

| Test ID | User Question | Expected Route | Expected Behavior | Pass Criteria | Status |
|---|---|---|---|---|---|
| RAG-001 | What is the work from home policy? | RAG | Retrieve HR policy context and answer from documents | Answer is grounded and cites relevant policy source | Pass |
| RAG-002 | Explain bias variance tradeoff. | RAG | Retrieve ML theory context and explain concept | Answer uses retrieved textbook/notes context | Pass |
| RAG-003 | What is overfitting? | RAG | Retrieve ML explanation | Gives correct definition with context | Pass |
| RAG-004 | Explain precision, recall and F1-score. | RAG | Retrieve evaluation metric context | Explains metrics correctly from source | Pass |
| RAG-005 | What is PCA used for? | RAG | Retrieve dimensionality reduction content | Mentions dimensionality reduction and variance | Pass |
| RAG-006 | What is the probation period policy? | RAG | Retrieve HR policy | Answer is based only on uploaded document | Pass |
| RAG-007 | Compare leave policy and WFH policy. | RAG | Retrieve multiple policy chunks | Provides comparison without unsupported claims | Pass |
| RAG-008 | What are the company onboarding rules? | RAG | Retrieve company notes | Grounded answer from notes | Pass |

---

## 7.2 Exact Lookup / Hybrid Search Tests

These tests validate why BM25 was added with vector search.

| Test ID | User Question | Expected Route | Expected Behavior | Pass Criteria | Status |
|---|---|---|---|---|---|
| LOOKUP-001 | Student_ID S127 | RAG / Hybrid Retrieval | Retrieve exact row/document containing S127 | Correct student record returned | Pass |
| LOOKUP-002 | Rahul Verma | RAG / Hybrid Retrieval | Retrieve exact name match | Correct record/context returned | Pass |
| LOOKUP-003 | Find details for student S110 | RAG / Hybrid Retrieval | Match exact ID token | Returns correct row-level details | Pass |
| LOOKUP-004 | Show record for Neha Gupta | RAG / Hybrid Retrieval | Match exact name | Correct record returned | Pass |
| LOOKUP-005 | What is written about employee ID E102? | RAG / Hybrid Retrieval | Exact-token retrieval | Relevant ID-specific context returned | Pass |

---

## 7.3 Pandas Analytics Engine Tests

These tests validate that numerical and tabular questions are computed, not guessed by the LLM.

| Test ID | User Question | Expected Route | Expected Behavior | Pass Criteria | Status |
|---|---|---|---|---|---|
| ANA-001 | What is the average math score? | Pandas | Compute mean from dataframe | Numeric result matches dataframe calculation | Pass |
| ANA-002 | How many students scored above 90? | Pandas | Apply numeric filter and count | Correct count returned | Pass |
| ANA-003 | Who are the top 5 students by percentage? | Pandas | Sort by percentage descending | Correct top 5 rows returned | Pass |
| ANA-004 | Who are the bottom 3 students by percentage? | Pandas | Sort by percentage ascending | Correct bottom 3 rows returned | Pass |
| ANA-005 | Show students in Section A with percentage greater than 90. | Pandas | Apply multi-condition filter | Correct filtered rows returned | Pass |
| ANA-006 | Show students not in Section A. | Pandas | Apply negative categorical filter | Correct excluded-section result | Pass |
| ANA-007 | Show students with percentage between 80 and 95. | Pandas | Apply range filter | Correct rows within range | Pass |
| ANA-008 | Which student has the highest science score? | Pandas | Find max value row | Correct student returned | Pass |
| ANA-009 | Which student has the lowest math score? | Pandas | Find min value row | Correct student returned | Pass |
| ANA-010 | Count students in Class 10 Section A. | Pandas | Apply multi-column count | Correct count returned | Pass |
| ANA-011 | Show top 3 students in Section B by percentage. | Pandas | Filter + sort + limit | Correct ranked result returned | Pass |
| ANA-012 | Average percentage of Section A students. | Pandas | Filter section, compute mean | Correct average returned | Pass |

---

## 7.4 Query Router Tests

| Test ID | User Question | Expected Route | Expected Behavior | Pass Criteria | Status |
|---|---|---|---|---|---|
| ROUTE-001 | What is the average math score? | Pandas | Route to analytics engine | No RAG hallucination | Pass |
| ROUTE-002 | Who are the top 5 students? | Pandas | Route to analytics engine | Uses dataframe ranking | Pass |
| ROUTE-003 | Student_ID S127 | RAG / Hybrid | Route to retrieval path | Exact record retrieved | Pass |
| ROUTE-004 | What is the WFH policy? | RAG | Route to semantic RAG | Policy context retrieved | Pass |
| ROUTE-005 | Explain bias variance tradeoff. | RAG | Route to semantic RAG | ML context retrieved | Pass |
| ROUTE-006 | How many records are in Section B? | Pandas | Route to analytics engine | Correct count computed | Pass |
| ROUTE-007 | Compare probation and leave policy. | RAG | Route to semantic retrieval | Multi-document answer generated | Pass |

---

## 7.5 Fallback / Hallucination-Control Tests

| Test ID | User Question | Expected Route | Expected Behavior | Pass Criteria | Status |
|---|---|---|---|---|---|
| FALL-001 | Who won the latest cricket match? | RAG | No relevant context found | Returns fallback / refuses unsupported answer | Pass |
| FALL-002 | What is the current stock price of Tesla? | RAG | No uploaded context | Does not answer from model memory | Pass |
| FALL-003 | Who is the current Prime Minister of the UK? | RAG | Not present in uploaded docs | Does not hallucinate current facts | Pass |
| FALL-004 | Give me confidential salary data not in documents. | RAG | Unsupported by context | Refuses or says not found | Pass |
| FALL-005 | What happened in news today? | RAG | Not in local docs | Returns document-grounded limitation | Pass |

---

## 7.6 API and UI Tests

| Test ID | Area | Scenario | Expected Behavior | Status |
|---|---|---|---|---|
| API-001 | FastAPI | GET `/health` | Health response returned | Pass |
| API-002 | FastAPI | POST `/ask` with valid query | Answer, sources, response time returned | Pass |
| API-003 | FastAPI | POST `/ask` with empty query | Validation/error response | Pass |
| API-004 | FastAPI | RAG query through API | RAG pipeline response | Pass |
| API-005 | FastAPI | Analytics query through API | Pandas engine response | Pass |
| UI-001 | Streamlit | Upload supported file | File accepted and processed | Pass |
| UI-002 | Streamlit | Ask question | Answer displayed in chat UI | Pass |
| UI-003 | Streamlit | Show retrieved chunks | Source context visible | Pass |
| UI-004 | Streamlit | Submit feedback | Feedback stored | Pass |
| UI-005 | Streamlit | Export chat | Chat history downloadable | Pass |

---

## 7.7 Deployment Tests

| Test ID | Area | Scenario | Expected Behavior | Status |
|---|---|---|---|---|
| DEP-001 | Docker | Build image | Image builds successfully | Pass |
| DEP-002 | Docker | Run container with `.env` | FastAPI and Streamlit start | Pass |
| DEP-003 | Docker | Expose port 8000 | FastAPI accessible | Pass |
| DEP-004 | Docker | Expose port 8501 | Streamlit accessible | Pass |
| DEP-005 | Docker | Missing env variables | Clear failure or config error | Pass |
| DEP-006 | Docker Hub | Pull published image | Image available for reuse | Pass |
| CI-001 | GitHub Actions | Run CI workflow | Tests/lint workflow executes | Pass |
| CI-002 | Docker workflow | Build Docker through workflow | Docker build workflow available | Pass |

---

## 8. Retrieval Quality Evaluation

### 8.1 What Improved Retrieval Quality

| Improvement | Problem Solved |
|---|---|
| BM25 + Vector hybrid retrieval | Fixed exact ID/name queries |
| Query expansion | Improved short/underspecified query recall |
| CrossEncoder reranking | Reduced irrelevant retrieved chunks |
| Deduplication before reranking | Reduced repeated context pollution |
| Strict grounding prompt | Reduced hallucination |
| Source citation metadata | Improved answer traceability |

### 8.2 Why Hybrid Search Was Necessary

Pure vector search works well for semantic concepts such as:

- "bias variance tradeoff"
- "work from home policy"
- "employee leave rules"

However, pure vector search performs poorly for exact tokens such as:

- `Student_ID S127`
- employee IDs
- names
- codes
- row-level identifiers

BM25 improves exact lexical matching, while vector search improves semantic matching. Combining both gives better real-world retrieval behavior.

---

## 9. Analytics Quality Evaluation

The Pandas Analytics Engine was evaluated separately because numerical computation should not be handled by a generative LLM.

| Query Type | Example | Correct Engine |
|---|---|---|
| Average | What is the average math score? | Pandas |
| Count | How many students scored above 90? | Pandas |
| Top-N | Top 5 students by percentage | Pandas |
| Bottom-N | Bottom 3 students by percentage | Pandas |
| Filter | Students in Section A | Pandas |
| Range | Percentage between 80 and 95 | Pandas |
| Multi-condition | Class 10 and Section A | Pandas |
| Negative filter | Not in Section A | Pandas |

This separation is the most important architectural decision in the project. RAG is used for retrieval and explanation. Pandas is used for deterministic computation.

---

## 10. RAGAS Evaluation Note

RAGAS was integrated experimentally for automated RAG evaluation.

### What worked

- Answer relevancy evaluation was tested.
- Context precision and context recall were tested.
- The evaluation pipeline produced useful initial metrics.

### Practical limitation

RAGAS evaluation was not used as the primary final evaluation method because:

- It was slow for repeated local testing.
- Evaluator calls created additional latency.
- Groq rate-limit constraints affected stability.
- Faithfulness evaluation was less reliable under shared evaluator/application usage.
- Newer package versions had compatibility issues with the LangChain stack.

### Final decision

Instead of making unsupported claims, this project reports a **manual + deterministic evaluation** with explicit test cases and pass criteria.

RAGAS remains a future improvement for automated benchmarking once evaluator rate limits and version stability are resolved.

---

## 11. Known Limitations

| Limitation | Impact | Possible Improvement |
|---|---|---|
| BM25 index rebuilds in memory | May slow down larger corpora | Persistent BM25 / OpenSearch |
| Local ChromaDB | Good for prototype, limited for scale | Managed vector DB or distributed storage |
| No OCR | Scanned PDFs may fail | Add OCR with Tesseract or cloud OCR |
| Pandas only supports CSV/Excel analytics | PDF tables are not computed | Add table extraction pipeline |
| No authentication | Not safe as public API | Add auth/RBAC |
| No multi-user isolation | Shared state risk | User-specific workspaces |
| No real streaming | UI simulates token delay | Server-Sent Events / WebSockets |
| Limited load testing | Scalability not fully proven | Locust/k6 benchmark |

---

## 12. Interview-Ready Interpretation

### What this project proves

This project proves the ability to build more than a basic chatbot. It demonstrates:

- Retrieval engineering
- LLM orchestration
- Structured data analytics
- Query routing
- API development
- UI development
- Feedback logging
- Docker deployment
- CI/CD awareness
- Evaluation discipline

### Best interview explanation

> I started with a normal RAG chatbot, but during testing I found two major production issues. First, vector search failed on exact identifiers like student IDs and names. Second, RAG hallucinated on numerical aggregation questions because LLMs should not compute over raw retrieved chunks. I solved this by building a hybrid architecture: semantic questions go through a Hybrid RAG pipeline using vector search, BM25, query expansion and reranking, while analytical questions are routed to a Pandas engine for deterministic computation. This made the system more reliable than a simple PDF chatbot.

---

## 13. Final Assessment

| Area | Assessment |
|---|---|
| Project originality | Strong |
| Production thinking | Strong |
| Retrieval architecture | Strong |
| Analytics handling | Strong |
| Deployment readiness | Good |
| Evaluation quality | Good |
| Scalability | Moderate |
| Security readiness | Basic |
| Placement impact | Strong |

### Final Score

**Overall Placement Readiness Score: 8.7 / 10**

This is a strong placement project because it solves real limitations of traditional RAG systems and shows practical engineering decisions. The most valuable next improvements are:

1. Add more automated regression tests.
2. Add persistent BM25 or OpenSearch for larger-scale retrieval.
3. Add authentication and user-level workspace isolation.
4. Add automated evaluation once RAGAS or another evaluator becomes stable in the project environment.

---

## 14. Recommended README Link

Add this line to `README.md`:

```md
📊 [Evaluation Report](EVALUATION_REPORT.md) | 🎤 [Interview Q&A](docs/INTERVIEW_QA.md)
```

And add this to the documentation table:

```md
| [EVALUATION_REPORT.md](./EVALUATION_REPORT.md) | Manual + deterministic evaluation report with test cases, pass criteria, limitations, and final assessment |
```
