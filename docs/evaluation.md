# RAG Assistant — Evaluation Report

📺 [Watch Demo](https://youtu.be/Rvdz9DKtz5o?si=GcMIR7nWABJQYCR1) | 🐳 [Docker Hub](https://hub.docker.com/repository/docker/shivamrajput130/enterprise-rag-assistant/general) | 💻 [GitHub](https://github.com/shivamrajput130/enterprise-rag-assistant)

**Evaluation date:** June 2026
**Build evaluated:** Dockerized build, published to Docker Hub

**Models:**
| Component | Model |
|---|---|
| Embeddings | `BAAI/bge-base-en-v1.5` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM (application) | `llama-3.1-8b-instant` via Groq |
| LLM (query router) | `llama-3.1-8b-instant` via Groq — part of Hybrid Query Router (schema-aware analytics detection + semantic routing) |
| RAGAS evaluator | `llama-3.1-8b-instant` via Groq — same provider, known conflict |

**Test environment:**
- OS: Windows 11
- CPU: Intel Core i5-12450H, RAM: 16 GB, GPU: GTX 1650
- Python 3.11, Local ChromaDB, Supabase PostgreSQL (cloud)

---

## Evaluation Approach

Three testing methods, in order of reliability:

1. **E2E keyword tests** — automated, catches regressions across both RAG and Pandas Engine paths
2. **Manual end-to-end testing** — full pipeline via Streamlit, catches routing errors and formatting issues
3. **RAGAS automated metrics** — experimental, documents retrieval quality direction but unstable with Groq as evaluator

> **Evaluation Bias Note:** All test cases were designed by the project author after building and debugging the system. The test set reflects known failure modes. Independent testing was not performed. These tests validate functionality, not statistical performance across an unbiased query distribution.

> **Reproducibility Note:** LLM outputs are non-deterministic. Results vary across runs with identical inputs.

---

## Corpus Description

**Controlled test corpus:**

| File | Type | Content |
|---|---|---|
| `ml_theory.pdf` | PDF | Machine learning concepts |
| `employee_faq.json` | JSON | HR FAQ entries |
| `hr_policy.docx` | DOCX | Company HR policies |
| `company_notes.txt` | TXT | Company values and processes |
| `student_marks.csv` | CSV | Student records with IDs and scores |

Approximately 80–120 chunks generated from this corpus.

**Real-world scale test:**

Two full ML/data science textbooks (~1800 pages combined):
- Total documents loaded: **1129**
- Total chunks created: **2538**
- Retrieval, source citations with page numbers, and fallback all worked correctly
- Re-upload of already-ingested files triggered "already exists. Skipping." correctly
- Average response time on warm queries: ~3.2 seconds

---

## E2E Test Suite

```csv
question,expected_keyword,pipeline
When is salary credited?,last working day,RAG
What is the work from home policy?,2 days per week,RAG
What is overfitting?,overfitting,RAG
What is the probation period?,6 months,RAG
Student_ID S127,Nikhil Sharma,RAG (BM25)
Who won IPL 2026?,don't know,RAG (fallback)
What is the average math score?,average,Pandas
Who is the top student?,percentage,Pandas
How many students scored above 90?,count,Pandas
```

Run:
```bash
python tests/test_e2e.py
```

The suite covers both pipelines. The Pandas Engine cases were added after the Query Router was built — earlier E2E tests only covered the RAG path.

**Coverage gaps:** Adversarial queries, empty queries, very large inputs, router misclassification edge cases, and corrupted file behavior are not in the automated suite.

---

## Section 1 — RAG Pipeline Evaluation

### Manual Results

| # | Query | Source | Expected | Status |
|---:|---|---|---|---|
| 1 | What is overfitting? | ml_theory.pdf | Explain overfitting | Passed |
| 2 | Explain bias variance tradeoff | ml_theory.pdf | High bias, high variance, balance | Passed |
| 3 | Types of gradient descent | ml_theory.pdf | SGD, Mini-batch, Adam, AdaGrad | Passed |
| 4 | When is salary credited? | employee_faq.json | Last working day of every month | Passed |
| 5 | What is the probation period? | employee_faq.json | 6 months from joining | Passed |
| 6 | What is the WFH policy? | hr_policy.docx | 2 days/week, approval required, not during probation | Passed |
| 7 | What are the company core values? | company_notes.txt | Four core values listed | Passed |
| 8 | Explain the incident response process | company_notes.txt | Detection, assessment, communication, resolution, documentation | Passed |
| 9 | Student_ID S127 | student_marks.csv | Complete student record | Passed after hybrid search + prompt fix |
| 10 | Rahul Verma | student_marks.csv | Matching student row | Passed |
| 11 | Communication tools + first day onboarding | employee_faq.json + company_notes.txt | Combined answer | Passed |
| 12 | Compare probation period and WFH policy | employee_faq.json + hr_policy.docx | Structured comparison | Passed |
| 13 | Who won IPL 2026? | Not in documents | Fallback response | Passed |

**Honest note:** 13/13 passing reflects test suite design as much as system quality. Tests were written after fixing known bugs.

### Actual Retrieval Failure Example

**Query:** `Student_ID S127` — before BM25 was added

Vector search retrieved this chunk:
```
Source: hr_policy.docx
...all employees are required to maintain confidentiality of personal records
including identification numbers assigned during the onboarding process...
```

The phrase "identification numbers" caused the vector search to retrieve an HR policy chunk. `S127` has no semantic meaning — it embedded near unrelated content about record-keeping.

System answer:
```
The company policy requires maintaining confidentiality of personal records
including identification numbers.
```

Completely wrong. No retrieval failure signal. Confident wrong answer.

Fix: BM25 added. `s127` as exact token matched the CSV row directly.

After fix:
```
Student ID: S127 | Name: Nikhil Sharma
Class: 10 | Section: A
Math: 96 | Science: 95 | English: 98 | Hindi: 93 | Social Science: 99
Total: 481 | Percentage: 96.2 | Grade: A+
```

### Hybrid Search Comparison

| Query | Vector Only | Hybrid (Vector + BM25) |
|---|---|---|
| `S127` | Failed — wrong document retrieved | Passed — BM25 exact match |
| `Rahul Verma` | Unreliable | Passed |
| `When is salary credited?` | Passed | Passed |
| `What is overfitting?` | Passed | Passed |
| `WFH policy` | Passed | Passed |

### Reranker Impact

Without reranker — "What is the probation period?":
- Sources: `hr_policy.docx`, `employee_faq.json`, `ml_theory.pdf` ← irrelevant

With reranker — same query:
- Sources: `employee_faq.json`, `hr_policy.docx`

The CrossEncoder scored `ml_theory.pdf` chunks near zero. Lowering `reranker_top_k` from 5 to 3 removed this pollution on focused queries. Occasional pollution still appears on genuinely broad queries.

---

## Section 2 — Pandas Analytics Engine Evaluation

These queries could not be answered before the Pandas Engine and Query Router were added. The RAG pipeline either hallucinated computed values or returned irrelevant text chunks.

### Manual Results

| # | Query | Operation | Expected | Status |
|---:|---|---|---|---|
| 1 | What is the average math score? | `mean()` | Correct mean | Passed |
| 2 | What is the average percentage? | `mean()` | Correct mean | Passed |
| 3 | Who is the top student by percentage? | `nlargest(1)` | Correct student name | Passed |
| 4 | Who are the top 3 students? | `nlargest(3)` | Correct 3 records | Passed |
| 5 | Who are the bottom 3 students? | `nsmallest(3)` | Correct 3 records | Passed |
| 6 | How many students scored above 90 in math? | conditional count | Correct count | Passed |
| 7 | Show students in class 10 section A | multi-condition filter | Correct rows | Passed |
| 8 | Show students who are not in class 10 | negative filter | Correct rows | Passed after bug fix |
| 9 | Show students with math between 80 and 95 | range filter | Correct rows | Passed |
| 10 | Who has the highest total marks? | `nlargest(1)` on total | Correct student | Passed |

**Negative filter bug:** Query 8 initially returned all students including class 10 students due to a boolean logic error in `tabular_query.py`. Fixed before final evaluation.

### Before vs After — Aggregation Query

**Query:** "What is the average math score?"

**Before Pandas Engine (RAG pipeline):**
```
Based on the provided documents, the math scores include 96, 88, 72, and others.
The average appears to be around 85.
```
Hallucinated estimate. Incorrect.

**After Pandas Engine:**
```
The average math score is 87.4.
```
Computed directly from the DataFrame. Correct and deterministic.

### Query Router Accuracy

| Query | Expected Route | Actual Route | Correct? |
|---|---|---|---|
| What is the average score? | Pandas | Pandas | ✅ |
| Who is the top student? | Pandas | Pandas | ✅ |
| What is overfitting? | RAG | RAG | ✅ |
| Student_ID S127 | RAG | RAG | ✅ |
| WFH policy | RAG | RAG | ✅ |
| Compare probation and WFH | RAG | RAG | ✅ |
| Show students above 90 | Pandas | Pandas | ✅ |
| Tell me about student performance | Ambiguous | RAG | Acceptable |

All tested queries routed correctly. The ambiguous case ("tell me about student performance") went to RAG — acceptable since BM25 can still retrieve relevant CSV records.

---

## Section 3 — RAGAS Evaluation

### Dependency Issues

RAGAS integration exposed significant dependency fragility:

| Package | Issue |
|---|---|
| `ragas` | Breaking API changes post-0.2.10 |
| `langchain-community` | Callback system incompatibility with newer RAGAS |
| `langchain-core` | Required version pinning |
| `vertexai` import errors | Surfaced from RAGAS internal dependencies even when not used |

`ragas==0.2.10` was the stable version. This was one of the most time-consuming parts of the project — the evaluation tooling, not the pipeline itself.

### Results

| Run | Answer Relevancy | Context Precision | Context Recall | Faithfulness |
|---|---|---|---|---|
| Run 1 | 0.5785 | 1.0000 | 0.7500 | not reliably measured |
| Run 2 | 0.8822 | 1.0000 | 1.0000 | not reliably measured |

**Interpretation:**

`context_precision: 1.0` in both runs — when retrieval worked, retrieved chunks were relevant. Observed on a small test set; not statistically representative of performance across all query types.

`answer_relevancy` varied from 0.58 to 0.88. The lower run included queries where the system answered correctly but with surrounding context RAGAS scored as irrelevant. The metric is LLM-dependent and varies between runs.

`faithfulness: not reliably measured` — Groq rate limits caused evaluator LLM failures during scoring. The application and the RAGAS evaluator shared the same Groq quota, creating rate limit conflicts. This is an evaluation infrastructure limitation, not evidence of hallucination.

> **Lesson learned:** Do not use the same rate-limited LLM provider for the application and the evaluation harness simultaneously.

---

## Section 4 — Docker Runtime Validation

| Test | Result |
|---|---|
| Docker image builds without error | Passed |
| FastAPI starts correctly in container | Passed |
| Streamlit starts correctly in container | Passed |
| GROQ_API_KEY loaded from `--env-file` | Passed |
| Supabase connection from container | Passed |
| ChromaDB persists within container volume | Passed |
| Document upload and ingestion in container | Passed |
| RAG query returns correct answer | Passed |
| Pandas Engine query returns correct answer | Passed |
| Fallback returns correct string | Passed |
| Re-upload skips existing documents | Passed |

---

## Observed Latency

Measured on: Intel i5-12450H, 16 GB RAM, GTX 1650, local deployment (not Docker).

| Scenario | Observed |
|---|---|
| Cold start — first query (model loading) | ~20 sec |
| Warm RAG query — average | ~3 sec |
| Warm RAG query — fastest | ~1.7 sec |
| Warm Pandas Engine query | ~0.5–1 sec |
| Reranker inference alone | ~0.3–0.8 sec |

Pandas Engine queries are faster than RAG queries because they skip vector search, BM25, and LLM generation — the answer is computed directly from the DataFrame.

Latency is not currently tested automatically.

---

## Summary by Area

| Area | Result |
|---|---|
| PDF concept questions | Consistent across short notes and 900-page textbooks |
| FAQ questions | Consistent |
| HR policy questions | Consistent |
| TXT process questions | Consistent |
| CSV exact record lookup | Working after hybrid search + prompt fix |
| Multi-document questions | Working |
| Out-of-document fallback | Working — tested in E2E and on real queries |
| CSV/Excel aggregations | Working via Pandas Engine |
| CSV/Excel rankings | Working |
| CSV/Excel filters | Working, including negative and range filters |
| Query Router | Correct on all tested queries |
| Source citations with page numbers | Working |
| Docker deployment | Validated |
| RAGAS Context Precision | Observed 1.00 on evaluated test set — small sample, not statistically representative |
| RAGAS Answer Relevancy | 0.58–0.88 — LLM-dependent variance |
| RAGAS Faithfulness | Not reliably measured |

---

## Known Gaps

| Gap | Detail |
|---|---|
| Test set selection bias | Cases designed after fixing known bugs |
| No latency regression tests | Slow reranker or expansion timeout not auto-caught |
| RAGAS faithfulness unmeasured | Requires non-rate-limited evaluator |
| No router misclassification stress test | Edge cases on ambiguous queries not systematically tested |
| Pandas Engine: PDF tables not supported | Tables in PDF/DOCX not extracted — go through RAG as text |
| No corrupted file tests | Broken PDF, empty CSV, invalid JSON behavior untested |
| No large file stress test | 500-page PDF or 50,000-row CSV untested |
| No concurrency test | Multi-user Docker behavior not measured |
| No authentication or RBAC | API is open — any user can query, ingest, or delete documents |
| No adversarial prompt tests | Prompt injection or jailbreak attempts not tested |

---

## Honest Summary

This project demonstrates a working two-pipeline document QA system: hybrid RAG for semantic queries and a Pandas Analytics Engine for tabular queries, with a **Hybrid Query Router** (schema-aware analytics detection + semantic routing) separating the two paths automatically.

What works reliably on this corpus: semantic retrieval, exact entity lookup, multi-document questions, aggregations, rankings, filters, fallback handling, and Docker deployment.

What is not validated: large-scale performance beyond 2538 chunks, concurrent users, adversarial robustness, and behavior on document types significantly different from the test corpus.

The strongest result is observed `context_precision: 1.0` across evaluated runs — retrieved chunks were relevant when retrieval worked. The most important failure during development was the vector search miss on `S127`, which produced a confident wrong answer with no retrieval failure signal. That failure motivated the hybrid search addition, which was the single largest quality improvement in the RAG pipeline.
