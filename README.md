# 🔍 RAG Evaluation Framework

A comprehensive Retrieval-Augmented Generation (RAG) pipeline with a full evaluation suite — combining LLM-as-judge metrics (Correctness, Relevance, Groundedness, Retrieval Relevance) with traditional NLP metrics (BLEU, ROUGE-L, Perplexity) — powered by LangChain, Astra DB, OpenAI, and LangSmith.

---

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack & Libraries](#tech-stack--libraries)
- [Project Workflow](#project-workflow)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. Vector Store Setup](#2-vector-store-setup)
  - [3. RAG Bot](#3-rag-bot)
  - [4. Evaluation Suite](#4-evaluation-suite)
- [Evaluation Metrics — Deep Dive](#evaluation-metrics--deep-dive)
  - [LLM-as-Judge Metrics](#llm-as-judge-metrics)
  - [Traditional NLP Metrics](#traditional-nlp-metrics)
- [Evaluation Results](#evaluation-results)
- [Dataset Used](#dataset-used)
- [Environment Setup](#environment-setup)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Key Design Decisions](#key-design-decisions)

---

## Project Overview

This project builds a production-grade RAG system that answers questions from a curated set of technical blog posts, and then rigorously evaluates the system's output quality using **seven distinct metrics** across two evaluation paradigms.

The core idea: rather than just building a RAG pipeline and hoping it works, this project treats evaluation as a first-class concern — running every answer through both semantic LLM judges and deterministic string-overlap metrics to get a 360° view of system performance.

**Source documents used:**
- [LLM-Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — Lilian Weng
- [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) — Lilian Weng
- [Adversarial Attacks on LLMs](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/) — Lilian Weng

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                            │
│                                                                 │
│  Web Pages  ──►  Text Splitter  ──►  OpenAI Embeddings          │
│                                           │                     │
│                                           ▼                     │
│                                    Astra DB Vector Store        │
│                                           │                     │
│  User Question  ──►  Retriever (k=4)  ◄──┘                      │
│                           │                                     │
│                           ▼                                     │
│                   GPT-4o-mini (LLM)  ──►  Answer                │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION SUITE                            │
│                                                                 │
│  LLM-as-Judge               │  Traditional NLP                  │
│  ─────────────────          │  ────────────────────             │
│  ✓ Correctness              │  ✓ BLEU Score                     │
│  ✓ Relevance                │  ✓ ROUGE-L Score                  │
│  ✓ Groundedness             │  ✓ Perplexity (GPT-2)             │
│  ✓ Retrieval Relevance      │                                   │
│                             │                                   │
│  Tracked & logged via LangSmith                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack & Libraries

| Library / Service | Version / Model | Purpose |
|---|---|---|
| **`langchain`** | Latest | Core orchestration framework for chaining LLM calls, retrieval, and prompts |
| **`langchain-openai`** | Latest | OpenAI integration — embeddings (`text-embedding-3-small`) and chat models (`gpt-4o-mini`, `gpt-4o`) |
| **`langchain-community`** | Latest | `WebBaseLoader` for scraping web pages as LangChain documents |
| **`langchain-astradb`** | Latest | Connector between LangChain and DataStax Astra DB vector store |
| **`astrapy`** | Latest | Low-level Python client for Astra DB's Data API |
| **`cassio`** | Latest | Cassandra/Astra DB utility library (connection helpers) |
| **`langsmith`** | Latest | Experiment tracking, dataset management, and evaluation result logging |
| **`openai`** | Latest (via langchain-openai) | Underlying provider for LLM calls and embeddings |
| **`evaluate`** | HuggingFace | Loads and computes BLEU and ROUGE metrics |
| **`rouge-score`** | Latest | Backend scorer for ROUGE metric computation |
| **`nltk`** | Latest | Tokenisation used internally by BLEU metric |
| **`transformers`** | HuggingFace | GPT-2 model and tokeniser for perplexity calculation |
| **`torch`** | Latest | PyTorch tensor operations; GPU acceleration for perplexity if available |
| **`datasets`** | HuggingFace | Dataset utilities (used as an evaluate dependency) |
| **`pandas`** | Latest | Converts LangSmith experiment results to a readable DataFrame |
| **`python-dotenv`** | Latest | Loads `.env` file for API keys |

---

## Project Workflow

### 1. Data Ingestion

Three technical blog posts from Lilian Weng's website are loaded using `WebBaseLoader`:

```python
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]
docs = [WebBaseLoader(url).load() for url in urls]
```

Documents are then split using `RecursiveCharacterTextSplitter` with tiktoken encoding:
- **Chunk size:** 250 tokens
- **Chunk overlap:** 25 tokens

This chunking strategy balances context window use against retrieval precision — smaller chunks retrieve more targeted passages, while the 25-token overlap prevents sentences from being cut mid-thought.

### 2. Vector Store Setup

Chunks are embedded using OpenAI's `text-embedding-3-small` model and stored in **Astra DB** (DataStax's serverless cloud vector database):

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="rag_evaluation_collection",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

The retriever fetches the **top 4 most semantically similar** chunks for every query.

### 3. RAG Bot

The `rag_bot` function is the core inference unit, decorated with `@traceable()` for automatic LangSmith logging:

```python
@traceable()
def rag_bot(question: str) -> dict:
    docs = retriever.invoke(question)
    docs_string = " ".join(doc.page_content for doc in docs)
    # System prompt + retrieved context passed to GPT-4o-mini
    ai_msg = llm.invoke([system_message, user_message])
    return {"answer": ai_msg.content, "documents": docs}
```

The LLM is instructed to answer in **three sentences maximum**, staying concise and grounded in the retrieved documents only.

### 4. Evaluation Suite

A benchmark dataset of 3 question-answer pairs is created in LangSmith, covering the three source documents. The full evaluation then runs every question through `rag_bot` and scores the result with all 7 evaluators simultaneously via `client.evaluate(...)`.

---

## Evaluation Metrics — Deep Dive

### LLM-as-Judge Metrics

These four metrics use GPT-4o or GPT-4o-mini as an automated "teacher" to grade responses. Each evaluator returns a boolean `True` / `False`.

---

#### ✅ Correctness
**What it measures:** Factual accuracy of the RAG answer vs. a ground-truth reference answer.

**How it works:** The grader LLM (GPT-4o-mini) receives the question, ground-truth answer, and the RAG bot's answer. It follows a strict rubric:
- Grade only on factual accuracy relative to ground truth
- Extra correct information is acceptable
- Conflicting statements → False

**Requires ground truth:** Yes

**Output schema:**
```python
class CorrectnessGrade(TypedDict):
    explanation: str   # Step-by-step reasoning
    correct: bool      # Final verdict
```

---

#### ✅ Relevance
**What it measures:** Whether the RAG answer actually addresses and helps answer the user's question — without needing a reference answer.

**How it works:** The grader LLM (GPT-4o) receives only the question and the RAG answer, then checks:
- Is the answer concise and on-topic?
- Does it genuinely help answer the question?

**Requires ground truth:** No

**Output schema:**
```python
class RelevanceGrade(TypedDict):
    explanation: str
    relevant: bool
```

---

#### ✅ Groundedness
**What it measures:** Whether the RAG answer is supported by the retrieved documents — i.e., does the model hallucinate?

**How it works:** The grader LLM (GPT-4o) is given the retrieved document chunks and the answer. It checks:
- Is every claim in the answer traceable to the provided facts?
- Does the answer introduce information not found in the documents?

**Requires ground truth:** No

**Output schema:**
```python
class GroundedGrade(TypedDict):
    explanation: str
    grounded: bool
```

---

#### ✅ Retrieval Relevance
**What it measures:** Whether the retrieved document chunks are actually relevant to the question — evaluating the retrieval component in isolation.

**How it works:** The grader LLM (GPT-4o) receives the question and retrieved facts. It applies a lenient standard:
- Any semantic or keyword overlap → relevant (True)
- Only flags False if documents are completely unrelated

**Requires ground truth:** No

**Output schema:**
```python
class RetrievalRelevanceGrade(TypedDict):
    explanation: str
    relevant: bool
```

---

### Traditional NLP Metrics

These three metrics are deterministic, reference-based, and computed using HuggingFace `evaluate` and GPT-2.

---

#### 📊 BLEU Score
**What it measures:** Exact n-gram (word/phrase) overlap between the generated answer and the reference answer.

**Range:** 0.0 → 1.0 (higher is more word-for-word similar)

**Interpretation:**
- High BLEU → answer uses similar phrasing to the reference
- Low BLEU → different wording, even if semantically correct
- Known limitation: BLEU penalises paraphrasing, so a fluent answer can still score low

**Computed via:** `evaluate.load("bleu")`

---

#### 📊 ROUGE-L Score
**What it measures:** Longest Common Subsequence (LCS) overlap between generated and reference answers — capturing content and structural similarity without requiring exact contiguous matches.

**Range:** 0.0 → 1.0 (higher is more content-similar)

**Interpretation:**
- Higher ROUGE-L → answer covers the same key content as the reference
- More forgiving than BLEU for paraphrasing
- ROUGE-L (vs ROUGE-1/2) focuses on sentence-level structure

**Computed via:** `evaluate.load("rouge")`

---

#### 📊 Perplexity
**What it measures:** How "surprised" GPT-2 is by the generated text — a proxy for fluency and linguistic naturalness.

**Range:** 1.0 → ∞ (lower is better / more fluent)

**Interpretation:**
- Low perplexity → confident, fluent, natural-sounding text
- High perplexity → unusual phrasing, fragmented sentences, or incoherent text
- Does NOT measure factual accuracy — only linguistic quality

**Computed via:** GPT-2 (`GPT2LMHeadModel`) from HuggingFace Transformers, with CUDA acceleration if available.

```python
def calculate_perplexity(text: str) -> float:
    encodings = perplexity_tokenizer(text, return_tensors="pt")
    outputs = perplexity_model(input_ids, labels=input_ids)
    return float(torch.exp(outputs.loss).item())
```

---

## Evaluation Results

The evaluation was run on 3 questions from the benchmark dataset. Results were logged to LangSmith under the experiment prefix `comprehensive-rag-eval`.

### Benchmark Questions

| # | Question | Ground Truth Answer |
|---|---|---|
| 1 | How does the ReAct agent use self-reflection? | ReAct integrates reasoning and acting, performing actions using tools like Wikipedia search and reasoning about the outputs. |
| 2 | What are the types of biases that can arise with few-shot prompting? | (1) Majority label bias, (2) Recency bias, (3) Common token bias |
| 3 | What are five types of adversarial attacks? | Token manipulation, Gradient-based attack, Jailbreak prompting, Human red-teaming, Model red-teaming |

### Expected Metric Ranges

| Metric | Type | Expected Range | What "Good" Looks Like |
|---|---|---|---|
| **Correctness** | LLM Judge (bool) | True / False | True → factually accurate vs. ground truth |
| **Relevance** | LLM Judge (bool) | True / False | True → answer addresses the question |
| **Groundedness** | LLM Judge (bool) | True / False | True → no hallucination vs. retrieved docs |
| **Retrieval Relevance** | LLM Judge (bool) | True / False | True → retrieved chunks are on-topic |
| **BLEU Score** | NLP (float) | 0.0 – 1.0 | > 0.3 is solid for open-domain QA |
| **ROUGE-L Score** | NLP (float) | 0.0 – 1.0 | > 0.4 indicates strong content overlap |
| **Perplexity** | NLP (float) | 10 – 200+ | < 50 indicates fluent, natural text |

> **Note:** Actual numeric values for each question are logged as a pandas DataFrame via LangSmith and will vary by run. View full results with:
> ```python
> print(df[safe_columns])
> ```

### Overall Assessment

The RAG pipeline is expected to perform strongly on **Relevance**, **Groundedness**, and **Retrieval Relevance** — the three metrics that do not require ground truth — because:

1. The source documents are high-quality, authoritative blog posts that directly cover the topics being queried.
2. The LLM (GPT-4o-mini) is instructed to limit answers to what's in the retrieved context.
3. The retriever uses semantic similarity with `k=4`, which typically returns highly relevant chunks for focused technical questions.

**Correctness** may vary depending on whether GPT-4o-mini's phrasing aligns closely enough with the manually curated ground truth answers.

**BLEU** will typically be lower (0.05–0.2) because RAG-generated answers are paraphrased rather than word-for-word reproductions. **ROUGE-L** will be moderately higher (0.2–0.5). **Perplexity** should be low (20–60) since GPT-4o-mini generates fluent text.

---

## Dataset Used

A custom LangSmith evaluation dataset named **"Production RAG Evaluation Dataset"** was created programmatically:

```python
dataset = client.create_dataset(dataset_name="Production RAG Evaluation Dataset")
client.create_examples(dataset_id=dataset.id, examples=examples)
```

Each example follows the schema:
```json
{
  "inputs": { "question": "..." },
  "outputs": { "answer": "..." }
}
```

---

## Environment Setup

### Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/)
- A [DataStax Astra DB](https://astra.datastax.com/) account (free tier works)
- A [LangSmith](https://smith.langchain.com/) account (free tier works)

### Installation

```bash
pip install astrapy
pip install langchain langchain-openai langchain-community cassio datasets evaluate nltk rouge-score langsmith
pip install langchain-astradb
pip install transformers torch
```

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=ls__...
ASTRA_DB_API_ENDPOINT=https://<your-db-id>.apps.astra.datastax.com
ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
ASTRA_DB_KEYSPACE=default_keyspace
```

---

## How to Run

1. Clone the repository and install dependencies (see above).
2. Fill in your `.env` file with valid credentials.
3. Open `RAG_evaluation.ipynb` in Jupyter.
4. Run all cells in order — the notebook is self-contained and sequential.
5. After evaluation completes, view results in the printed DataFrame or navigate to your LangSmith dashboard to see detailed per-question scores and traces.

---

## Project Structure

```
.
├── RAG_evaluation.ipynb      # Main notebook — full pipeline + evaluation
├── .env                      # API keys (not committed to version control)
└── README.md                 # This file
```

---

## Key Design Decisions

**Why Astra DB?** Astra DB is a serverless vector database built on Apache Cassandra. It offers a free tier, no infrastructure management, and a native LangChain integration — ideal for prototyping production RAG systems.

**Why chunk size 250 tokens?** Smaller chunks improve retrieval precision (each chunk is more topically focused). The 25-token overlap ensures continuity across chunk boundaries.

**Why k=4 for retrieval?** Retrieving 4 chunks provides enough context for the LLM to synthesize a good answer without overloading the context window or diluting relevance.

**Why LLM-as-judge instead of only string metrics?** String metrics like BLEU can't capture semantic correctness — a paraphrased but factually perfect answer scores low. LLM judges evaluate meaning, not surface form. Using both paradigms gives a complete picture.

**Why GPT-2 for perplexity?** GPT-2 is lightweight, open-source, and reproducible. It provides a consistent fluency baseline without requiring API calls. Note that perplexity is model-relative — lower values only mean "less surprising to GPT-2", not absolutely better.

**Why LangSmith?** LangSmith provides end-to-end observability — tracing every retrieval + LLM call, storing benchmark datasets, and aggregating evaluator scores into a structured experiment view, all with zero boilerplate.
