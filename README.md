# 📚 RAG Comparative Evidence Engine

A **Retrieval-Augmented Generation (RAG)** system for **extracting, validating, and ranking comparative evidence from research papers** in response to analytical research questions.

This project focuses on **evidence-grounded reasoning**, avoiding hallucinations by grounding all outputs in retrieved academic text.

---

## 🚀 What This Project Does

Given a **comparative research question** (for example, *“Does online learning improve academic performance compared to face-to-face learning?”*), the system:

1. Parses the question into a structured comparison  
2. Retrieves evidence-rich chunks from uploaded research papers  
3. Extracts explicit claims from each paper  
4. Filters out non-result or descriptive claims  
5. Validates claims against the provided evidence  
6. Summarizes and ranks claims by relevance  
7. Returns paper-wise ranked comparative evidence  

All outputs remain **traceable to source text**.

---

## 🧠 Motivation

Large language models often:
- Hallucinate facts  
- Mix evidence across sources  
- Produce ungrounded summaries  

This system separates **retrieval, extraction, validation, and ranking** to ensure that responses remain evidence-based and inspectable.

---

## 🏗️ System Architecture

```text

User Question
↓
Query Parser (LLM)
↓
Evidence-Biased Retrieval (Sentence Transformers)
↓
Claim Extraction (LLM)
↓
Result-Only Filtering (Rule-based)
↓
Claim Validation (LLM)
↓
Claim Summarization (LLM)
↓
Claim Ranking (LLM)

```

## 📁 Project Structure


```text

RAG_PROJECT/
│
├── backend/
│   ├── pipeline/
│   │   ├── query_parser.py
│   │   ├── retrieval.py
│   │   ├── claim_extraction.py
│   │   ├── claim_validation.py
│   │   ├── claim_summarizer.py
│   │   └── claim_ranker.py
│   │
│   ├── prompts/
│   │   ├── parse_query.txt
│   │   ├── extract_claims.txt
│   │   ├── validate_claim.txt
│   │   ├── summarize_claim.txt
│   │   └── rank_claims.txt
│   │
│   ├── scripts/
│   │   └── ingest_pdf.py
│   │
│   ├── data/
│   │   ├── papers/
│   │   ├── processed_chunks.json
│   │   ├── processed_embeddings.npy
│   │   └── embedding_meta.json
│   │
│   ├── server.py
│   └── requirements.txt
│
├── frontend/
│   ├── app/
│   ├── public/
│   ├── package.json
│   └── package-lock.json
│
└── README.md

```
---

## 🧪 Evaluation (Overview)

The system has been evaluated using manually curated comparative questions and expected claim sets to verify:

- Claim relevance  
- Evidence faithfulness  
- Comparative correctness  

---

## 🧠 Embedding Model & Performance Notes

This system uses the **`intfloat/e5-small-v2`** sentence embedding model for retrieval.

### Why this model was chosen
- Designed specifically for **information retrieval**
- Strong performance on semantic similarity tasks

### Design Tradeoff
- This project intentionally prioritizes **retrieval accuracy and evidence faithfulness** over raw speed.
- Latency is driven by a multi-stage, evidence-grounded design involving multiple LLM calls per query to ensure domain-agnostic claim extraction, validation, and ranking.

## 🔒 Environment Variables

The backend requires the following environment variable:

```env
GROQ_API_KEY=your_api_key_here

```

## ▶️ Running Locally

Backend 
```text
cd backend 
pip install -r requirements.txt 
python server.py
```

Frontend 
```text
cd frontend 
npm install 
npm run dev
```

## 📜 License

This project is for educational and research purposes.

