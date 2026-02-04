Evidence-Aware Research Comparison RAG
Overview

This project is a Retrieval-Augmented Generation (RAG) system designed to answer comparative research questions such as:

“Do transformers outperform LSTMs on machine translation tasks?”

Unlike typical RAG systems that generate confident answers from loosely related text, this system is evidence-aware and conservative by design.
It only produces conclusions when they are justified by the papers provided.

The Problem

Most research QA systems:

retrieve chunks from papers,

summarize them,

and produce an answer even when evidence is weak or indirect.

This leads to:

overconfident conclusions,

hallucinated claims,

and loss of trust.

In real research reading, humans:

look for explicit claims, and

if none exist, infer from evaluation metrics (tables, scores),

otherwise, admit insufficient evidence.

This project replicates that behavior explicitly and transparently.

Core Idea

Answer comparative research questions by extracting explicit claims first, and only inferring from metrics when necessary — while clearly labeling inference.

Pipeline Architecture
User Question
   ↓
Query Parsing
   ↓
Document Retrieval
   ↓
Claim Extraction
   ↓
Claim Validation
   ↓
(If no explicit claims)
Metric-Based Inference
   ↓
Claim Comparison
   ↓
Final Conclusion

Key Features
1. Structured Query Understanding

The system parses the user question into:

Model A

Model B

Task

This prevents irrelevant evidence from influencing the result.

2. Evidence-Focused Retrieval

Only the most relevant chunks from each paper are retrieved.
The system does not summarize entire PDFs.

3. Explicit Claim Extraction

The system first looks for clear textual claims such as:

“Model A outperforms Model B on task X”

If no such claims exist, it does not guess.

4. Claim Validation (Honesty Layer)

Each extracted claim is validated against:

the original question

the structured query

Irrelevant or tangential claims are discarded.

5. Metric-Based Inference (Optional Fallback)

If no explicit claims are found, the system:

analyzes reported metrics (e.g., BLEU scores),

checks task, dataset, and metric consistency,

infers a conclusion only when justified.

All such conclusions are explicitly labeled as:

claim_type: inferred_from_tables


This ensures the system never pretends an inference was a textual claim.

6. Conservative Final Output

The system outputs one of:

AGREE

MIXED

INSUFFICIENT_EVIDENCE

Each result is backed by transparent reasoning and evidence.

Example Output

For the question:

Do transformers outperform LSTMs on machine translation tasks?

The system may output:

COMPARISON RESULT: AGREE

Basis: inferred_from_tables
Evidence: Transformer BLEU scores are consistently higher than LSTM-based models on WMT14 datasets.


If no sufficient evidence exists, it will clearly state that instead.

Why This Project Is Different

❌ Does not hallucinate answers

❌ Does not over-generalize from weak evidence

❌ Does not mix inference with extraction silently

✅ Separates extraction, validation, inference, and comparison

✅ Labels inferred conclusions explicitly

✅ Matches how real researchers read papers

Intended Use Cases

Research literature comparison

Evidence-based model benchmarking

Academic assistance tools

Trustworthy RAG system demonstrations

Limitations

Does not infer when datasets or metrics are incomparable

Requires pre-processed document chunks

Optimized for comparative questions, not general summarization

These limitations are intentional to preserve correctness.

Conclusion

This project demonstrates that RAG systems do not need to be overconfident to be useful.
By enforcing evidence discipline and transparency, it produces conclusions that can be trusted — or honestly withheld.