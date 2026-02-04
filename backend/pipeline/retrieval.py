import os
import json
import hashlib
import numpy as np
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
EMBEDDING_MODEL = "intfloat/e5-small-v2"
CHUNK_FILE = "data/processed_chunks.json"
EMBED_FILE = "data/processed_embeddings.npy"
META_FILE = "data/embedding_meta.json"

model = SentenceTransformer(EMBEDDING_MODEL)

RESULT_SECTION_TERMS = [
    "results", "experiments", "evaluation", "findings",
    "analysis", "outcomes", "performance", "comparison"
]

OUTCOME_LANGUAGE_TERMS = [
    "significant", "no significant difference",
    "outperformed", "performed better", "higher than", "lower than",
    "improved", "declined", "equivalent", "similar performance",
    "effect size", "confidence interval", "p <", "p ="
]

NON_EVIDENCE_TERMS = [
    "introduction", "background", "related work",
    "survey", "overview", "methodology"
]

def _evidence_likelihood(text: str) -> float:
    """
    Estimate how likely a chunk contains evaluated outcomes.
    Pure heuristic, domain-agnostic.
    """
    t = text.lower()
    score = 0.0

    for k in RESULT_SECTION_TERMS:
        if k in t:
            score += 0.15

    for k in OUTCOME_LANGUAGE_TERMS:
        if k in t:
            score += 0.10

    for k in NON_EVIDENCE_TERMS:
        if k in t:
            score -= 0.10

    return score


def build_query_text(structured_query: dict) -> str:
    """
    Query remains semantic, but nudged toward outcomes.
    """
    return (
        "query: results outcomes effect size evaluation impact "
        f"{structured_query.get('model_a', '')} "
        f"{structured_query.get('task', '')} "
        f"{structured_query.get('model_b', '')} "
        f"{structured_query.get('metric', '')} "
        f"{structured_query.get('dataset', '')}"
    ).strip()


def _load_chunks_from_disk():
    with open(CHUNK_FILE, encoding="utf-8") as f:
        return json.load(f)

def _chunks_signature(chunks):
    hasher = hashlib.sha256()
    hasher.update(EMBEDDING_MODEL.encode("utf-8"))
    for c in chunks:
        hasher.update(c["paper_id"].encode("utf-8"))
        hasher.update(c["text"][:200].encode("utf-8"))
    return hasher.hexdigest()


def _compute_and_cache_embeddings(chunks):
    texts = [f"passage: {c['text']}" for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    np.save(EMBED_FILE, embeddings)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump({"signature": _chunks_signature(chunks)}, f)

    return embeddings


def _load_or_create_embeddings(chunks):
    signature = _chunks_signature(chunks)

    if os.path.exists(EMBED_FILE) and os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("signature") == signature:
                return np.load(EMBED_FILE)
        except Exception:
            pass

    return _compute_and_cache_embeddings(chunks)


def retrieve_top_k_per_paper(
    structured_query: dict,
    k: int = 3,
    chunks: list | None = None
):
    """
    Retrieve top-k evidence-biased chunks PER paper.
    """

    if structured_query is None:
        raise ValueError("structured_query is required")

    if chunks is None:
        chunks = _load_chunks_from_disk()

    if not chunks:
        return {}

    embeddings = _load_or_create_embeddings(chunks)

    query_text = build_query_text(structured_query)
    query_embedding = model.encode([query_text], normalize_embeddings=True)

    semantic_scores = cosine_similarity(query_embedding, embeddings)[0]

    # --- combine semantic similarity + evidence likelihood ---
    combined_scores = []
    for idx, chunk in enumerate(chunks):
        evidence_boost = _evidence_likelihood(chunk.get("text", ""))
        combined_scores.append(semantic_scores[idx] + evidence_boost)

    paper_groups = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        paper_groups[chunk["paper_id"]].append((idx, chunk))

    results = {}
    for paper_id, items in paper_groups.items():
        indices = [i for i, _ in items]
        scores = [combined_scores[i] for i in indices]
        top_pos = np.argsort(scores)[::-1][:k]
        results[paper_id] = [items[p][1] for p in top_pos]

    return results
