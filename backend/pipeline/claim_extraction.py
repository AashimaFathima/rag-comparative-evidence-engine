import json
import os
from dotenv import load_dotenv
from groq import Groq
import re
import time

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
client = Groq(api_key=GROQ_API_KEY)

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def safe_json_load(text: str):
    """
    Safely extract and parse JSON from LLM output.
    Returns parsed JSON or None.
    """
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end])
    except Exception:
        return None


def _heuristic_extract_from_text(text: str, max_claims: int = 5):
    """
    Conservative regex-based fallback to find candidate claim sentences.
    """
    keywords = [
        "significant", "no significant", "increased", "decreased", "improved",
        "worse", "better", "compared to", "compared with", "equivalent",
        "similar", "performance", "outcome", "grade", "score", "p <", "p =",
        "odds ratio", "effect size", "confidence interval"
    ]
    sents = _SENT_SPLIT_RE.split(text)
    candidates = []
    for s in sents:
        lower = s.lower()
        if any(k in lower for k in keywords):
            sent = s.strip()
            if len(sent) > 20:
                candidates.append({
                    "claim": re.sub(r'\s+', ' ', sent)[:600],
                    "evidence": sent[:1200]
                })
        if len(candidates) >= max_claims:
            break
    return candidates


def extract_claims_per_paper(
    retrieved_chunks: dict,
    structured_query: dict,
    prompt_template: str,
    question: str
) -> dict:
    """
    Extract EXPLICIT claims from each paper.
    """

    results = {}

    def _safe_str(x):
        return x if isinstance(x, str) else ""

    model_a = _safe_str(structured_query.get("model_a")) or _safe_str(structured_query.get("entity_a"))
    model_b = _safe_str(structured_query.get("model_b")) or _safe_str(structured_query.get("entity_b"))
    task = _safe_str(structured_query.get("task"))


    for paper_id, chunks in retrieved_chunks.items():
        start_time = time.time()

        if not chunks:
            results[paper_id] = {"claims": []}
            continue

        # ---- CONTEXT SELECTION ----
        chosen = []
        for c in chunks[:3]:
            if c not in chosen:
                chosen.append(c)
        for c in chunks[3:6]:
            if c not in chosen:
                chosen.append(c)
        for c in chunks[-3:]:
            if c not in chosen:
                chosen.append(c)

        combined_text = "\n\n".join(c["text"][:2000] for c in chosen)

        prompt = (
            prompt_template
            .replace("{{MODEL_A}}", model_a)
            .replace("{{MODEL_B}}", model_b)
            .replace("{{TASK}}", task)
            .replace("{{RETRIEVED_CHUNKS}}", combined_text)
        )

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=600
            )
        except Exception as e:
            print(f"[EXTRACT][ERROR] LLM call failed for {paper_id}: {e}")
            results[paper_id] = {"claims": []}
            continue

        raw = completion.choices[0].message.content.strip()

        parsed = safe_json_load(raw)

        # ---- PRIMARY PATH: LLM-extracted claims ----
        if parsed and "claims" in parsed:
            extracted_claims = []
            for c in parsed.get("claims", []):
                extracted_claims.append({
                    "claim": (c.get("claim") or c.get("text") or "").strip(),
                    "evidence": (c.get("evidence") or c.get("evidence_text") or "").strip(),
                    "source": "explicit"  # LLM-extracted
                })

            results[paper_id] = {"claims": extracted_claims}

            elapsed = time.time() - start_time
            print(f"[EXTRACT] Parsed {len(extracted_claims)} claims for {paper_id} (took {elapsed:.2f}s)")
            continue


        # ---- FALLBACK PATH: Heuristic extraction ----
        print(f"[EXTRACT] JSON parse failed for {paper_id} — attempting fallback extraction")

        fallback_claims = _heuristic_extract_from_text(combined_text)

        results[paper_id] = {
            "claims": [
                {
                    "claim": c.get("claim", "").strip(),
                    "evidence": c.get("evidence", "").strip(),
                    "source": "explicit_fallback"  # heuristic fallback
                }
                for c in fallback_claims
            ]
        }
        continue

    return results
