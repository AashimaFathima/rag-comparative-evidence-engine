import os
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langserve import add_routes
from langchain_core.runnables import RunnableLambda

# --- PIPELINE IMPORTS ---
from pipeline.query_parser import parse_query
from pipeline.retrieval import retrieve_top_k_per_paper
from pipeline.claim_extraction import extract_claims_per_paper
from pipeline.claim_validation import validate_claim
from pipeline.claim_summarizer import summarize_claims
from pipeline.claim_ranker import rank_claims

app = FastAPI(title="Comparative Research Evidence Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- INPUT SCHEMA ----------------
class ResearchInput(BaseModel):
    question: str


# ---------------- UTILS ----------------
def load_prompts():
    with open("prompts/parse_query.txt", encoding="utf-8") as f:
        parse_p = f.read()
    with open("prompts/extract_claims.txt", encoding="utf-8") as f:
        extract_p = f.read()
    with open("prompts/validate_claim.txt", encoding="utf-8") as f:
        validate_p = f.read()
    with open("prompts/rank_claims.txt", encoding="utf-8") as f:
        rank_p = f.read()

    return parse_p, extract_p, validate_p, rank_p


def compute_graph_stats(claims_dict: dict) -> dict:
    real_ids = [pid for pid in claims_dict.keys() if pid != "TABLE_INFERRED"]
    table_claims = claims_dict.get("TABLE_INFERRED", {}).get("claims", [])
    table_count = len(table_claims) if table_claims else 0

    bars = []
    max_c = 0

    for i, pid in enumerate(real_ids):
        claims_list = claims_dict[pid].get("claims", [])
        count = len(claims_list) if claims_list else 0

        if i == 2:
            count += table_count

        max_c = max(max_c, count)
        bars.append({"id": pid, "count": count})

    return {"data": bars, "y_max": max_c + 1}


# ---------------- PIPELINE ----------------
def run_pipeline(payload) -> dict:
    if isinstance(payload, BaseModel):
        payload = payload.dict()

    question = payload.get("question")
    if not question:
        return {"stage": "error", "claims": {}, "graph_stats": {"data": [], "y_max": 0}}

    parse_p, extract_p, validate_p, rank_p = load_prompts()

    # 1. Parse
    structured_query = parse_query(question, parse_p)
    if structured_query.get("error"):
        return {"stage": "error", "claims": {}, "graph_stats": {"data": [], "y_max": 0}}

    # 2. Retrieve
    retrieved = retrieve_top_k_per_paper(structured_query=structured_query, k=6)
    
    # 3. Extract
    extracted_claims = extract_claims_per_paper(
        retrieved_chunks=retrieved,
        structured_query=structured_query,
        prompt_template=extract_p,
        question=question
    )
    
    # 4. Validate + rank
    validated_claims = {}

    for paper_id, data in extracted_claims.items():
        raw_claims = data.get("claims", [])
        valid = []

        for claim in raw_claims:
            verdict = validate_claim(
                question=question,
                structured_query=structured_query,
                claim=claim,
                prompt_template=validate_p
            )

            if isinstance(verdict, dict) and verdict.get("is_valid") is True:
                valid.append(claim)
            elif verdict is True or verdict is None:
                valid.append(claim)

        if valid:
            summarized = summarize_claims(valid)
            ranked = rank_claims(question=question, claims=summarized, prompt_template=rank_p)
            validated_claims[paper_id] = {"claims": ranked[:3]}
        else:
            validated_claims[paper_id] = {"claims": []}

    # 5. Result
    graph_stats = compute_graph_stats(validated_claims)

    return {
        "stage": "Synthesizing results…",
        "claims": validated_claims,
        "graph_stats": graph_stats
    }


# ---------------- LANGSERVE ----------------
rag_chain = RunnableLambda(run_pipeline).with_types(input_type=ResearchInput)
add_routes(app, rag_chain, path="/analyze")


# ---------------- UPLOAD ENDPOINT ----------------
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    if os.path.exists("data/processed_embeddings.npy"):
        os.remove("data/processed_embeddings.npy")
    if os.path.exists("data/embedding_meta.json"):
        os.remove("data/embedding_meta.json")

    if os.path.exists("data/papers"):
        shutil.rmtree("data/papers")
    os.makedirs("data/papers", exist_ok=True)

    for file in files:
        path = os.path.join("data/papers", file.filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    from scripts.ingest_pdf import ingest_pdfs
    ingest_pdfs()
    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
