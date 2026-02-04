import json
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
client = Groq(api_key=GROQ_API_KEY)


def rank_claims(
    question: str,
    claims: list,
    prompt_template: str
) -> list:
    """
    Rank claims by relevance to the question.
    Guarantees LLM-extracted claims ('explicit') are ranked ABOVE fallback claims
    """

    if len(claims) <= 1:
        return claims

    # --- 1. Separate by provenance ---
    explicit_claims = [c for c in claims if c.get("source") == "explicit"]
    fallback_claims = [c for c in claims if c.get("source") != "explicit"]

    def _rank_with_llm(claim_group):
        if len(claim_group) <= 1:
            return claim_group

        claims_block = []
        for i, c in enumerate(claim_group):
            claims_block.append(f"{i+1}. {c.get('claim')}")

        prompt = (
            prompt_template
            .replace("{{QUESTION}}", question)
            .replace("{{CLAIMS}}", "\n".join(claims_block))
        )

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=120
        )

        raw = completion.choices[0].message.content.strip()

        s = raw.find("{")
        e = raw.rfind("}") + 1
        if s == -1 or e == -1:
            return claim_group  

        try:
            parsed = json.loads(raw[s:e])
            order = parsed.get("ranking", [])
            return [claim_group[i-1] for i in order if 1 <= i <= len(claim_group)]
        except Exception:
            return claim_group

    # --- 2. Rank each group independently ---
    ranked_explicit = _rank_with_llm(explicit_claims)
    ranked_fallback = _rank_with_llm(fallback_claims)

    # --- 3. Combine (explicit ALWAYS first) ---
    return ranked_explicit + ranked_fallback

