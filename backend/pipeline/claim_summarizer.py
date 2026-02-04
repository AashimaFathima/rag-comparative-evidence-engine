import json
import os
from typing import List, Dict

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"

client = Groq(api_key=GROQ_API_KEY)

PROMPT_PATH = "prompts/summarize_claim.txt"


def load_prompt() -> str:
    with open(PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def summarize_claims(claims: List[Dict]) -> List[Dict]:
    """
    Convert extracted evidence into concise, paper-faithful claims.
    """

    prompt_template = load_prompt()
    summarized = []

    for claim_obj in claims:
        evidence = claim_obj.get("evidence", "").strip()

        if not evidence:
            summarized.append(claim_obj)
            continue

        prompt = prompt_template.replace("{{EVIDENCE}}", evidence)

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=120
        )

        raw = completion.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw)
            summary = parsed.get("summary", "").strip()
        except Exception:
            summary = ""

        summarized.append({
            **claim_obj,
            "claim": summary if summary else claim_obj.get("claim", "")
        })

    return summarized
