import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"

client = Groq(api_key=GROQ_API_KEY)


def safe_json_load(text: str):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end])
    except Exception:
        return None

def validate_claim(
    question: str,
    structured_query: dict,
    claim: dict,
    prompt_template: str
):
    prompt = (
        prompt_template
        .replace("{{QUESTION}}", question)
        .replace("{{STRUCTURED_QUERY}}", json.dumps(structured_query))
        .replace("{{CLAIM}}", claim.get("claim", ""))
        .replace("{{EVIDENCE}}", claim.get("evidence", ""))
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=120
        )
    except Exception as e:
            return {
                "is_valid": True,
                "reason": f"Validator LLM error; default keep ({str(e)})"
            }


    raw = completion.choices[0].message.content.strip()
    parsed = safe_json_load(raw)

    if not parsed:
        return {"is_valid": True, "reason": "Malformed output; default keep"}

    if "is_valid" not in parsed:
        return {"is_valid": True, "reason": "Missing is_valid; default keep"}

    return parsed  

