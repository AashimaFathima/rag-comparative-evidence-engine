import json
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"

client = Groq(api_key=GROQ_API_KEY)

def parse_query(question: str, prompt_template: str) -> dict:
    prompt = prompt_template.replace("{{USER_QUESTION}}", question)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )

    text = completion.choices[0].message.content.strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    try:
        parsed = json.loads(text[start:end])
        return parsed

    except Exception as e:
        return {
            "model_a": None,
            "model_b": None,
            "task": None,
            "metric": None,
            "dataset": None,
            "scope": None,
            "error": "Invalid JSON returned by model"
        }
