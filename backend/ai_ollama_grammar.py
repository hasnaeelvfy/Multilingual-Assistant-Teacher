import json
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class AssistantResult:
    answer: str
    corrected: str
    explanation: str


SYSTEM_PROMPT = """You are an English-speaking AI assistant and English teacher.
You must ALWAYS reply in valid JSON only (no extra text).

Return this JSON object:
{
  "answer": string,        // a helpful, friendly answer in simple English
  "corrected": string,     // corrected version of the user's sentence in natural English
  "explanation": string    // short, simple explanation of the main grammar mistakes
}

Rules:
- Keep "answer" short (1-2 sentences).
- Keep "explanation" simple (1-3 short sentences).
- If the user's English is already correct, set "corrected" to the same text and say "Looks correct." in "explanation".
"""


def _safe_json_extract(text: str) -> dict:
    """
    Ollama usually follows the JSON rule, but this makes the code more robust
    if the model adds extra text by mistake.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def ask_ollama_with_grammar(
    user_text: str,
    model: str = "llama3.1:8b",
    host: str = "http://localhost:11434",
    timeout_s: int = 120,
) -> AssistantResult:
    """
    Calls Ollama's local HTTP API.
    Make sure Ollama is running, and the model is pulled.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "stream": False,
        "format": "json",
        # Keep the model loaded in memory for faster next replies.
        # "10m" means keep it alive for 10 minutes after the last request.
        "keep_alive": "10m",
        "options": {
            "temperature": 0.2,
            # Limit output length so replies come faster.
            "num_predict": 200,
        },
    }

    r = requests.post(f"{host}/api/chat", json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    content = (((data or {}).get("message") or {}).get("content")) or ""
    obj = _safe_json_extract(content)

    answer = str(obj.get("answer") or "").strip()
    corrected = str(obj.get("corrected") or "").strip()
    explanation = str(obj.get("explanation") or "").strip()

    if not answer:
        answer = "Sorry, I could not generate an answer."
    if not corrected:
        corrected = user_text.strip()
    if not explanation:
        explanation = "I could not generate an explanation."

    return AssistantResult(answer=answer, corrected=corrected, explanation=explanation)


if __name__ == "__main__":
    text = input("Say something in English: ").strip()
    res = ask_ollama_with_grammar(text)
    print(json.dumps(res.__dict__, ensure_ascii=False, indent=2))
