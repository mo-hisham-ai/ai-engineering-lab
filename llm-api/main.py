import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-3.5-flash-lite"

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = (
    "You are a helpful teacher. Answer the user's question and return JSON "
    "with exactly these fields: answer (string), difficulty "
    "(beginner/intermediate/advanced), key_points (list of 3 short strings)."
)


def ask(question: str, retries: int = 4) -> dict:
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            if "503" in str(e) and attempt < retries:
                wait = 2 ** attempt
                print(f"Server busy, retrying in {wait}s... ({attempt}/{retries})")
                time.sleep(wait)
            else:
                raise


if __name__ == "__main__":
    question = input("Ask something: ")
    try:
        result = ask(question)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")