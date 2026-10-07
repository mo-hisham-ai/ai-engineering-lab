import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Literal
from pydantic import BaseModel
from typing import Literal
from pydantic import BaseModel, ValidationError

load_dotenv()

MODEL = "gemini-3.5-flash-lite"

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)
class Answer(BaseModel):
    answer: str
    difficulty: Literal["beginner", "intermediate", "advanced"]
    key_points: list[str]

SYSTEM_PROMPT = (
    "You are a helpful teacher. Answer the user's question and return JSON "
    "with exactly these fields: answer (string), difficulty "
    "(beginner/intermediate/advanced), key_points (list of 3 short strings)."
)

class Answer(BaseModel):
    answer: str
    difficulty: Literal["beginner", "intermediate", "advanced"]
    key_points: list[str]


def ask(question: str, retries: int = 4) -> Answer:
    validation_failed = False
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
            print(f"Tokens: {response.usage_metadata.total_token_count}")
            return Answer.model_validate_json(response.text)
        except ValidationError:
            if validation_failed:
                raise
            validation_failed = True
            print("Invalid JSON shape, retrying once...")
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
        print(result.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error: {e}")
        