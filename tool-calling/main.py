import os
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-3.5-flash-lite"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def add(a: float, b: float) -> float:
    print(f"[tool called] add({a}, {b})")
    return a + b


def get_current_time() -> str:
    """Return the current date and time."""
    print("[tool called] get_current_time()")
    return datetime.now().isoformat()


question = input("Ask something: ")

response = client.models.generate_content(
    model=MODEL,
    contents=question,
    config=types.GenerateContentConfig(tools=[add, get_current_time]),
)
print(response.text)