import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-3.5-flash-lite"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    print(f"[tool called] add({a}, {b})")
    return a + b


response = client.models.generate_content(
    model=MODEL,
    contents="What is 123.5 + 877.25?",
    config=types.GenerateContentConfig(tools=[add]),
)
print(response.text)