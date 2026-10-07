import math
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = "gemini-embedding-001"


def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into chunks of `size` characters, overlapping by `overlap`."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += size - overlap
    return [c for c in chunks if c]


def embed(texts: list[str]) -> list[list[float]]:
    result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return [e.values for e in result.embeddings]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def search(question: str, chunks: list[str], vectors: list[list[float]], top_k: int = 3):
    question_vector = embed([question])[0]
    scored = [
        (cosine_similarity(question_vector, v), chunk)
        for chunk, v in zip(chunks, vectors)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    with open("doc.txt", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    print(f"Document split into {len(chunks)} chunks")
    vectors = embed(chunks)

    while True:
        question = input("\nQuestion (or 'exit'): ")
        if question.lower() == "exit":
            break
        for score, chunk in search(question, chunks, vectors):
            print(f"\n[{score:.3f}] {chunk}")