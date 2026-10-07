import math
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = "gemini-embedding-001"

texts = [
    "كيف اشتراك في حسابي",
    "ماهي اشهر 5 مدن",
    "ماهي محتوايات بيتزا",
]

result = client.models.embed_content(model=EMBED_MODEL, contents=texts)

for text, emb in zip(texts, result.embeddings):
    print(f"{text!r} -> {len(emb.values)} numbers, first 3: {emb.values[:3]}")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


vectors = [e.values for e in result.embeddings]
print("password vs credentials:", cosine_similarity(vectors[0], vectors[1]))
print("password vs pizza:", cosine_similarity(vectors[0], vectors[2]))