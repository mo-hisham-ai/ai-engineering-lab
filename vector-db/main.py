import os

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = "gemini-embedding-001"

db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection(name="handbook")


def embed(texts: list[str]) -> list[list[float]]:
    result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return [e.values for e in result.embeddings]


def chunk_by_paragraph(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def index_document(path: str) -> None:
    if collection.count() > 0:
        print(f"Already indexed: {collection.count()} chunks (no embedding needed)")
        return
    with open(path, encoding="utf-8") as f:
        chunks = chunk_by_paragraph(f.read())
    collection.add(
        ids=[f"{path}-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
        metadatas=[{"source": path, "chunk": i} for i in range(len(chunks))],
    )
    print(f"Indexed {len(chunks)} chunks")


def search(question: str, top_k: int = 3):
    results = collection.query(query_embeddings=embed([question]), n_results=top_k)
    return list(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    )


if __name__ == "__main__":
    index_document("doc.txt")

    while True:
        question = input("\nQuestion (or 'exit'): ")
        if question.lower() == "exit":
            break
        for doc, meta, dist in search(question):
            print(f"\n[distance {dist:.3f}] (chunk {meta['chunk']}) {doc[:150]}...")