from google.genai import types

from main import chunk_text, client, embed, search

MODEL = "gemini-3.5-flash-lite"

SYSTEM_PROMPT = (
    "You answer questions using ONLY the context provided. "
    "If the answer is not in the context, reply exactly: "
    "I don't know based on the document. "
    "Never use outside knowledge. Keep the answer short."
)


def answer(question: str, chunks: list[str], vectors: list[list[float]]):
    top = search(question, chunks, vectors, top_k=3)
    context = "\n\n".join(f"[{i}] {chunk}" for i, (_, chunk) in enumerate(top, 1))
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text, top


if __name__ == "__main__":
    with open("doc.txt", encoding="utf-8") as f:
        chunks = chunk_text(f.read())
    vectors = embed(chunks)
    print(f"Ready: {len(chunks)} chunks indexed")

    while True:
        question = input("\nQuestion (or 'exit'): ")
        if question.lower() == "exit":
            break
        text, top = answer(question, chunks, vectors)
        print(f"\nAnswer: {text}")
        print("\nSources:")
        for score, chunk in top:
            print(f"  [{score:.3f}] {chunk[:70]}...")