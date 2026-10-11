import time

from google.genai import types

from main import client, index_document, search

MODEL = "gemini-3.5-flash-lite"

SYSTEM_PROMPT = (
    "You answer questions using ONLY the context provided. "
    "If the answer is not in the context, reply exactly: "
    "I don't know based on the document. "
    "Never use outside knowledge. Keep the answer short."
)


def ask_llm(prompt: str, retries: int = 4) -> str:
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            return response.text
        except Exception as e:
            if "503" in str(e) and attempt < retries:
                wait = 2 ** attempt
                print(f"Server busy, retrying in {wait}s... ({attempt}/{retries})")
                time.sleep(wait)
            else:
                raise


def answer(question: str, top_k: int = 3):
    top = search(question, top_k)
    context = "\n\n".join(f"[{i}] {doc}" for i, (doc, _, _) in enumerate(top, 1))
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return ask_llm(prompt), top


if __name__ == "__main__":
    index_document("doc.txt")

    while True:
        question = input("\nQuestion (or 'exit'): ")
        if question.lower() == "exit":
            break
        text, top = answer(question)
        print(f"\nAnswer: {text}")
        print("\nSources:")
        for doc, meta, dist in top:
            print(f"  [chunk {meta['chunk']}, distance {dist:.3f}] {doc[:80]}...")