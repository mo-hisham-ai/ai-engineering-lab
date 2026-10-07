# 04 - RAG Basics: Chunking, Semantic Search and Full RAG

A small RAG (Retrieval-Augmented Generation) system built from scratch, with no frameworks. It answers questions about a text document using only what the document says, and shows the sources it used.

## How it works

```
Preparation (once):
doc.txt -> chunk_text() -> embed() -> vectors (kept in memory)

Each question:
question -> embed() -> cosine_similarity vs every chunk -> top 3 chunks
         -> prompt = chunks + question -> LLM -> answer + sources
```

## Files
- `main.py`: retrieval. Chunking, embeddings, cosine similarity and search.
- `rag.py`: full RAG. Sends the top chunks plus the question to the LLM and prints the answer with sources.
- `doc.txt`: sample document (a fictional employee handbook).

## What I built
- `chunk_text(text, size, overlap)`: splits text into overlapping character chunks.
- `embed(texts)`: turns texts into vectors with the Gemini embedding model.
- `cosine_similarity(a, b)`: measures how close two vectors are in meaning.
- `search(question, chunks, vectors, top_k)`: returns the closest chunks to a question.
- `answer(question, chunks, vectors)`: builds a prompt from the top chunks and asks the LLM to answer only from them.

## Part 1: Retrieval results (chunk size 400, overlap 80, 5 chunks)

| Question | Top chunk contained the answer? |
|---|---|
| How many vacation days do I get? | Yes (annual leave, 21 days). The document says "annual leave", not "vacation", so this matched by meaning. |
| Can I work from home? | Yes (remote work, up to two days per week) |
| How much money can I spend on courses? | Yes (learning budget, 1,500 dollars) |

## What I noticed in retrieval
- Search by meaning works: the questions used different words than the document and the right chunk still ranked first.
- Chunks are cut in the middle of words and sentences, because I split by character count. Overlap helps but does not fix it.
- Scores are close to each other (0.67 vs 0.67 on the first question), so the ranking is useful but the absolute number is not.
- A chunk can contain the end of one topic and the start of another, which adds noise.

## Part 2: Full RAG

Retrieval alone returns raw text snippets. Full RAG gives those snippets to an LLM so it can write a direct answer, and the system prompt forces it to use only the provided context. If the answer is not there, it must say it does not know. This is what prevents hallucination.

| Question | Result |
|---|---|
| How many vacation days do I get? | _fill in_ |
| Do I need a medical certificate for sick leave? | _fill in_ |
| What is the CEO's name? (not in the document) | _fill in_ |
| What is the capital of France? (not in the document) | _fill in_ |

**Prompt experiment:** I removed the "if the answer is not in the context, say you don't know" rule and asked the two out-of-document questions again: _fill in what happened_. The model does not stay inside the document by itself. The prompt is what restricts it.

## Chunk size experiment

- size=400: works, but chunks are cut mid-sentence


Small chunks lose context. Large chunks mix several topics in one vector.

## Limitations and next steps
- Split on paragraphs or sentences instead of raw characters.
- Store embeddings in a vector database instead of recomputing them on every run.
- Add retry handling for 503 errors from the API.
- Measure answer quality with an evaluation set instead of checking by eye.

## Run
1. Install dependencies: `py -m pip install google-genai python-dotenv`
2. Create a `.env` file with `GEMINI_API_KEY=your_key`
3. Retrieval only: `py main.py`
4. Full RAG: `py rag.py`