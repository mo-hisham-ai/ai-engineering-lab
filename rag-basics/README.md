# 04 - RAG Basics: Chunking, Semantic Search and Full RAG

A small RAG (Retrieval-Augmented Generation) system built from scratch, with no frameworks. It answers questions about a text document using only what the document says, and shows the sources it used.

## How it works

```
Preparation (once per run):
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

## Part 1: Retrieval (chunk size 400, overlap 80, 5 chunks)

| Question | Top chunk contained the answer? |
|---|---|
| How many vacation days do I get? | Yes (annual leave, 21 days). The document says "annual leave", not "vacation", so this matched by meaning. |
| Can I work from home? | Yes (remote work, up to two days per week) |
| How much money can I spend on courses? | Yes (learning budget, 1,500 dollars) |

What I noticed:
- Search by meaning works: the questions used different words than the document and the right chunk still ranked first.
- Chunks are cut in the middle of words and sentences, because I split by character count. Overlap helps but does not fix it.
- Scores are close to each other (0.675 vs 0.674 on the first question), so the ranking is useful but the absolute number is not.

## Part 2: Full RAG

Retrieval alone returns raw text snippets. Full RAG gives those snippets to an LLM so it can write a direct answer. The system prompt forces it to use only the provided context, and to say it does not know when the answer is missing.

| Question | Answer |
|---|---|
| How many vacation days do I get? | 21 days of paid annual leave per year (correct) |
| Do I need a medical certificate for sick leave? | Yes, if the leave lasts more than two consecutive days (correct) |
| What is the CEO's name? (not in the document) | I don't know based on the document (correct refusal) |
| What is the capital of France? (not in the document) | I don't know based on the document (correct refusal) |

### Experiment: removing the restriction
I replaced the strict system prompt with a generic `You are a helpful assistant.` and asked the same out-of-document questions.

| Question | Strict prompt | Generic prompt |
|---|---|---|
| What is the capital of France? | I don't know based on the document | The capital of France is Paris. |
| What is the CEO's name? | I don't know based on the document | No mention in the context (the model has no outside knowledge of this company) |

Without the restriction the model answers from its own training data instead of the document. The prompt is what keeps answers grounded and traceable to the source.

## Part 3: Chunk size experiment

| Setting | Chunks | Sick leave question | Vacation days question |
|---|---|---|---|
| size=400, overlap=80 | 5 | Correct | Correct (21 days) |
| size=150, overlap=80 | 20 | Correct | **Failed** |
| size=1500 | not tested yet | | |

Why size=150 failed on the vacation question: the sentence containing "21 days" was split across several tiny chunks. The top 3 results were fragments that mentioned leave but not the number 21. The model was honest and said it could not find the total, and listed only the numbers it did see (5 carry-over days, 10 sick days).

Lessons:
- When RAG gives a wrong or incomplete answer, check what retrieval returned before blaming the model.
- Small chunks lose context. Large chunks mix several topics into one vector.
- With size=150 I kept overlap=80, which is more than half of each chunk, so there was a lot of duplicated text. A smaller overlap would have been better.

## Limitations and next steps
- Split on paragraphs or sentences instead of raw characters.
- Store embeddings in a vector database instead of recomputing them on every run.
- Try a higher `top_k` (5 instead of 3).
- Add retry handling for 503 errors from the API.
- Measure answer quality with an evaluation set instead of checking by eye.

## Run
1. Install dependencies: `py -m pip install google-genai python-dotenv`
2. Create a `.env` file with `GEMINI_API_KEY=your_key`
3. Retrieval only: `py main.py`
4. Full RAG: `py rag.py`