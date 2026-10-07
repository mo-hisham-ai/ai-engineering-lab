# 04 - RAG Basics: Chunking + Semantic Search

The retrieval half of RAG: split a document into chunks, embed each chunk, and find the chunks closest in meaning to a question. (The generation half, where an LLM answers from the retrieved chunks, comes next.)

## How it works

```
doc.txt -> chunk_text() -> embed() -> vectors (stored in memory)
question -> embed() -> cosine_similarity vs every chunk -> top 3 chunks
```

## What I built
- `chunk_text(text, size, overlap)`: splits text into overlapping character chunks.
- `embed(texts)`: turns texts into vectors with the Gemini embedding model.
- `cosine_similarity(a, b)`: measures how close two vectors are in meaning.
- `search(question, chunks, vectors, top_k)`: returns the closest chunks to a question.

## Results (chunk size 400, overlap 80, 5 chunks)

| Question | Top chunk contained the answer? |
|---|---|
| How many vacation days do I get? | Yes (annual leave, 21 days). The document says "annual leave", not "vacation", so this matched by meaning. |
| Can I work from home? | Yes (remote work, up to two days per week) |
| How much money can I spend on courses? | Yes (learning budget, 1,500 dollars) |

## What I noticed
- Search by meaning works: the questions used different words than the document and the right chunk still ranked first.
- Chunks are cut in the middle of words and sentences (for example "he team lead..."), because I split by character count. Overlap helps but does not fix it.
- Scores are close to each other (0.67 vs 0.67 on the first question), so ranking is useful but the absolute number is not.
- A chunk can contain the end of one topic and the start of another, which adds noise.

## Chunk size experiment
_To fill in after testing `size=150` and `size=1500`:_
- size=150: ...
- size=1500: ...

## Next steps
- Split on sentences or paragraphs instead of raw characters.
- Send the top chunks to the LLM so it answers from them (full RAG).

## Run
1. Install dependencies: `py -m pip install google-genai python-dotenv`
2. Create a `.env` file with `GEMINI_API_KEY=your_key`
3. Run: `py main.py` and ask questions about `doc.txt`