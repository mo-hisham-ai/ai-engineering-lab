# 05 - Vector Database (ChromaDB) + Full RAG

Semantic search and question answering over a document, using ChromaDB as a persistent vector database, paragraph-based chunking, automatic re-indexing, and an LLM that answers only from the retrieved context.

## What changed compared to `rag-basics`
| | rag-basics | vector-db |
|---|---|---|
| Chunking | fixed number of characters | one chunk per paragraph |
| Storage | in memory, lost on exit | on disk (`chroma_db/`) |
| Embeddings | recomputed on every run | computed once, reused |
| Search | my own cosine similarity loop | `collection.query()` |
| Score | similarity (higher is closer) | distance (**lower** is closer) |
| Document changes | not applicable | detected with a file hash, then re-indexed |

## How it works

```
Every run:
hash(doc.txt) == stored hash?  yes -> reuse chroma_db/
                               no  -> delete old chunks, split by paragraph,
                                      embed, store, save new hash

Each question:
question -> embed -> ChromaDB top 3 chunks -> prompt (context + question)
         -> LLM -> answer + sources
```

## Files
- `main.py`: indexing (with hash check) and search.
- `rag.py`: full RAG. Retrieves chunks, asks the LLM to answer only from them, prints sources. Includes retry on 503 errors.
- `doc.txt`: sample document (a fictional employee handbook).
- `chroma_db/`: the local database and hash file, created automatically and not committed to Git.

## Problem I found: stale index
After I edited a number in `doc.txt`, the search kept returning the old text until I deleted the `chroma_db/` folder.

**Cause:** the first version of `index_document()` only checked `collection.count() > 0` and returned early. It never looked at `doc.txt` again, so the database kept the old content.

**Fix:** store a SHA-256 hash of the document inside `chroma_db/`. On every run, compare it with the current hash. If they differ, print `Document changed, re-indexing...`, delete the old chunks and index again.

Check after the fix: _fill in: edit doc.txt without deleting chroma_db, run again, and write what you saw._

## Results

Questions asked through `rag.py` (document says 25 days of annual leave):

| Question | Answer | Top chunk (distance) |
|---|---|---|
| How many vacation days do I get? | 25 days of paid annual leave | Annual leave (0.601) |
| How much is the learning budget? | 1,500 dollars | Learning budget (0.382) |
| What is the CEO's name? (not in document) | I don't know based on the document | Equipment (0.920) |
| كم عدد أيام الإجازة السنوية؟ (Arabic) | 25 يومًا. | Annual leave (0.509) |

What I noticed:
- The Arabic question found the English paragraph, and its distance (0.509) was even smaller than for the English question (0.601). The embedding model is multilingual, and the LLM answered in the language of the question.
- When the answer exists, the best distance is clearly small (0.38 to 0.60). When it does not exist (CEO question), all three distances are high and close together (0.92 to 0.94). This could later be used as a rule to say "I don't know" before even calling the LLM, but I need more test questions before trusting a threshold.
- Every source is a complete paragraph, so there are no sentences cut in half like in `rag-basics`.

## What I learned
- A vector database stores embeddings on disk and searches them quickly, so I do not recompute them on every run.
- Splitting by paragraph keeps each idea in one piece.
- Distance and similarity are opposites: in ChromaDB a smaller number means a closer match.
- Persistent storage creates a new responsibility: keeping the index in sync with the source document.
- A system prompt that restricts the model to the context is what makes the answers grounded.

## Limitations and next steps
- Paragraph splitting fails on documents with very long or very short paragraphs.
- Only one document is supported.
- Quality is checked by eye. Next step: an evaluation script with questions and expected answers.

## Run
1. Install dependencies: `py -m pip install google-genai python-dotenv chromadb`
2. Create a `.env` file with `GEMINI_API_KEY=your_key`
3. Search only: `py main.py`
4. Full RAG: `py rag.py`
5. To force re-indexing manually, delete the `chroma_db/` folder.