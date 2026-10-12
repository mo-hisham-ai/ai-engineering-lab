# 05 - Vector Database (ChromaDB), Full RAG, API and Docker

A question-answering system over a document, built step by step without frameworks: ChromaDB as a persistent vector database, paragraph-based chunking, automatic re-indexing, an LLM that answers only from the retrieved context, an evaluation script, a FastAPI service, and a Docker image.

## What changed compared to `rag-basics`
| | rag-basics | vector-db |
|---|---|---|
| Chunking | fixed number of characters | one chunk per paragraph |
| Storage | in memory, lost on exit | on disk (`chroma_db/`) |
| Embeddings | recomputed on every run | computed once, reused |
| Search | my own cosine similarity loop | `collection.query()` |
| Score | similarity (higher is closer) | distance (**lower** is closer) |
| Document changes | not applicable | detected with a file hash, then re-indexed |
| Interface | terminal | terminal, evaluation script, HTTP API, Docker |

## How it works

```
Startup:
hash(doc.txt) == stored hash?  yes -> reuse chroma_db/
                               no  -> delete old chunks, split by paragraph,
                                      embed, store, save new hash

Each question:
question -> embed -> ChromaDB top 3 chunks -> prompt (context + question)
         -> LLM -> answer + sources
```

## Files
- `main.py`: indexing (with hash check) and search.
- `rag.py`: full RAG. Retrieves chunks, asks the LLM to answer only from them, prints sources. Retries on 503 errors.
- `eval.py`: evaluation script (retrieval hit rate, answer correctness, refusals).
- `api.py`: FastAPI service with `GET /health` and `POST /ask`.
- `Dockerfile`, `requirements.txt`, `.dockerignore`: container setup.
- `doc.txt`: sample document (a fictional employee handbook).
- `chroma_db/`: local database and hash file, created automatically and not committed to Git.

## Problem I found: stale index
After I edited a number in `doc.txt`, the search kept returning the old text until I deleted the `chroma_db/` folder.

**Cause:** the first version of `index_document()` only checked `collection.count() > 0` and returned early. It never looked at `doc.txt` again.

**Fix:** store a SHA-256 hash of the document inside `chroma_db/`. On every run, compare it with the current hash. If they differ, print `Document changed, re-indexing...`, delete the old chunks and index again.

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
- When the answer exists, the best distance is small (0.38 to 0.60). When it does not exist, all three distances are high and close together (0.92 to 0.94). This could become a rule to say "I don't know" before calling the LLM, but I need more test questions before trusting a threshold.
- Every source is a complete paragraph, so there are no sentences cut in half like in `rag-basics`.

## Evaluation

`eval.py` runs a fixed list of questions and reports three numbers:
- **Retrieval hit@3:** is the right chunk among the top 3 results? (measures search only)
- **Correct answers:** does the answer contain the expected fact? (measures search and model together)
- **Correct refusals:** does the model refuse when the answer is not in the document?

Keeping them separate shows whether a wrong answer comes from retrieval or from the model.

| Run | Retrieval hit@3 | Correct answers | Correct refusals |
|---|---|---|---|
| First 7 easy questions + 2 out-of-document | 7/7 | 7/7 | 2/2 |
| Added 3 harder questions, run 1 | 10/10 | 8/10 | 2/2 |
| Same questions, run 2 | 10/10 | 9/10 | 2/2 |

What I found:
- The easy set passed 100%, so it was too easy to say anything about quality.
- Failing questions: "Can I carry over 10 unused leave days?" (document: maximum 5) and, in run 1 only, "How many total paid days off do I get, vacation and sick?" (25 + 10).
- Retrieval was correct in every failure. The right chunk reached the model, but the model answered "I don't know" because the question needed a small step (a comparison or a sum). This is a **false refusal**: a strict prompt prevents hallucination but also rejects legitimate questions.
- The same question passed in one run and failed in another, so a single run is not enough to judge the system.
- Keyword matching is a crude check. A correct number inside a wrong explanation would still pass. A later improvement is an LLM judge.

## API (FastAPI)

Start: `py -m uvicorn api:app --reload`, then open `http://127.0.0.1:8000/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | returns `{"status": "ok"}` |
| `POST /ask` | body `{"question": "...", "top_k": 3}`, returns the answer and its sources |

Pydantic validates the input: an empty question is rejected automatically with a 422 error. Output has a fixed shape (`answer` plus a list of `sources` with chunk number, distance and text).

## Docker

Build and run:

```
docker build -t handbook-rag .
docker run --rm -p 8000:8000 --env-file .env handbook-rag
```

Then open `http://127.0.0.1:8000/docs`.

Design choices:
- **The API key is not in the image.** `.env` is listed in `.dockerignore` and passed at run time with `--env-file`. A key baked into an image can be read by anyone who gets the image.
- **`requirements.txt` is copied before the code.** Docker caches each step. Dependencies rarely change and take long to install, so they stay cached when only the code changes.

The image builds successfully. Installing the dependencies takes about 4 minutes the first time (`chromadb` is heavy).

Known limitation: `chroma_db/` is not inside the image and `--rm` deletes the container with its data, so the document is re-indexed and embedded again on every start. A Docker volume would keep the index between runs.

## What I learned
- A vector database stores embeddings on disk, so I do not recompute them on every run.
- Splitting by paragraph keeps each idea in one piece.
- Distance and similarity are opposites: in ChromaDB a smaller number means a closer match.
- Persistent storage creates a new responsibility: keeping the index in sync with the source document.
- When a RAG answer is wrong, check what retrieval returned before blaming the model.
- A test set that always passes is too easy. Harder questions found real weaknesses.
- Model answers are not fully deterministic, so evaluation needs repeated runs.

## Limitations and next steps
- Paragraph splitting fails on documents with very long or very short paragraphs.
- Only one document is supported.
- Keyword-based evaluation is basic. Add an LLM judge and a larger question set.
- Persist the index with a Docker volume.
- Deploy the API online.

## Run locally
1. Install dependencies: `py -m pip install -r requirements.txt`
2. Create a `.env` file with `GEMINI_API_KEY=your_key`
3. Search only: `py main.py`
4. Full RAG in the terminal: `py rag.py`
5. Evaluation: `py eval.py`
6. API: `py -m uvicorn api:app --reload`
7. To force re-indexing manually, delete the `chroma_db/` folder.