# 05 - Vector Database (ChromaDB)

Semantic search over a document using ChromaDB as a persistent vector database, with paragraph-based chunking.

## What changed compared to `rag-basics`
| | rag-basics | vector-db |
|---|---|---|
| Chunking | fixed number of characters | one chunk per paragraph |
| Storage | in memory, lost on exit | on disk (`chroma_db/`) |
| Embeddings | recomputed on every run | computed once, reused |
| Search | my own cosine similarity loop | `collection.query()` |
| Score | similarity (higher is closer) | distance (**lower** is closer) |

## How it works

```
First run:
doc.txt -> split by paragraph -> embed -> collection.add() -> saved in chroma_db/

Every later run:
chroma_db/ already has the chunks -> skip embedding -> search directly
```

## Files
- `main.py`: indexing and search.
- `doc.txt`: sample document (a fictional employee handbook).
- `chroma_db/`: the local database, created automatically and not committed to Git.

## Results

First run: `Indexed 7 chunks`. Second run: `Already indexed: 7 chunks (no embedding needed)`, so no embedding calls were needed to start.

Question: "How many vacation days do I get?"

| Rank | Distance | Chunk |
|---|---|---|
| 1 | 0.621 | Annual leave (contains the answer: 21 days) |
| 2 | 0.645 | Sick leave |
| 3 | 0.771 | Remote work |

Every result is a complete paragraph. There are no chunks cut in the middle of a word, which was the main problem in `rag-basics`. The question says "vacation" and the document says "annual leave", and the right chunk still ranked first, so this is still matching by meaning.

## Experiment: editing the document
I changed "21 days" to "30 days" in `doc.txt`. The change did **not** appear in search results until I deleted the `chroma_db/` folder and ran the script again. After re-indexing, the result showed "30 days".

**Why:** `index_document()` checks `collection.count() > 0` and returns early. It never reads `doc.txt` again, so the database keeps the old text forever. This is the stale index problem.

Possible fixes (not done yet):
- Store a hash of the file and re-index when it changes.
- Use `collection.upsert()` and re-embed only the chunks that changed.
- Add a `--reindex` option to rebuild on demand.

## What I learned
- A vector database stores embeddings on disk and searches them quickly, so I do not recompute them on every run or lose them when the program closes.
- Splitting by paragraph keeps each idea in one piece. Splitting by characters can cut a sentence in half and lose the answer.
- Distance and similarity are opposites: in ChromaDB a smaller number means a closer match.
- Persistent storage creates a new responsibility: keeping the index in sync with the source document.

## Limitations and next steps
- Detect document changes and re-index automatically.
- Connect the LLM to these results to get full answers with sources (RAG on top of ChromaDB).
- Paragraph splitting fails on documents with very long or very short paragraphs.
- Measure retrieval quality with a small test set of questions and expected chunks.

## Run
1. Install dependencies: `py -m pip install google-genai python-dotenv chromadb`
2. Create a `.env` file with `GEMINI_API_KEY=your_key`
3. Run: `py main.py`
4. To force re-indexing, delete the `chroma_db/` folder.