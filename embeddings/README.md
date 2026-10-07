# 03 - Embeddings

Turning text into numbers that represent its meaning, then comparing texts by meaning using cosine similarity.

## What I built
A small script that:
1. Sends three sentences to the Gemini embedding model.
2. Gets back a vector of 3072 numbers for each sentence.
3. Measures how close two sentences are in meaning with `cosine_similarity`.

## What I learned
- An embedding is a list of numbers that represents the meaning of a text. Texts with similar meaning get similar numbers, even when they share no words.
- Cosine similarity measures how close two embeddings are. Closer to 1 means closer in meaning.
- What matters is the ranking (which text is closest), not the absolute number. Even unrelated texts get a medium score.
- RAG needs embeddings to search documents by meaning instead of exact keywords: embed the documents once, embed the question, then take the closest chunks and give them to the LLM.

## Experiment

| Pair | Similarity |
|---|---|
| "How do I reset my password?" vs "I forgot my login credentials" | 0.683 |
| "How do I reset my password?" vs "What is the best pizza recipe?" | 0.525 |

The two password-related sentences share almost no words, but the model still ranked them closer than the pizza sentence.

## Run
1. Install dependencies: `py -m pip install google-genai python-dotenv`
2. Create a `.env` file with `GEMINI_API_KEY=your_key`
3. Run: `py main.py`