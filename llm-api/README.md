# 01 - LLM API Fundamentals

Python script that sends a question to the Gemini API and returns structured JSON.

## What I learned
- Calling an LLM API directly (no frameworks)
- Storing the API key in `.env` (never in code)
- System prompt vs user prompt
- Structured JSON output with `response_mime_type`
- Retry with exponential backoff on 503 errors
- Choosing a model by listing the available ones (`list_models.py`)
- Validating LLM output with Pydantic (`BaseModel`, `Literal`)
- Writing tests with pytest (valid case, missing field, invalid value)

## Run
1. `py -m pip install google-genai python-dotenv`
2. Create `.env` with `GEMINI_API_KEY=your_key`
3. `py main.py`