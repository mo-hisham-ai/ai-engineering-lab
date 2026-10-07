# 02 - Tool Calling

A Gemini model that decides by itself when to call my Python functions (`add` and `get_current_time`).

## What I built
A small script that:
1. Takes a question from the user.
2. Passes two Python functions to the model as tools.
3. Lets the model decide whether to use a tool or answer directly.

## What I learned
- The model does not run my code. It asks the SDK to call the function on my machine, then writes the final answer from the result.
- The model chooses the tool (or no tool at all) based on the function description, not on `if` conditions I wrote.
- Docstrings and type hints are how the model understands what a function does and what inputs it takes.
- This is the foundation of agents: model + tools + a decision about when to use each tool.

## Experiments

| Question | Tool called | Why |
|---|---|---|
| `What is 15 + 27?` | `add(15, 27)` | Math question, matching tool available |
| `What time is it now?` | `get_current_time()` | The model cannot know the current time by itself |
| `What is Python?` | none | The model already knows the answer |

**Docstring experiment:** I removed the docstring from `add` and the model still called it, because the function name and parameters were clear. With vague names (like `process`) or several similar tools, the docstring becomes essential. I always write one.

## Run
1. Install dependencies: `py -m pip install google-genai python-dotenv`
2. Create a `.env` file with `GEMINI_API_KEY=your_key`
3. Run: `py main.py`
4. Type a question when prompted.