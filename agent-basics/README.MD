# 06 - Agent Basics: a manual agent loop

An agent written by hand, without LangChain or LangGraph, to understand what those frameworks do underneath. The model decides which tool to call, my code runs it, the result goes back to the model, and the loop repeats until the model can answer.

## How it works

```
question
   |
   v
+------------------------------+
| LLM: answer, or call a tool? |<-----------------+
+------------------------------+                  |
   | tool call                                    |
   v                                              |
run the tool in my code (errors are returned     |
to the model instead of crashing)                 |
   |                                              |
   +---- add the result to the conversation ------+

no tool call -> final answer      MAX_STEPS reached -> stop with a message
```

## Tools
| Tool | Purpose |
|---|---|
| `search_handbook(query)` | semantic search over the handbook (ChromaDB, reused from `vector-db`) |
| `calculate(expression)` | safe arithmetic (parses the expression instead of using `eval`) |
| `get_current_time()` | current date and time |

## Files
- `agent.py`: tools and the agent loop.
- `main.py`: indexing and search, copied from `vector-db`.
- `doc.txt`: sample document (a fictional employee handbook).

## What is different from tool calling in lab 02
In lab 02 the SDK ran my functions automatically and hid the loop. Here automatic calling is disabled (`AutomaticFunctionCallingConfig(disable=True)`), so I handle each step myself: read the tool calls, run them, send the results back, and decide when to stop.

## Example run

Question: `If I take 5 vacation days, how many do I have left?`

```
[step 1] tool: search_handbook({'query': 'vacation days allowance remaining accrual'})
          -> {'result': 'Annual leave: Every full-time employee receives 25 days of paid annual leave per year. L
[step 2] tool: calculate({'expression': '25 - 5'})
          -> {'result': 20}

[done] LLM calls: 3, tokens: 1157

Answer: Full-time employees receive 25 days of paid annual leave per year. If you take 5 vacation days, you will have 20 days left.
```

What I noticed:
- The model combined two tools in the right order: search first, then calculate with the number it found. The plain RAG pipeline in `vector-db` could not do this kind of question reliably.
- The model rewrote my question into its own search query (`vacation days allowance remaining accrual`). This can help retrieval, but it can also drift away from what I asked, so the trace is needed to see it.
- Two tool calls cost three LLM calls (decide to search, decide to calculate, write the answer). An agent is slower and uses more tokens than a fixed pipeline that calls the model once.

## Design choices
- **`MAX_STEPS = 5`:** a loop without a limit can run forever and burn tokens if the model keeps calling tools.
- **Tool errors go back to the model:** a failing tool (for example a bad expression) returns `{"error": ...}` instead of crashing the program.
- **`calculate` does not use `eval`:** it parses the expression and allows only numbers and `+ - * /`, so the model cannot run arbitrary code through this tool.
- **The system prompt restricts policy answers to the handbook**, as in the RAG labs.

## Limitations and next steps
- Only one successful question has been tested so far. The loop limit and the tool-error path are written but not tested yet.
- No automated tests or evaluation for the agent yet.
- No retry on 503 errors in this lab (the earlier RAG lab has it).
- Next: rebuild the same agent with LangGraph and compare it with this manual loop.

## Run
1. Install dependencies: `py -m pip install google-genai python-dotenv chromadb`
2. Create a `.env` file with `GEMINI_API_KEY=your_key`
3. Run: `py agent.py`