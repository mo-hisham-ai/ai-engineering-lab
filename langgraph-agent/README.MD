# 07 - LangGraph Agent

The same HR-handbook agent from `agent-basics`, rebuilt as a LangGraph graph. I built it in small steps to understand what LangGraph does underneath, instead of copying a finished example.

## The graph

```
START -> call_model --(tool requested?)--+-- yes --> run_tools --+
              ^                          |                       |
              |                          +-- no  --> END         |
              |                          +-- limit reached --> give_up --> END
              +--------------------------------------------------+
```

- **State** (`AgentState`): conversation, pending tool calls, answer, LLM call count, token count.
- **Nodes**: `call_model` (asks Gemini), `run_tools` (runs the tools the model asked for), `give_up` (stops when `MAX_STEPS` is reached).
- **Conditional edge** (`route`): after the model answers, choose between running tools, finishing, or giving up.
- **Loop**: the edge from `run_tools` back to `call_model`.

## Steps I built

| File | What it taught |
|---|---|
| `step1.py` | State, nodes and edges with no model. Edge order changes the result (`HELLOX` vs `HELLOx`). |
| `step2.py` | Conditional edge and a loop (a counter). A loop needs a stop condition. |
| `step3.py` | The agent shape with fake nodes: 3 model calls and 2 tool runs. |
| `step4.py` | A real model node. The model asks for a tool but nothing runs it yet, so the answer is empty. |
| `step5.py` | The complete agent: tools, routing and a step limit. |

`agent.py` (tools and config) and `main.py` (indexing and search) are copied from earlier labs. The tools are not the lesson here, the graph is.

## Example run (`step5.py`)

Question: `If I take 5 vacation days, how many do I have left?`

```
[step 1] tool: search_handbook({'query': 'vacation days allocation policy'})
          -> {'result': 'Annual leave: Every full-time employee receives 25 days of paid annual leave per year. L
[step 2] tool: calculate({'expression': '25 - 5'})
          -> {'result': 20}
[done] LLM calls: 3, tokens: 1152
answer: Full-time employees receive 25 days of paid annual leave per year. If you take 5 vacation days, you will have **20** days left.
```

This is the same behavior as the manual loop in `agent-basics`.

## Manual loop vs LangGraph

| `agent-basics/agent.py` | LangGraph |
|---|---|
| `for step in range(MAX_STEPS)` | the edge from `run_tools` back to `call_model` |
| `if not calls: return response.text` | `route` returns `"done"`, then `END` |
| check for `MAX_STEPS` | `route` returns `"limit"`, then the `give_up` node |
| local variables | fields in `AgentState` |

For an agent this small, LangGraph adds no new capability and the manual loop is shorter. It pays off when the agent grows: many branches, saving and resuming state, human approval before a sensitive tool, or several agents working together.

## A bug I hit

The first run of `step5.py` returned `{'result': ''}` for every search. The agent tried three different queries and then answered "I don't know" instead of inventing a number. The cause was in my code: this folder had a new, empty `chroma_db`, and `step5.py` never called `index_document("doc.txt")`. The old `agent.py` indexed inside its `__main__` block, so I missed it.

Lessons:
- When an agent gives a strange answer, read the tool results in the trace before blaming the model.
- The agent behaved safely with empty results: it refused to guess.



## Limitations and next steps
- Only a few questions were tried. No automated tests or evaluation for this agent yet.
- No retry on 503 errors in this lab.
- Next: evaluate the agent on a fixed question set (tool choice and answer quality), and add tracing of cost and latency.

## Run
1. Install dependencies: `py -m pip install langgraph google-genai python-dotenv chromadb`
2. Create a `.env` file with `GEMINI_API_KEY=your_key`
3. Run: `py step5.py`