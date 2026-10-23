from typing import Any, TypedDict

from google.genai import types
from langgraph.graph import END, START, StateGraph
from main import index_document

from agent import MAX_STEPS, MODEL, TOOLS, client, config


class AgentState(TypedDict):
    contents: list[Any]
    pending_calls: list[Any]
    answer: str
    llm_calls: int
    tokens: int


def call_model(state: AgentState) -> dict:
    response = client.models.generate_content(
        model=MODEL, contents=state["contents"], config=config
    )
    calls = response.function_calls or []
    update = {
        "llm_calls": state["llm_calls"] + 1,
        "tokens": state["tokens"] + (response.usage_metadata.total_token_count or 0),
        "pending_calls": calls,
        "answer": "" if calls else response.text,
    }
    if calls:
        update["contents"] = state["contents"] + [response.candidates[0].content]
    return update


def run_tools(state: AgentState) -> dict:
    parts = []
    for call in state["pending_calls"]:
        args = dict(call.args or {})
        print(f"[step {state['llm_calls']}] tool: {call.name}({args})")
        try:
            result = {"result": TOOLS[call.name](**args)}
        except Exception as e:
            result = {"error": str(e)}
        print(f"          -> {str(result)[:100]}")
        parts.append(types.Part.from_function_response(name=call.name, response=result))
    tool_message = types.Content(role="user", parts=parts)
    return {"contents": state["contents"] + [tool_message], "pending_calls": []}


def give_up(state: AgentState) -> dict:
    print(f"[stopped] reached MAX_STEPS={MAX_STEPS}")
    return {"answer": "I could not finish within the allowed number of steps."}


def route(state: AgentState) -> str:
    if not state["pending_calls"]:
        return "done"
    if state["llm_calls"] >= MAX_STEPS:
        return "limit"
    return "tools"


builder = StateGraph(AgentState)
builder.add_node("call_model", call_model)
builder.add_node("run_tools", run_tools)
builder.add_node("give_up", give_up)
builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model", route, {"tools": "run_tools", "done": END, "limit": "give_up"}
)
builder.add_edge("run_tools", "call_model")
builder.add_edge("give_up", END)
graph = builder.compile()
index_document("doc.txt")

def ask(question: str) -> None:
    start = {
        "contents": [types.Content(role="user", parts=[types.Part.from_text(text=question)])],
        "pending_calls": [],
        "answer": "",
        "llm_calls": 0,
        "tokens": 0,
    }
    final = graph.invoke(start)
    print(f"[done] LLM calls: {final['llm_calls']}, tokens: {final['tokens']}")
    print("answer:", final["answer"])


ask("Say hello in one short sentence.")
ask("If I take 5 vacation days, how many do I have left?")