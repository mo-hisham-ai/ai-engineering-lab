from typing import Any, TypedDict

from google.genai import types
from langgraph.graph import END, START, StateGraph

from agent import MODEL, client, config


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
    print(f"[model] tool calls requested: {len(calls)}")
    return {
        "llm_calls": state["llm_calls"] + 1,
        "tokens": state["tokens"] + (response.usage_metadata.total_token_count or 0),
        "pending_calls": calls,
        "answer": "" if calls else response.text,
    }


builder = StateGraph(AgentState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)
graph = builder.compile()


def ask(question: str) -> None:
    start = {
        "contents": [types.Content(role="user", parts=[types.Part.from_text(text=question)])],
        "pending_calls": [],
        "answer": "",
        "llm_calls": 0,
        "tokens": 0,
    }
    final = graph.invoke(start)
    print("answer:", final["answer"])
    print("tool calls:", [c.name for c in final["pending_calls"]])
    print("llm_calls:", final["llm_calls"], "| tokens:", final["tokens"])


ask("Say hello in one short sentence.")
ask("How many vacation days do I get?")