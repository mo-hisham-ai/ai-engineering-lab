from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    steps: int
    need_tool: bool


def call_model(state: State) -> dict:
    steps = state["steps"] + 1
    need_tool = steps < 3  # نتظاهر أن الموديل يطلب أداة في أول مرتين
    print(f"[model] call {steps}, wants a tool: {need_tool}")
    return {"steps": steps, "need_tool": need_tool}


def run_tools(state: State) -> dict:
    print("[tools] running the tool")
    return {}


def route(state: State) -> str:
    if state["need_tool"]:
        return "tools"
    return "done"


builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_node("run_tools", run_tools)
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", route, {"tools": "run_tools", "done": END})
builder.add_edge("run_tools", "call_model")
graph = builder.compile()

print(graph.invoke({"question": "hi", "steps": 0, "need_tool": False}))