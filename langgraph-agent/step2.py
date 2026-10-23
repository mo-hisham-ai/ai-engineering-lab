from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    count: int


def increment(state: State) -> dict:
    print(f"count is {state['count']}, adding 1")
    return {"count": state["count"] + 1}


def route(state: State) -> str:
    if state["count"] < 5:
        return "again"
    return "stop"


builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_conditional_edges("increment", route, {"again": "increment", "stop": END})
graph = builder.compile()

print(graph.invoke({"count": 0}))