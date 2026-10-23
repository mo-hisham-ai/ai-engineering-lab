from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    text: str


def shout(state: State) -> dict:
    return {"text": state["text"].upper()}


def add_mark(state: State) -> dict:
    return {"text": state["text"] + "x"}


builder = StateGraph(State)
builder.add_node("shout", shout)
builder.add_node("add_mark", add_mark)
builder.add_edge(START, "shout")
builder.add_edge("shout", "add_mark")
builder.add_edge("add_mark", END)
graph = builder.compile()

print(graph.invoke({"text": "hello"}))