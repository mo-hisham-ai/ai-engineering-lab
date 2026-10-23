import ast
import operator
from datetime import datetime

from google.genai import types

from main import client, index_document, search

MODEL = "gemini-3.5-flash-lite"
MAX_STEPS = 5

SYSTEM_PROMPT = (
    "You are an HR assistant for Nile Tech. "
    "Use search_handbook for any question about company policy. "
    "Use calculate for any arithmetic instead of computing in your head. "
    "If the handbook does not contain the answer, say you don't know. "
    "Never use outside knowledge about company policy."
)

# ---------- Tools ----------
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculate(expression: str) -> float:
    """Evaluate a math expression such as '25 - 5 * 2'. Supports + - * / and parentheses."""
    return _eval(ast.parse(expression, mode="eval").body)


def search_handbook(query: str) -> str:
    """Search the employee handbook and return the most relevant paragraphs."""
    return "\n\n".join(doc for doc, _, _ in search(query, top_k=3))


def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now().isoformat()


TOOLS = {f.__name__: f for f in (calculate, search_handbook, get_current_time)}

# ---------- Agent loop ----------
config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=list(TOOLS.values()),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def run_agent(question: str) -> str:
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    llm_calls = 0
    tokens = 0

    for step in range(1, MAX_STEPS + 1):
        response = client.models.generate_content(
            model=MODEL, contents=contents, config=config
        )
        llm_calls += 1
        tokens += response.usage_metadata.total_token_count or 0

        calls = response.function_calls
        if not calls:
            print(f"\n[done] LLM calls: {llm_calls}, tokens: {tokens}")
            return response.text

        contents.append(response.candidates[0].content)
        parts = []
        for call in calls:
            args = dict(call.args or {})
            print(f"[step {step}] tool: {call.name}({args})")
            try:
                result = {"result": TOOLS[call.name](**args)}
            except Exception as e:
                result = {"error": str(e)}
            print(f"          -> {str(result)[:100]}")
            parts.append(types.Part.from_function_response(name=call.name, response=result))
        contents.append(types.Content(role="user", parts=parts))

    print(f"\n[stopped] reached MAX_STEPS={MAX_STEPS}, LLM calls: {llm_calls}")
    return "I could not finish within the allowed number of steps."


if __name__ == "__main__":
    index_document("doc.txt")
    while True:
        question = input("\nQuestion (or 'exit'): ")
        if question.lower() == "exit":
            break
        print(f"\nAnswer: {run_agent(question)}")