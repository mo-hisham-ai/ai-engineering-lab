import re
import time

from graph_agent import index_document, run_agent

REFUSAL = r"don't know|do not know|no mention|not mentioned|cannot find|can't find|can only help"

# q: the question | tools: the exact set of tools the agent SHOULD use
# answer: a regex the answer must match (None = do not check the answer)
TESTS = [
    {"q": "How many vacation days do I get?",
     "tools": {"search_handbook"}, "answer": r"\b25\b"},
    {"q": "If I take 5 vacation days, how many do I have left?",
     "tools": {"search_handbook", "calculate"}, "answer": r"\b20\b"},
    {"q": "What is 15% of 1500?",
     "tools": {"calculate"}, "answer": r"\b225\b"},
    {"q": "How many total paid days off do I get, vacation plus sick?",
     "tools": {"search_handbook", "calculate"}, "answer": r"\b35\b"},
    {"q": "What time is it now?",
     "tools": {"get_current_time"}, "answer": None},
    {"q": "Say hello in one short sentence.",
     "tools": set(), "answer": None},
    {"q": "What time does the office open?",
     "tools": {"search_handbook"}, "answer": r"9\s*AM"},
    {"q": "What is the CEO's name?",
     "tools": {"search_handbook"}, "answer": REFUSAL},
    {"q": "What is the capital of France?",
     "tools": set(), "answer": REFUSAL},
]


def run_with_retry(question: str, tries: int = 3):
    for attempt in range(1, tries + 1):
        start = time.perf_counter()
        try:
            result = run_agent(question)
            return result, time.perf_counter() - start
        except Exception as e:
            msg = str(e)
            if ("429" in msg or "503" in msg) and attempt < tries:
                wait = 45 if "429" in msg else 5 * attempt
                print(f"    {msg[:3]} error, waiting {wait}s then retrying ({attempt}/{tries})")
                time.sleep(wait)
            else:
                raise


def run_tests() -> None:
    index_document("doc.txt")
    tool_ok = answer_ok = finished = 0
    total_calls = total_tokens = 0
    total_seconds = 0.0

    for t in TESTS:
        try:
            r, seconds = run_with_retry(t["q"])
        except Exception as e:
            print(f"ERROR | {t['q']} | {str(e)[:120]}")
            continue

        used = set(r["tools_used"])
        t_ok = used == t["tools"]
        a_ok = t["answer"] is None or re.search(t["answer"], r["answer"], re.IGNORECASE) is not None

        finished += 1
        tool_ok += t_ok
        answer_ok += a_ok
        total_calls += r["llm_calls"]
        total_tokens += r["tokens"]
        total_seconds += seconds

        print(
            f"tools={'OK' if t_ok else 'FAIL'} answer={'OK' if a_ok else 'FAIL'} "
            f"| calls={r['llm_calls']} tokens={r['tokens']} time={seconds:.1f}s | {t['q']}"
        )
        print(f"    answer: {r['answer'][:100]}")
        if not (t_ok and a_ok):
            print(f"    expected tools: {sorted(t['tools'])} | used: {sorted(used)}")
        time.sleep(1)

    print("\n--- Summary ---")
    print(f"Questions finished: {finished}/{len(TESTS)}")
    if finished:
        print(f"Correct tool choice: {tool_ok}/{finished}")
        print(f"Correct answers: {answer_ok}/{finished}")
        print(f"Average LLM calls: {total_calls / finished:.1f}")
        print(f"Average tokens: {total_tokens / finished:.0f}")
        print(f"Average time: {total_seconds / finished:.1f}s")


if __name__ == "__main__":
    run_tests()