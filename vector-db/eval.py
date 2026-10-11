import time
import re
from main import index_document
from rag import answer

# chunk: رقم المقطع الذي فيه الجواب | keyword: كلمة لازم تظهر في الجواب
TESTS = [
    {"q": "How many vacation days do I get?", "chunk": 3, "keyword": "25"},
    {"q": "Can I work from home?", "chunk": 2, "keyword": "two days"},
    {"q": "How much is the learning budget?", "chunk": 6, "keyword": "1,500"},
    {"q": "Do I need a medical certificate for sick leave?", "chunk": 4, "keyword": "two consecutive days"},
    {"q": "What time does the office open?", "chunk": 1, "keyword": "9 AM"},
    {"q": "Who do I report lost equipment to?", "chunk": 5, "keyword": "IT department"},
    {"q": "كم عدد أيام الإجازة السنوية؟", "chunk": 3, "keyword": "25"},
    {"q": "What is the CEO's name?", "chunk": None, "keyword": None},
    {"q": "What is the capital of France?", "chunk": None, "keyword": None},
    {"q": "Can I take a day off when I feel ill?", "chunk": 4, "keyword": "10"},
    {"q": "Can I carry over 10 unused leave days?", "chunk": 3, "keyword": r"\b5\b"},
    {"q": "How many total paid days off do I get, vacation and sick?", "chunk": 3, "keyword": "35"},
]


def run() -> None:
    index_document("doc.txt")
    retrieval_ok = answer_ok = refusal_ok = 0
    in_doc = sum(1 for t in TESTS if t["chunk"] is not None)
    out_doc = len(TESTS) - in_doc

    for t in TESTS:
        text, top = answer(t["q"])
        found = [meta["chunk"] for _, meta, _ in top]
        text_low = text.lower()

        if t["chunk"] is not None:
            r = t["chunk"] in found
            a = re.search(t["keyword"], text, re.IGNORECASE) is not None
            retrieval_ok += r
            answer_ok += a
            status = f"retrieval={'OK' if r else 'FAIL'} answer={'OK' if a else 'FAIL'}"
        else:
            refused = "i don't know" in text_low
            refusal_ok += refused
            status = f"refusal={'OK' if refused else 'FAIL'}"

        print(f"{status} | {t['q']}")
        if "FAIL" in status:
            print(f"    answer: {text[:100]}")
            print(f"    retrieved chunks: {found}")
        time.sleep(1)

    print("\n--- Summary ---")
    print(f"Retrieval hit@3: {retrieval_ok}/{in_doc}")
    print(f"Correct answers: {answer_ok}/{in_doc}")
    print(f"Correct refusals: {refusal_ok}/{out_doc}")


if __name__ == "__main__":
    run()