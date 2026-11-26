import csv
import json
import time
from rag import retrieve, answer

base = "Lab-05-RAG-Pipeline/group-activity/"
methods = ["fixed", "section", "parent_child"]

with open(base + "eval_questions.json", encoding="utf-8") as f:
    questions = json.load(f)


def flat(text):
    # the PDF breaks lines in the middle of phrases, so compare with single spaces
    return " ".join(text.split()).lower()


def fact_score(facts, text):
    # a fact counts if any one of its accepted spellings is in the text
    text = flat(text)
    found = sum(1 for spellings in facts if any(flat(s) in text for s in spellings))
    return found / len(facts)


# the first call loads both models into memory, so do it once before timing anything
answer("warm up", retrieve("fixed", "warm up"))

rows = []
for q in questions:
    for method in methods:
        start = time.time()
        context = retrieve(method, q["question"])
        reply = answer(q["question"], context)
        seconds = time.time() - start

        rows.append({
            "id": q["id"],
            "method": method,
            "context_recall": round(fact_score(q["facts"], context), 2),
            "answer_score": round(fact_score(q["facts"], reply), 2),
            "seconds": round(seconds, 1),
            "context_chars": len(context),
            "answer": " ".join(reply.split()),
        })
        r = rows[-1]
        print(f"Q{q['id']:>2} {method:12} context {r['context_recall']:.2f}  answer {r['answer_score']:.2f}  {r['seconds']}s")

with open(base + "results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# averages per method as a markdown table, ready for the report
lines = ["| Method | Context recall | Answer score | Avg seconds | Avg context chars |",
         "|---|---|---|---|---|"]
for method in methods:
    r = [row for row in rows if row["method"] == method]
    n = len(r)
    lines.append(f"| {method} | {sum(x['context_recall'] for x in r) / n:.2f} | "
                 f"{sum(x['answer_score'] for x in r) / n:.2f} | "
                 f"{sum(x['seconds'] for x in r) / n:.1f} | "
                 f"{sum(x['context_chars'] for x in r) // n} |")

with open(base + "results_summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n" + "\n".join(lines))
