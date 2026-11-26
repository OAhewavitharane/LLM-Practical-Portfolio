# part C: the memory agent from part B plus the Lab 5 RAG as a third tool
import os
import sqlite3
from datetime import datetime
import chromadb
import ollama
from pydantic import ValidationError
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# num_ctx: room for 5 guide chunks + memory + chat history (same as the Lab 5 group activity)
llm = ChatOllama(model="llama3.2:3b", temperature=0, num_ctx=4096)

HERE = os.path.dirname(os.path.abspath(__file__))

# LONG-TERM MEMORY: a SQLite file, so the facts are still there after the program closes
# same file as memory_agent.py, so what the student said in part B is still remembered
DB_PATH = os.path.join(HERE, "student_memory.db")
conn = sqlite3.connect(DB_PATH)
conn.execute("""CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL,
    saved_at TEXT NOT NULL
)""")
conn.commit()


def save_memory(fact):
    # without this check, saying "My name is Alex" twice saves it twice
    found = conn.execute("SELECT id FROM memories WHERE lower(fact) = lower(?)", (fact,)).fetchone()
    if found:
        return False
    conn.execute("INSERT INTO memories (fact, saved_at) VALUES (?, ?)",
                 (fact, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    return True


extract_prompt = """You read one message from a student and pick out facts about them worth remembering,
like their name, degree, university, year, modules, goals or preferences.
Write each fact on its own line, starting with "- The user".
A question is not a fact. Never answer the message. If there are no facts, reply with NONE.

Message: My name is Alex and I study Software Engineering at NSBM
Facts:
- The user's name is Alex.
- The user studies Software Engineering at NSBM.

Message: What is my name?
Facts:
NONE

Message: What is 15 * 12?
Facts:
NONE

Message: I'm in my third year and my favourite module is AI
Facts:
- The user is in their third year.
- The user's favourite module is AI.

Message: What do I study?
Facts:
NONE"""


def extract_memories(user_input):
    # a separate call with no tools, it only has to decide what to save
    # the examples in the prompt matter: with only rules, the 3B model answered
    # "What is my name?" here and saved "I don't have that information" as a memory
    reply = llm.invoke([SystemMessage(content=extract_prompt),
                        HumanMessage(content="Message: " + user_input + "\nFacts:")])
    facts = []
    for line in reply.content.splitlines():
        fact = line.strip().lstrip("-* ").strip()
        # anything not written as a fact about the user is the model answering, so skip it
        if fact.lower().startswith("the user"):
            facts.append(fact)
    return facts


@tool
def calculator(expression: str) -> str:
    """Work out a maths expression and return the answer, for example "25*4" or "(10+5)/3"."""
    expression = expression.replace("×", "*").replace("÷", "/").replace(" ", "")
    # only numbers and + - * / ( ) . so eval can't run anything else
    if not expression or not all(ch in "0123456789+-*/()." for ch in expression):
        return "I can only calculate numbers with + - * / and brackets."
    try:
        return str(eval(expression))
    except (SyntaxError, ZeroDivisionError):
        return "That expression can't be calculated: " + expression


# these are in almost every question, so they don't help find the right memory
STOP_WORDS = ["what", "who", "which", "where", "how", "the", "are", "does", "you", "your",
              "about", "know", "remember", "tell", "and", "user", "user's"]


@tool
def search_student_memory(query: str) -> str:
    """Search the long-term memory for facts the student told you before,
    like their name, degree, modules or goals. Use it when the student asks about themselves."""
    words = [w.strip("?.,!").lower() for w in query.split()]
    words = [w for w in words if len(w) > 2 and w not in STOP_WORDS]

    rows = []
    if words:
        # a fact matches if it has any of the words, e.g. "name" finds "The user's name is Alex"
        sql = "SELECT fact FROM memories WHERE " + " OR ".join(["lower(fact) LIKE ?"] * len(words))
        rows = conn.execute(sql + " ORDER BY id DESC", ["%" + w + "%" for w in words]).fetchall()

    if not rows:
        # "study" doesn't match "studies", so if nothing matched give back everything saved
        # (newest first, in case the student changed something)
        rows = conn.execute("SELECT fact FROM memories ORDER BY id DESC LIMIT 20").fetchall()

    if not rows:
        return "No matching memories found."
    return "\n".join(row[0] for row in rows)


# the ChromaDB that Lab 5's ingest.py built from the NSBM MSc in IT guide (collection nsbm_guide)
chroma_client = chromadb.PersistentClient(path=os.path.join(HERE, "..", "Lab-05-RAG-Pipeline", "chroma_db"))
guide_collection = chroma_client.get_collection(name="nsbm_guide")


@tool
def search_university_knowledge(query: str) -> str:
    """Search the NSBM MSc in IT programme guide. Use it for any question about the university
    or the programme, like modules, credits, grades, GPA, fees, entry requirements or rules."""
    print(f"\n[EXECUTING] Searching university knowledge for: '{query}'...")
    # has to be the same embedding model ingest.py used, or the vectors don't match
    response = ollama.embed(model="qwen3-embedding:0.6b", input=query)
    # started with 3 chunks like rag_pipeline.py, but for "medium of instruction" the cover page
    # came first and the answer wasn't in the 3, so 5 like app.py and the group activity
    results = guide_collection.query(query_embeddings=[response["embeddings"][0]], n_results=5)

    found = []
    for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
        found.append("Source: " + metadata["source"] + "\n" + document)
    return "\n\n".join(found)


tools = [calculator, search_student_memory, search_university_knowledge]
llm_with_tools = llm.bind_tools(tools)
tools_by_name = {t.name: t for t in tools}

system_prompt = """You are a friendly student assistant at NSBM. You have three tools:
- calculator: any maths calculation
- search_student_memory: search the facts the student told you before (name, degree, modules, goals)
- search_university_knowledge: search the NSBM MSc in IT programme guide (modules, credits, grades, fees, rules)
Use the calculator for any maths. Use search_student_memory when the student asks about themselves.
Use search_university_knowledge for any question about the university or the programme.
Never make up facts. Keep replies short."""

print("=" * 40)
print(" INTELLIGENT STUDENT AI AGENT")
print(" SHORT-TERM MEMORY")
print(" LONG-TERM SQLITE MEMORY")
print(" WORKING MEMORY")
print(" UNIVERSITY KNOWLEDGE (RAG)")
print("=" * 40)
print("Type exit to stop.")

# SHORT-TERM MEMORY: this conversation only, it's gone when the program closes
chat_history = []

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break

    new_facts = []
    for fact in extract_memories(user_input):
        if save_memory(fact):
            new_facts.append(fact)

    if new_facts:
        print("\n[NEW MEMORIES DETECTED]")
        for fact in new_facts:
            print("[SAVED]", fact)
    else:
        print("\n[MEMORY] No new information to save.")

    # WORKING MEMORY: what the model works with for this one question, built again every turn:
    # instructions + facts from SQLite + recent chat + the new message (+ tool results below)
    # (only the latest 20 facts, older ones can still be found with the search tool)
    rows = conn.execute("SELECT fact FROM memories ORDER BY id DESC LIMIT 20").fetchall()
    memory_text = ""
    if rows:
        memory_text = "\n\nWhat you remember about this student:\n" + "\n".join("- " + row[0] for row in rows)

    messages = [SystemMessage(content=system_prompt + memory_text)] + chat_history + [HumanMessage(content=user_input)]
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        results = []
        for tool_call in response.tool_calls:
            print("\n[TOOL SELECTED]", tool_call["name"])
            print("[TOOL ARGUMENTS]", tool_call["args"])

            selected = tools_by_name.get(tool_call["name"])
            if selected is None:
                result = "There is no tool called " + tool_call["name"]
            else:
                try:
                    result = selected.invoke(tool_call["args"])
                except ValidationError:
                    result = "The tool was called with the wrong arguments."

            # a RAG result is 5 chunks (about 5000 characters), so only the start is printed
            # the model still gets all of it
            if len(result) > 500:
                print("[TOOL RESULT]", result[:500] + "...")
            else:
                print("[TOOL RESULT]", result)
            # the tool name and arguments go in too: with only "21" the model said
            # it couldn't find the calculation result
            results.append(tool_call["name"] + " " + str(tool_call["args"]) + " returned:\n" + result)

        # the answer comes from a plain prompt with the results written in, like the RAG in Lab 5
        # when the results went back as tool messages, after a restart llama3.2 answered
        # "What is my name?" with "I don't know yet", then with the degree, even though the
        # search had found "The user's name is Oshani"
        answer_prompt = ("You are a friendly student assistant at NSBM. Answer the student's question "
                         "using only the information below. If the answer isn't there, say you couldn't "
                         "find it. Keep it short." + memory_text +
                         "\n\nResults of the tools you just used:\n" + "\n".join(results))
        response = llm.invoke([SystemMessage(content=answer_prompt)] + chat_history + [HumanMessage(content=user_input)])

    # only the question and the final answer go into short-term memory, not the tool steps
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))
    # last 10 messages (5 turns) is enough, and the prompt doesn't keep growing
    chat_history = chat_history[-10:]
    print("\n[SHORT-TERM MEMORY] Conversation updated.")

    print("\nAgent:", response.content)

conn.close()
