import json
import chromadb
import ollama

db_path = "Lab-05-RAG-Pipeline/group-activity/chroma_db"
parents_path = "Lab-05-RAG-Pipeline/group-activity/parents.json"

# how much each method sends to the model
fixed_k = 5        # same as rag_pipeline.py
section_k = 5
child_k = 10       # children searched for parent-child
parent_pages = 2   # full pages sent for parent-child

client = chromadb.PersistentClient(path=db_path)
with open(parents_path, encoding="utf-8") as f:
    parents = json.load(f)


def retrieve(method, question):
    response = ollama.embed(model="qwen3-embedding:0.6b", input=question)
    question_embedding = response["embeddings"][0]

    if method == "fixed":
        results = client.get_collection("fixed").query(query_embeddings=[question_embedding], n_results=fixed_k)
        return "\n\n".join(results["documents"][0])

    if method == "section":
        results = client.get_collection("section").query(query_embeddings=[question_embedding], n_results=section_k)
        return "\n\n".join(results["documents"][0])

    # parent-child: search the small chunks, then swap each one for the whole page it came from
    results = client.get_collection("child").query(query_embeddings=[question_embedding], n_results=child_k)
    pages = []
    for metadata in results["metadatas"][0]:
        if metadata["page"] not in pages:
            pages.append(metadata["page"])
    return "\n\n".join(parents[str(page)] for page in pages[:parent_pages])


def answer(question, context):
    prompt = f"""Answer the question using only the context.
If the answer is not in the context, say "I couldn't find the answer in the document."

Context:
{context}

Question: {question}

Answer:"""
    # temperature 0 so a question gives the same answer every run (fair comparison)
    # num_ctx 4096 because two full pages can be longer than ollama's smaller default window
    result = ollama.generate(model="llama3.2:3b", prompt=prompt, options={"temperature": 0, "num_ctx": 4096})
    return result["response"]
