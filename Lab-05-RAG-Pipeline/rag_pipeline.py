
import chromadb
import ollama

client = chromadb.PersistentClient(path="Lab-05-RAG-Pipeline/chroma_db")           # Connect to the stored ChromaDB database
collection = client.get_collection(name="nsbm_guide")

while True:
    question = input("\nAsk a university question (or exit): ")
    if question.lower() == "exit":
        break

    response = ollama.embed(model="qwen3-embedding:0.6b", input=question)
    question_embedding = response["embeddings"][0]

    results = collection.query(query_embeddings=[question_embedding], n_results=3)
    context = "\n\n".join(results["documents"][0])

    for doc in results["documents"][0]:
      
     print("  retrieved:", doc[:80].replace("\n", " "))



    prompt = f"""Answer using only the context.
If the answer is not in the context, say you couldn't find it in the document.

Context: {context}

Question: {question}

Answer:"""

    answer = ollama.generate(model="llama3.2:3b", prompt=prompt)
    print("\nAnswer:", answer["response"])