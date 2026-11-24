
import chromadb
import ollama


# Connect to the stored ChromaDB database
client = chromadb.PersistentClient(
    path="Lab-05-RAG-Pipeline/chroma_db"
)

collection = client.get_collection(
    name="nsbm_guide"
)

question = input("Ask a question about the PDF: ")      # Ask a question


# Convert the question into an embedding
response = ollama.embed(
    model="qwen3-embedding:0.6b",
    input=question
)

question_embedding = response["embeddings"][0]


# Search for relevant chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

context = "\n\n".join(results["documents"][0])


 # generate an answer

prompt = f"""
Answer using only the context.
If the answer is not in the context, say you
couldn't find it in the document.

Context: {context}

Question: {question}

Answer:
"""

answer = ollama.generate(
    model="llama3.2:3b",
    prompt=prompt
)

print(answer["response"])
