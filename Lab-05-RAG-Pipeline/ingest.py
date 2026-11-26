import os
import chromadb
import ollama
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data_folder = "Lab-05-RAG-Pipeline/data"

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

client = chromadb.PersistentClient(path="Lab-05-RAG-Pipeline/chroma_db")
collection = client.get_or_create_collection(name="nsbm_guide")

for file_name in os.listdir(data_folder):
    if not file_name.endswith(".pdf"):
        continue

    reader = PdfReader(os.path.join(data_folder, file_name))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # scanned pages come back empty
            text += page_text + "\n"

    chunks = splitter.split_text(text)
    print(file_name, "-", len(chunks), "chunks")

    for i, chunk in enumerate(chunks):
        response = ollama.embed(model="qwen3-embedding:0.6b", input=chunk)
        collection.add(
            ids=[file_name + "_" + str(i)],
            documents=[chunk],
            embeddings=[response["embeddings"][0]],
            metadatas=[{"source": file_name}]
        )

print("Total chunks in ChromaDB:", collection.count())