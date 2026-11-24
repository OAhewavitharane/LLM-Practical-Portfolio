import os
import chromadb
import ollama

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = "Lab-05-RAG-Pipeline/data/NSBM-MSc-in-IT-Guide.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print("PDF loaded successfully")
print("Total characters:", len(text))


# Split the PDF text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(text)

print("Total chunks:", len(chunks))          # Check the number of chunks
print("First chunk:")
print(chunks[0])


# Create embeddings for each chunk
embeddings = []

for chunk in chunks:
    response = ollama.embed(
        model="qwen3-embedding:0.6b",
        input=chunk
    )

    embeddings.append(response["embeddings"][0])

print("Embeddings created:", len(embeddings))


client = chromadb.PersistentClient(                      # Create a ChromaDB client
    path="Lab-05-RAG-Pipeline/chroma_db"
)

collection = client.get_or_create_collection(            # Create a collection to store the PDF data
    name="nsbm_guide"
)

# Store the chunks 
for i, chunk in enumerate(chunks):
    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embeddings[i]]
    )

print("PDF data stored in ChromaDB")


