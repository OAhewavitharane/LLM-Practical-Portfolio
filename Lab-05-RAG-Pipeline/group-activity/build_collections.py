import json
import chromadb
import ollama
from chunking import load_pages, fixed_chunks, section_chunks, child_chunks

pdf_path = "Lab-05-RAG-Pipeline/data/NSBM-MSc-in-IT-Guide.pdf"
db_path = "Lab-05-RAG-Pipeline/group-activity/chroma_db"
parents_path = "Lab-05-RAG-Pipeline/group-activity/parents.json"

pages = load_pages(pdf_path)
client = chromadb.PersistentClient(path=db_path)


def store(name, chunks):
    # start fresh every run so old chunks don't stay behind
    client.get_or_create_collection(name=name)
    client.delete_collection(name=name)
    collection = client.create_collection(name=name)

    # one embed call for 20 chunks is much faster than one call per chunk
    for start in range(0, len(chunks), 20):
        batch = chunks[start:start + 20]
        response = ollama.embed(model="qwen3-embedding:0.6b", input=[c["text"] for c in batch])
        collection.add(
            ids=[name + "_" + str(start + i) for i in range(len(batch))],
            documents=[c["text"] for c in batch],
            embeddings=response["embeddings"],
            metadatas=[{"page": c.get("page", 0)} for c in batch]
        )
    print(name, "-", collection.count(), "chunks")


store("fixed", fixed_chunks(pages))
store("section", section_chunks(pages))
store("child", child_chunks(pages))

# the parents are the full pages; parent-child retrieval looks them up by page number
with open(parents_path, "w", encoding="utf-8") as f:
    json.dump({str(page_no): text for page_no, text in pages}, f, ensure_ascii=False, indent=1)
print("parents.json -", len(pages), "pages")
