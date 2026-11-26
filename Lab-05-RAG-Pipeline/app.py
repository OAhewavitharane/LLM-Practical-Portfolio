
import streamlit as st
import chromadb
import ollama

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Check if the retrieved text can answer the question
def check_relevance(question, context):

    prompt = f"""Check if the context contains useful information to answer the question.

Question:
{question}

Context:
{context}

Reply with only YES or NO."""
    result = ollama.generate(model="llama3.2:3b", prompt=prompt)
    return result["response"].strip().upper().startswith("YES")


def search_documents(collection, question_embedding, number):
    if collection.count() == 0:
        return []

    number = min(number, collection.count())    # chroma complains if you ask for more results than it has stored
    results = collection.query(query_embeddings=[question_embedding], n_results=number)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    found_documents = []

    found_documents = []
    for i, document in enumerate(documents):
        source = "NSBM Guide"
        if metadatas[i] and "source" in metadatas[i]:
            source = metadatas[i]["source"]
        found_documents.append("Source: " + source + "\n" + document)
    return found_documents


# Prepare the retrieved text
def prepare_context(question_embedding, number):
    context = ""
    for document in search_documents(guide_collection, question_embedding, number):
        context += document + "\n\n"
    for document in search_documents(upload_collection, question_embedding, number):
        context += document + "\n\n"
    return context


st.set_page_config(page_title="University AI Assistant", page_icon="🎓")
st.title("🎓 University AI Assistant")
st.caption("Ask questions based on university documents")


# Connect to ChromaDB
client = chromadb.PersistentClient(path="Lab-05-RAG-Pipeline/chroma_db")
guide_collection = client.get_collection(name="nsbm_guide")
upload_collection = client.get_or_create_collection(name="user_uploaded_pdfs")


# Keep the chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []


# PDF upload section
with st.sidebar:
    st.header("University Documents")
    uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("Process PDFs", disabled=not uploaded_files):
        # start fresh so old uploads don't mix with the new ones
        client.delete_collection(name="user_uploaded_pdfs")
        upload_collection = client.create_collection(name="user_uploaded_pdfs")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = []
        embeddings = []
        ids = []
        metadatas = []

        for n, pdf in enumerate(uploaded_files):
            text = ""
            for page in PdfReader(pdf).pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            for i, chunk in enumerate(splitter.split_text(text)):
                response = ollama.embed(model="qwen3-embedding:0.6b", input=chunk)
                documents.append(chunk)
                embeddings.append(response["embeddings"][0])
                # n is there so two files with the same name don't get the same ids
                ids.append(str(n) + "_" + pdf.name + "_" + str(i))
                metadatas.append({"source": pdf.name})

        if documents:
            upload_collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            st.success(f"Processed {len(uploaded_files)} PDFs ({len(documents)} chunks)")
        else:
            st.warning("No readable text found in the PDFs.")

# Show previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# Get the user's question
question = st.chat_input("Ask your university question...")


if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Create an embedding for the question
    response = ollama.embed(model="qwen3-embedding:0.6b", input=question)
    question_embedding = response["embeddings"][0]


    context = prepare_context(question_embedding, 3)
    is_relevant = check_relevance(question, context)


    # if the first 3 chunks don't help, try again with more
    if not is_relevant:
        st.info("Trying to find better information...")
        context = prepare_context(question_embedding, 6)
        is_relevant = check_relevance(question, context)


    # Generate the answer
    with st.chat_message("assistant"):
        with st.spinner("Finding the answer..."):
            if is_relevant and context.strip():
                prompt = f"""Answer the question using only the context.
If the answer is not available in the context, say: "I couldn't find the answer in the document."

Context:
{context}

Question:
{question}

Answer:"""
                answer = ollama.generate(model="llama3.2:3b", prompt=prompt)
                answer_text = answer["response"]
            else:
                answer_text = ("I couldn't find enough relevant information "
                               "in the documents to answer this question.")
        st.write(answer_text)

    st.session_state.messages.append({"role": "assistant", "content": answer_text})