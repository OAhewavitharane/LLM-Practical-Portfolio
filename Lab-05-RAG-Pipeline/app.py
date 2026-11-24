
import streamlit as st
import chromadb
import ollama

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Check if the retrieved text can answer the question
def check_relevance(question, context):

    prompt = f"""
Check if the context contains useful information
to answer the question.

Question:
{question}

Context:
{context}

Reply with only YES or NO.
"""

    result = ollama.generate(
        model="llama3.2:3b",
        prompt=prompt
    )

    answer = result["response"].strip().upper()

    return answer.startswith("YES")


# Search a collection for relevant text
def search_documents(collection, question_embedding, number):

    if collection.count() == 0:
        return []

    number = min(number, collection.count())

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    found_documents = []

    for i, document in enumerate(documents):

        source = "NSBM Guide"

        if metadatas[i] and "source" in metadatas[i]:
            source = metadatas[i]["source"]

        found_documents.append(
            "Source: " + source + "\n" + document
        )

    return found_documents


# Prepare the retrieved text
def prepare_context(question_embedding, number):

    context = ""

    guide_documents = search_documents(
        guide_collection,
        question_embedding,
        number
    )

    for document in guide_documents:
        context += document + "\n\n"

    uploaded_documents = search_documents(
        upload_collection,
        question_embedding,
        number
    )

    for document in uploaded_documents:
        context += document + "\n\n"

    return context


# Set up the page
st.set_page_config(
    page_title="University AI Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 University AI Assistant")
st.caption("Ask questions based on university documents")


# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="Lab-05-RAG-Pipeline/chroma_db"
)

guide_collection = client.get_collection(
    name="nsbm_guide"
)

upload_collection = client.get_or_create_collection(
    name="user_uploaded_pdfs"
)


# Keep the chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []


# PDF upload section
with st.sidebar:

    st.header("📄 University Documents")

    uploaded_files = st.file_uploader(
        "Upload your PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process PDFs", disabled=not uploaded_files):

        # Clear the old uploaded PDFs
        client.delete_collection(
            name="user_uploaded_pdfs"
        )

        upload_collection = client.create_collection(
            name="user_uploaded_pdfs"
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        documents = []
        embeddings = []
        ids = []
        metadatas = []

        for pdf in uploaded_files:

            reader = PdfReader(pdf)

            text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            chunks = splitter.split_text(text)

            for i, chunk in enumerate(chunks):

                response = ollama.embed(
                    model="qwen3-embedding:0.6b",
                    input=chunk
                )

                documents.append(chunk)
                embeddings.append(
                    response["embeddings"][0]
                )

                ids.append(
                    pdf.name + "_" + str(i)
                )

                metadatas.append(
                    {"source": pdf.name}
                )

        if documents:

            upload_collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

            st.success("PDFs processed successfully!")

            st.write("Uploaded PDFs:", len(uploaded_files))
            st.write("Total chunks:", len(documents))

        else:
            st.warning("No readable text found in the PDFs.")


# Show previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# Get the user's question
question = st.chat_input(
    "Ask your university question..."
)


if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)


    # Create an embedding for the question
    response = ollama.embed(
        model="qwen3-embedding:0.6b",
        input=question
    )

    question_embedding = response["embeddings"][0]


    # First, retrieve 3 chunks
    context = prepare_context(
        question_embedding,
        3
    )

    is_relevant = check_relevance(
        question,
        context
    )


    # If the results are not relevant, try again
    if not is_relevant:

        st.info("Trying to find better information...")

        context = prepare_context(
            question_embedding,
            6
        )

        is_relevant = check_relevance(
            question,
            context
        )


    # Generate the answer
    with st.chat_message("assistant"):

        with st.spinner("Finding the answer..."):

            if is_relevant and context.strip():

                prompt = f"""
Answer the question using only the context.

If the answer is not available in the context,
say:
"I couldn't find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

                answer = ollama.generate(
                    model="llama3.2:3b",
                    prompt=prompt
                )

                answer_text = answer["response"]

            else:

                answer_text = (
                    "I couldn't find enough relevant information "
                    "in the documents to answer this question."
                )

        st.write(answer_text)


    # Save the answer
    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text}
    )
