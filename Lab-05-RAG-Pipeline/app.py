
import streamlit as st
import chromadb
import ollama

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


# Store chat messages
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

        # Remove old uploaded documents
        try:
            client.delete_collection(
                name="user_uploaded_pdfs"
            )
        except Exception:
            pass

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

            st.write("Documents uploaded:", len(uploaded_files))
            st.write("Total chunks:", len(documents))

        else:
            st.warning("No readable text found in the PDFs.")


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# Chat input
question = st.chat_input(
    "Ask your university question..."
)


if question:

    # Display the user's question
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


    # Search the NSBM guide
    guide_results = guide_collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )


    # Search uploaded PDFs
    uploaded_results = []

    if upload_collection.count() > 0:

        uploaded_results = upload_collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )


    # Prepare the context
    context = ""

    for i, document in enumerate(
        guide_results["documents"][0]
    ):
        context += "\nSource: NSBM Guide\n"
        context += document + "\n"


    if uploaded_results:

        for i, document in enumerate(
            uploaded_results["documents"][0]
        ):

            source = uploaded_results["metadatas"][0][i]["source"]

            context += "\nSource: " + source + "\n"
            context += document + "\n"


    # Create the prompt
    prompt = f"""
Answer the question using only the context.

If the answer is not available in the context,
say that you couldn't find it in the documents.

Context:
{context}

Question:
{question}

Answer:
"""


    # Generate the answer
    with st.chat_message("assistant"):

        with st.spinner("Finding the answer..."):

            answer = ollama.generate(
                model="llama3.2:3b",
                prompt=prompt
            )

            answer_text = answer["response"]

        st.write(answer_text)


    # Save the answer in chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text}
    )
