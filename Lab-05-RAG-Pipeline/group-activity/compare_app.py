import time
import streamlit as st
from rag import retrieve, answer

st.set_page_config(page_title="Group 2 - Chunking comparison", layout="wide")
st.title("Context-aware RAG: chunking comparison")
st.caption("One question, three ways of chunking the NSBM MSc in IT guide")

methods = {
    "fixed": "Fixed-size (baseline)",
    "section": "Section-based",
    "parent_child": "Parent-child",
}

question = st.text_input("Ask a question about the guide")

if st.button("Compare", disabled=not question):
    for column, method in zip(st.columns(3), methods):
        with column:
            st.subheader(methods[method])
            with st.spinner("Finding the answer..."):
                start = time.time()
                context = retrieve(method, question)
                reply = answer(question, context)
                seconds = time.time() - start
            st.write(reply)
            st.caption(f"{seconds:.1f} s, {len(context)} characters of context")
            # handy in the demo to show why one method got it right and another didn't
            with st.expander("What the model was given"):
                st.text(context)
