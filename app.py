import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq 

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢"
)

st.title("🏢 Zyro Dynamics HR Help Desk")

@st.cache_resource
def load_rag():

    docs = []

    pdf_folder = "pdfs"

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(
                os.path.join(pdf_folder, file)
            )
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k":4}
    )

    llm = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    return retriever, llm


retriever, llm = load_rag()

question = st.chat_input(
    "Ask an HR question..."
)

if question:

    st.chat_message("user").write(question)

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are the Zyro Dynamics HR Assistant.

Answer ONLY from the provided context.

If the answer is not available in the context,
respond exactly with:

I can only answer HR-related questions based on Zyro Dynamics policy documents.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    st.chat_message("assistant").write(
        response.content
    )

    with st.expander("Sources"):

        for doc in docs:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            st.write(source)
