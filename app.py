import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import Anthropic
from langchain.chains.question_answering import load_qa_chain
import os

st.title("📚 AI Assistant for NLP - HMM")

role = st.sidebar.selectbox("Login as", ["User", "Admin"])

embeddings = HuggingFaceEmbeddings()

# ADMIN PANEL
if role == "Admin":

    st.header("Admin Panel - Upload Subject PDFs")

    uploaded_file = st.file_uploader("Upload Subject PDF", type="pdf")

    if uploaded_file:

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            text += page.extract_text()

        splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        texts = splitter.split_text(text)

        db = FAISS.from_texts(texts, embeddings)

        db.save_local("vectorstore")

        st.success("Vector database created successfully!")

# USER PANEL
if role == "User":

    st.header("Ask Questions")

    if os.path.exists("vectorstore"):

        db = FAISS.load_local("vectorstore", embeddings)

        question = st.text_input("Enter your question")

        if question:

            docs = db.similarity_search(question)

            llm = Anthropic()

            chain = load_qa_chain(llm)

            answer = chain.run(input_documents=docs, question=question)

            st.write("### Answer")
            st.write(answer)

    else:

        st.warning("No subject database found. Admin must upload PDFs.")
