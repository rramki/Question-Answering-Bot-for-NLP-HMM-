import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.question_answering import load_qa_chain

import os

st.title("📚 AI Assistant for NLP - HMM")

role = st.sidebar.selectbox("Login as", ["User", "Admin"])

embeddings = HuggingFaceEmbeddings()
os.environ["ANTHROPIC_API_KEY"] = "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"
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

            llm = ChatAnthropic(model="claude-3-haiku-20240307",temperature=0)

            #chain = load_qa_chain(llm)

            #answer = chain.run(input_documents=docs, question=question)
            
            chain = create_stuff_documents_chain(llm)
            response = chain.invoke({"input_documents": docs,"question": question})
            answer=response

            st.write("### Answer")
            st.write(answer)

    else:

        st.warning("No subject database found. Admin must upload PDFs.")
