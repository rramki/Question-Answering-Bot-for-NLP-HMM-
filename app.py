import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("📄 PDF Question Answering AI")

model = SentenceTransformer("all-MiniLM-L6-v2")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

documents = []
index = None

if uploaded_file:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    documents = text.split("\n")

    embeddings = model.encode(documents)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(np.array(embeddings))

    st.success("PDF processed successfully!")

question = st.text_input("Ask a question about the PDF")

if question and index:

    query_vector = model.encode([question])

    distances, ids = index.search(np.array(query_vector), k=1)

    answer = documents[ids[0][0]]

    st.write("### Answer")
    st.write(answer)
