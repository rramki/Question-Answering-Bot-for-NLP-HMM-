import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import anthropic

st.title("📚 AI Tutor for NLP")
ADMIN_USER = "admin"
ADMIN_PASS = "nlp123"
model = SentenceTransformer("all-MiniLM-L6-v2")

#Sidebar for Admin
role = st.sidebar.selectbox("Login as", ["User", "Admin"])

if role == "Admin":

    username = st.sidebar.text_input("Admin Username")
    password = st.sidebar.text_input("Admin Password", type="password")

    if username == ADMIN_USER and password == ADMIN_PASS:
        st.success("Admin logged in")
        admin_logged = True
    else:
        admin_logged = False

    if admin_logged:
        subject = st.selectbox(
        "Select Subject",
        ["nlp", "ml", "cloud"]
    )

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    #Uploading  the File

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

        folder = f"/vectorstore/{subject}"

        os.makedirs(folder, exist_ok=True)

        faiss.write_index(index, f"{folder}/index.faiss")

        np.save(f"{folder}/docs.npy", documents)

        st.success(f"{subject} vector database created!")


#User Interface Page
st.header("Ask Questions")

subject = st.selectbox(
    "Select Subject",
    ["NLP", "MachineLearning", "CloudComputing"]
)

#Load the vector database:
folder = f"vectorstore/{subject}"

if os.path.exists(f"{folder}/index.faiss"):

    index = faiss.read_index(f"{folder}/index.faiss")

    documents = np.load(f"{folder}/docs.npy", allow_pickle=True)


question = st.text_input("Enter your question")

if question:

    query_vector = model.encode([question])

    distances, ids = index.search(np.array(query_vector), k=2)

    context = ""

    for i in ids[0][:2]:
        context += documents[i][:700] + "\n"

