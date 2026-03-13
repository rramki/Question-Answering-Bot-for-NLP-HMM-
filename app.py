import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import anthropic

st.title("📚 AI Tutor for NLP / HMM")

model = SentenceTransformer("all-MiniLM-L6-v2")
role = st.sidebar.selectbox("Login as", ["User", "Admin"])

documents = []
index = None

# -------------------
# ADMIN PANEL
# -------------------

if role == "Admin":

    st.header("Admin: Upload Subject PDF")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

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

        if not os.path.exists("vectorstore"):
            os.makedirs("vectorstore")

        faiss.write_index(index, "vectorstore/index.faiss")
        np.save("vectorstore/docs.npy", documents)
        st.success("Vector database created!")

# -------------------
# USER PANEL
# -------------------

if role == "User":

    st.header("Ask Questions")

    if os.path.exists("vectorstore/index.faiss"):

        index = faiss.read_index("vectorstore/index.faiss")

        documents = np.load("vectorstore/docs.npy", allow_pickle=True)

        question = st.text_input("Enter your question")

        if question:
            query_vector = model.encode([question])
            distances, ids = index.search(np.array(query_vector), k=3)

            context = ""

            for i in ids[0]:
                context += documents[i] + "\n"
            client = anthropic.Anthropic(
                api_key=st.secrets["ANTHROPIC_API_KEY"]
            )

            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": f"Answer based on context:\n{context}\nQuestion:{question}"
                    }
                ]
            )
            st.write("### Answer")
            st.write(answer)
    else:

        st.warning("Admin must upload PDF first.")
