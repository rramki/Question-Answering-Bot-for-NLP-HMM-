import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import anthropic

st.title("📚 AI Tutor for NLP / HMM")

model = SentenceTransformer("all-MiniLM-L6-v2")
def split_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        
    return chunks


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

        documents = split_text(text)   

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
    distances, ids = index.search(np.array(query_vector), k=2)

    context = ""

    for i in ids[0][:2]:
        context += documents[i][:700] + "\n"

    context = context[:1500]

    answer = None

    try:
        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:{question}"
            }]
        )

        answer = response.content[0].text

    except Exception:
        st.error("Claude API error. Try asking a shorter question.")

    if answer:
        st.write("### Answer")
        st.write(answer)
    else:

        st.warning("Admin must upload PDF first.")
