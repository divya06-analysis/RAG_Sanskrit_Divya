import re
import faiss
import pickle
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# -----------------------------------
# Streamlit Page
# -----------------------------------
st.set_page_config(
    page_title="Sanskrit RAG System",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Sanskrit RAG System")
st.write("Ask questions from Sanskrit documents")

# -----------------------------------
# Text Cleaning Function
# -----------------------------------
def clean_text(text):

    text = text.replace("\n", " ")

    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"[^ऀ-ॿa-zA-Z0-9\s।॥]",
        "",
        text
    )

    return text.strip()

# -----------------------------------
# Load Embedding Model
# -----------------------------------
@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return model

# -----------------------------------
# Load FAISS + Chunks
# -----------------------------------
@st.cache_resource
def load_data():

    index = faiss.read_index("sanskrit_index.faiss")

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks

# -----------------------------------
# Load Resources
# -----------------------------------
with st.spinner("Loading Sanskrit RAG System..."):

    embedding_model = load_model()

    index, chunks = load_data()

st.success("System Ready")

# -----------------------------------
# User Query
# -----------------------------------
query = st.text_input(
    "Enter Sanskrit Query"
)

# -----------------------------------
# Search Button
# -----------------------------------
if st.button("Search"):

    if query.strip() == "":

        st.warning("Please enter a query")

    else:

        # -----------------------------------
        # Clean Query
        # -----------------------------------
        query = clean_text(query)

        # -----------------------------------
        # Query Embedding
        # -----------------------------------
        query_embedding = embedding_model.encode([query])

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        # -----------------------------------
        # FAISS Search
        # -----------------------------------
        distances, indices = index.search(
            query_embedding,
            5
        )

        retrieved_sentences = []

        # -----------------------------------
        # Extract Relevant Sentences
        # -----------------------------------
        for idx in indices[0]:

            if idx < len(chunks):

                chunk = clean_text(chunks[idx])

                sentences = chunk.split("।")

                for sentence in sentences:

                    sentence = sentence.strip()

                    if len(sentence) < 10:
                        continue

                    score = 0

                    for word in query.split():

                        if word in sentence:
                            score += 1

                    if score > 0:

                        retrieved_sentences.append(
                            (score, sentence)
                        )

        # -----------------------------------
        # Sort by relevance
        # -----------------------------------
        retrieved_sentences.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        # -----------------------------------
        # Top Retrieved Context
        # -----------------------------------
        top_sentences = []

        for item in retrieved_sentences[:2]:

            top_sentences.append(
                item[1] + "।"
            )

        context = "\n".join(top_sentences)

        # -----------------------------------
        # Automatic Answer Extraction
        # -----------------------------------
        answer = "Answer not found in document."

        best_sentence = ""
        best_score = 0

        for item in retrieved_sentences:

            score = item[0]
            sentence = item[1]

            if score > best_score:

                best_score = score
                best_sentence = sentence

        # -----------------------------------
        # Final Answer
        # -----------------------------------
        if best_sentence != "":

            answer = best_sentence + "।"

        # -----------------------------------
        # Display Results
        # -----------------------------------
        st.subheader("Retrieved Context")

        st.write(context)

        st.subheader("Generated Answer")

        st.success(answer)
