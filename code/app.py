import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import pickle
import numpy as np

# ---------------------------
# LOAD MODELS
# ---------------------------
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    index = faiss.read_index("sanskrit_index.faiss")

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    return embedding_model, index, chunks, tokenizer, model


embedding_model, index, chunks, tokenizer, model = load_models()

# ---------------------------
# UI
# ---------------------------
st.title("📘 Sanskrit RAG System")

query = st.text_input("Enter Sanskrit Query:")

# ---------------------------
# RAG PIPELINE
# ---------------------------
if st.button("Get Answer"):

    if not query.strip():
        st.warning("Please enter a query")
        st.stop()

    # -----------------------
    # RETRIEVAL (IMPROVED)
    # -----------------------
    query_vec = embedding_model.encode([query]).astype("float32")

    scores, indices = index.search(query_vec, k=3)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n".join(retrieved_chunks)

    # -----------------------
    # DISPLAY CONTEXT
    # -----------------------
    st.subheader("Retrieved Context")
    st.write(context)

    # -----------------------
    # STRICT PROMPT (NO HALLUCINATION)
    # -----------------------
    prompt = f"""
You are a strict Sanskrit question-answering assistant.

RULES:
- Use ONLY the given context
- Do NOT use outside knowledge
- If answer is not present, say "उत्तरः न उपलब्धः"

CONTEXT:
{context}

QUESTION:
{query}

FINAL ANSWER in Sanskrit:
"""

    # -----------------------
    # TOKENIZE
    # -----------------------
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # -----------------------
    # GENERATE ANSWER
    # -----------------------
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        num_beams=6,
        do_sample=False,
        temperature=0.3
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # -----------------------
    # SAFETY FALLBACK
    # -----------------------
    if not answer or len(answer) < 2:
        answer = "उत्तरः न उपलब्धः (No answer found in context)"

    # -----------------------
    # OUTPUT
    # -----------------------
    st.subheader("Generated Answer")
    st.success(answer)