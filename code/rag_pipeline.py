from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import pickle

# Load embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Embedding Model Loaded")

# Load FAISS + chunks
index = faiss.read_index("sanskrit_index.faiss")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("FAISS + Chunks Loaded")

# Load LLM
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

print("LLM Loaded")

print("\n===== Sanskrit RAG Ready =====")

while True:

    query = input("\nEnter Sanskrit Query: ")

    # Exit condition
    if query.lower() == "exit":
        break

    # Retrieval
    query_vec = embedding_model.encode([query]).astype("float32")

    _, indices = index.search(query_vec, k=1)

    context = chunks[indices[0][0]]

    print("\nRetrieved Context:\n", context)

    # Prompt
    prompt = f"Question: {query}\nContext: {context}\nAnswer in Sanskrit:"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        num_beams=5,
        do_sample=True,
        temperature=0.3
    )

    # Decode
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nGenerated Answer:\n", answer)