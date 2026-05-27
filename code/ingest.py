from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load file
with open(r"data\knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Text Loaded Successfully")

# Split into lines (chunks)
chunks = [t.strip() for t in text.split("\n") if t.strip()]

print("Chunks Created:", len(chunks))

# Embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

print("Embedding Model Loaded")

# Create embeddings
embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

# FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print("FAISS Index Created")

# Save index + chunks
faiss.write_index(index, "sanskrit_index.faiss")

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("All Files Saved Successfully")