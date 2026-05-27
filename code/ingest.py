from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pypdf import PdfReader
import os


# -----------------------------
# Load PDF safely
# -----------------------------
def load_pdf(path):
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Main pipeline
# -----------------------------
def main():

    file_path = r"data\Rag-docs.pdf"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load file
    text = load_pdf(file_path)
    print("Text Loaded Successfully")

    # Clean chunks
    chunks = [t.strip() for t in text.split("\n") if t.strip()]
    print("Chunks Created:", len(chunks))

    if len(chunks) == 0:
        raise ValueError("No text extracted from PDF!")

    # Load embedding model
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    print("Embedding Model Loaded")

    # Create embeddings
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    # FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print("FAISS Index Created")

    # Save index
    faiss.write_index(index, "sanskrit_index.faiss")

    # Save chunks
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("All Files Saved Successfully")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    main()
