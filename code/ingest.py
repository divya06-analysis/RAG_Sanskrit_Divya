from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import fitz
import re
import logging
import os

# -----------------------------------
# Create logs folder
# -----------------------------------
os.makedirs("logs", exist_ok=True)

# -----------------------------------
# Logging
# -----------------------------------
logging.basicConfig(
    filename="logs/ingest.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------------
# Load Embedding Model
# -----------------------------------
try:

    print("Loading embedding model...")

    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("Embedding model loaded")

except Exception as e:

    print("Embedding model error:", e)
    exit()

# -----------------------------------
# PDF Path
# -----------------------------------
pdf_path = "data/Rag-docs.pdf"

# -----------------------------------
# Read PDF
# -----------------------------------
try:

    print("Reading PDF...")

    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:

        page_text = page.get_text()

        text += page_text + "\n"

    print("PDF text extracted")

except Exception as e:

    print("PDF reading error:", e)
    exit()

# -----------------------------------
# Sanskrit Cleaning Function
# -----------------------------------
def clean_sanskrit_text(text):

    # remove English translations
    text = re.sub(r"\(.*?\)", " ", text)

    # remove emails
    text = re.sub(r"\S+@\S+", " ", text)

    # remove latin text
    text = re.sub(r"[A-Za-z]+", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text)

    # OCR cleanup
    replacements = {

        "र्ो": "को",
        "र्रो": "करो",
        "र्ृ": "कृ",
        "तर्ं": "किं",
        "बार्": "बाध",
        "र्ायक": "कार्य",
        "भर्": "भव",
        "र्क": "क",
        "र्ु": "कु",
        "र्ि": "यदि",
        "िदा": "तदा",
        "ििः": "ततः",
        "िस्य": "तस्य",
        "िेन": "तेन",
        "अिः": "अतः",
        "चिुर": "चतुर",
        "र्ालीदास": "कालीदास",
        "शर्करा": "शर्करा",
    }

    for wrong, correct in replacements.items():

        text = text.replace(wrong, correct)

    # keep Sanskrit chars only
    text = re.sub(
        r"[^ऀ-ॿ\s।॥]",
        " ",
        text
    )

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# -----------------------------------
# Clean Text
# -----------------------------------
print("Cleaning Sanskrit text...")

text = clean_sanskrit_text(text)

print("Text cleaned")

# -----------------------------------
# Save cleaned text
# -----------------------------------
with open("cleaned_text.txt", "w", encoding="utf-8") as f:

    f.write(text)

print("Cleaned text saved")

# -----------------------------------
# Chunking
# -----------------------------------
print("Creating chunks...")

sentences = text.split("।")

chunks = []

chunk = ""

for sentence in sentences:

    sentence = sentence.strip()

    if len(sentence) < 10:
        continue

    if len(chunk) + len(sentence) < 350:

        chunk += sentence + "। "

    else:

        chunks.append(chunk.strip())

        chunk = sentence + "। "

if chunk:
    chunks.append(chunk.strip())

print(f"Total chunks: {len(chunks)}")

# -----------------------------------
# Generate Embeddings
# -----------------------------------
print("Generating embeddings...")

embeddings = embedding_model.encode(
    chunks,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

print("Embeddings generated")

# -----------------------------------
# Create FAISS Index
# -----------------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS index created")

# -----------------------------------
# Save Index
# -----------------------------------
faiss.write_index(index, "sanskrit_index.faiss")

# -----------------------------------
# Save Chunks
# -----------------------------------
with open("chunks.pkl", "wb") as f:

    pickle.dump(chunks, f)

print("Chunks saved")

print("\n===== INGESTION COMPLETED =====")
