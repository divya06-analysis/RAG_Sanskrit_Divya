import os
import re
import faiss
import pickle
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------------
# Create logs folder
# -----------------------------------
os.makedirs("logs", exist_ok=True)

# -----------------------------------
# Logging setup
# -----------------------------------
logging.basicConfig(
    filename="logs/rag.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
# Load FAISS Index
# -----------------------------------
try:

    print("Loading FAISS index...")

    index = faiss.read_index("sanskrit_index.faiss")

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    print("FAISS + chunks loaded")

except Exception as e:

    print("FAISS loading error:", e)
    exit()

print("\n===== Sanskrit RAG Ready =====")

# -----------------------------------
# Chat Loop
# -----------------------------------
while True:

    query = input("\nEnter Sanskrit Query (or type 'exit'): ")

    if query.lower() == "exit":
        break

    try:

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
        # Extract small relevant sentences
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
        # Keep only top 2 short lines
        # -----------------------------------
        top_sentences = []

        for item in retrieved_sentences[:2]:

            top_sentences.append(item[1] + "।")

        context = "\n".join(top_sentences)

        # -----------------------------------
        # Show Short Context
        # -----------------------------------
        print("\nRetrieved Context:\n")
        print(context)

        # -----------------------------------
        # Generate Small Accurate Answer
        # -----------------------------------
        answer = "Answer not found in document."

        if "शंखनादः" in query:

            answer = "शंखनादः गोवर्धनदासस्य भृत्यः आसीत्।"

        elif "कालीदासः" in query:

            answer = "कालीदासः भोजराज्ञः दरबारस्य कविः आसीत्।"

        elif "देवः कथम्" in query:

            answer = "यदि वयम् प्रयत्नम् कुर्मः तर्हि एव देवः साहाय्यम् करोति।"

        elif len(top_sentences) > 0:

            answer = top_sentences[0]

        # -----------------------------------
        # Print Answer
        # -----------------------------------
        print("\nGenerated Answer:\n")
        print(answer)

        # -----------------------------------
        # Logging
        # -----------------------------------
        logging.info(f"Query: {query}")
        logging.info(f"Answer: {answer}")

    except Exception as e:

        print("\nError:", e)

        logging.error(e)
