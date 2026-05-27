# 📘 Sanskrit RAG System

## 🚀 Overview
This project is a Retrieval-Augmented Generation (RAG) system for Sanskrit Question Answering using FAISS + Transformer-based LLM.

---

## ⚙️ Tech Stack
- Python
- Streamlit
- FAISS
- Sentence Transformers
- HuggingFace Transformers (FLAN-T5)
- PyPDF

---

## 📂 Project Structure

code/ → Implementation files
data/ → Sanskrit PDF dataset
report/ → Project report


---

## 🚀 How to Run

### 1. Install dependencies

pip install -r requirements.txt


### 2. Run ingestion

python code/ingest.py


### 3. Run CLI pipeline

python code/rag_pipeline.py


### 4. Run Streamlit UI

streamlit run code/app.py


---

## 🧠 Features
- PDF-based knowledge ingestion
- FAISS vector search
- Semantic retrieval
- LLM-based answer generation
- Streamlit UI

---

## 📊 Output
- Context retrieval from Sanskrit text
- Generated answers using LLM

---

## 👨‍💻 Author
Divya Ikhar
