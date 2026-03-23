# 🧠 RAG Application with PostgreSQL (Chat with Your Data)

## 📌 Overview

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to query both **structured data (PostgreSQL tables)** and **unstructured data (documents)** using natural language.

It combines:

* SQL-based data retrieval
* Vector similarity search (embeddings)
* LLM-based response generation

---

## 🚀 Features

* 🔎 Natural language querying over PostgreSQL
* 📊 Structured data access (customers, orders, products)
* 📄 Unstructured document retrieval (support articles)
* 🧠 Embedding-based semantic search
* ⚡ Modular RAG pipeline (easy to extend to AWS / GCP)

---

## 🏗️ Architecture

User Query
→ Embedding Generation
→ Vector Search (PostgreSQL / Chroma)
→ Retrieve Relevant Context
→ LLM Response Generation

---

## 📂 Project Structure

```
rag-postgres-poc/
│
├── app.py                # Main application entry
├── rag_pipeline.py       # Core RAG logic
├── embeddings.py         # Embedding generation
├── db.py                 # PostgreSQL connection & queries
├── config.py             # Configuration settings
├── Test_conn.py          # DB connection testing
├── test_setup.py         # Setup validation
│
├── chroma_db/            # Vector store (ignored in Git)
├── requirements.txt      # Dependencies
├── .gitignore            # Ignore sensitive & temp files
├── .env.example          # Sample environment variables
└── README.md
```

---

## 🛠️ Tech Stack

* **Python**
* **PostgreSQL**
* **pgvector / Chroma (Vector DB)**
* **OpenAI API (Embeddings + LLM)**
* **LangChain (optional integration)**

---

## ⚙️ Setup Instructions

### 1. Clone repository

```bash
git clone https://github.com/njomaha/rag-postgres-p
```
