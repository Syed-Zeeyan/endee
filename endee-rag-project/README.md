# Endee RAG Knowledge Assistant

**Author:** Syed Zeeyan  
**Project Type:** Retrieval Augmented Generation (RAG) using Endee Vector Database  
**Purpose:** Endee Labs Internship Evaluation

## 🔍 Project Overview

This project demonstrates a **complete Retrieval Augmented Generation (RAG) pipeline** built using the **Endee vector database** as the core semantic search engine.

The system ingests documents, converts them into embeddings, stores them in Endee, and retrieves relevant context for user queries using vector similarity search.

This implementation is designed to be **fully local**, **cost-free**, and **production-structured**, showcasing practical understanding of vector databases and RAG architecture.

## 🚀 Problem Statement

Traditional search systems rely on keyword matching and often fail to understand semantic meaning.

**Modern AI systems require:**
- Semantic search over knowledge bases
- Context retrieval for intelligent responses
- Fast vector similarity search
- Local/private data processing

This project solves these challenges by implementing a complete RAG pipeline using **Endee**.

## ✨ Key Features

- ✅ Uses **Endee** as the core vector database
- ✅ Implements **complete RAG pipeline**
- ✅ Fully local (**no paid APIs required**)
- ✅ Document ingestion + semantic search
- ✅ Clean modular Python backend
- ✅ Docker-based Endee deployment
- ✅ GitHub-ready project structure

## 🏗️ System Architecture

```mermaid
graph TD
    User([User Query]) -->|Embedding| QueryEmb[Query Embedding (MiniLM)]
    QueryEmb -->|Search| Endee[(Endee Vector Database)]
    Endee -->|Semantic Similarity| TopK[Top-K Relevant Chunks]
    TopK -->|Use Context| Response[Context-based Response]
```

## 🛠️ How Endee is Used

Endee is the core engine of this project.

### 1. Index Creation
Creates a vector index for storing embeddings.

```http
POST /api/v1/index/create
{
  "index_name": "knowledge_base",
  "dim": 384,
  "space_type": "cosine"
}
```

### 2. Vector Storage
Document chunks are converted into embeddings and stored.

```http
POST /api/v1/index/{index_name}/vector/insert
```

Each vector stores:
- **embedding** (vector array)
- **meta** including text source 

### 3. Semantic Search
User query → embedding → vector search:

```http
POST /api/v1/index/{index_name}/search
{
  "vector": [...],
  "k": 3
}
```
Returns most relevant document chunks.

## 💻 Tech Stack

| Component | Technology |
|---|---|
| **Vector Database** | Endee |
| **Backend** | Python |
| **Embeddings** | sentence-transformers |
| **Model** | all-MiniLM-L6-v2 |
| **Container** | Docker |
| **Interface** | CLI |

## 📂 Project Structure

```
endee-rag-app/
│
├── src/
│   ├── ingestion/          # Document loading & chunking
│   ├── embeddings/         # Embedding generation
│   ├── retrieval/          # Query engine (RAG)
│   ├── endee/              # Endee API client
│   └── main.py             # CLI entry point
│
├── data/documents/         # Knowledge base files
├── docker-compose.yml      # Endee setup (if applicable)
├── requirements.txt
└── README.md
```

## ⚡ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/endee-rag-app.git
cd endee-rag-app
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3. Start Endee
Ensure you have Endee running (e.g., via Docker):
```bash
docker-compose up -d
```
Verify it's running:
```bash
curl http://localhost:8080/health
```

## ▶️ Running the Project

### Ingest Documents
Place your files inside `data/documents/`:
```bash
python -m src.main ingest data/documents
```

### Query System
Ask semantic questions about your documents:
```bash
python -m src.main query "What is artificial intelligence?"
```

> **Note:** Responses are generated from retrieved document context. Quality depends on documents ingested into the system.

## 🌟 Why This Project Matters

This project demonstrates:
- Practical use of **vector databases**
- Real-world **RAG implementation**
- **API integration** and debugging
- **System design** understanding
- Clean **modular backend** development

It shows the ability to build production-style AI systems using open-source tools.

## 🔮 Future Improvements

- [ ] Web UI (FastAPI + React)
- [ ] Hybrid search support
- [ ] Streaming responses
- [ ] Multi-document ranking
- [ ] Local LLM integration

## 👨‍💻 Author

**Syed Zeeyan**  
Backend & AI Engineering Candidate

This project was built as part of the **Endee Labs internship evaluation** to demonstrate real-world vector database and RAG system implementation.
