# RAG Document Q&A System

### Portfolio Chatbot Demo · Built with LangChain · ChromaDB · FastAPI · Groq · Hugging Face

---

A production-ready, document-grounded Q&A system using Retrieval-Augmented Generation (RAG) — demonstrated as a portfolio chatbot but architected as a reusable solution for any document corpus. Users ask natural language questions and receive accurate, context-grounded answers drawn directly from a custom knowledge base.

## How It Works

1. Documents in `knowledge/` are chunked and embedded locally using Hugging Face's `all-MiniLM-L6-v2` via sentence-transformers
2. Embeddings are stored in a local ChromaDB vector store
3. On each question, the top-k semantically similar chunks are retrieved and injected as grounded context into the LLM prompt
4. Groq's Llama 3.3 70B generates a factual, context-grounded answer
5. FastAPI exposes the RAG logic as a REST endpoint
6. A vanilla JS widget embedded in the portfolio HTML calls the API

## Tech Stack

| Layer | Technology |
|---|---|
| Embedding model | Hugging Face all-MiniLM-L6-v2 (runs locally) |
| Vector store | ChromaDB (local) |
| Orchestration | LangChain |
| LLM inference | Groq API — Llama 3.3 70B (free tier) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla JavaScript widget |

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/rskrahil/portfolio-chatbot
cd portfolio-chatbot
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
nano .env
```
Get a free Groq API key at console.groq.com — no credit card needed.

### 5. Add your documents
Replace the files in `knowledge/` with your own `.txt` documents. The system works with any text corpus.

### 6. Build the vector store
```bash
python3 ingest.py
```

### 7. Run the server
```bash
uvicorn server:app --reload --port 8000 --host 0.0.0.0
```

### 8. Test it
Visit `http://localhost:8000/docs` for the interactive API docs.

Or via curl:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are your skills?"}'
```

## API Reference

### GET /
Health check — confirms server is running.

### POST /chat
Ask a question, get a grounded answer.

**Request:**
```json
{ "question": "What projects have you built?" }
```

**Response:**
```json
{ "answer": "..." }
```

## Customization

Swap the knowledge base for any domain:
1. Replace `.txt` files in `knowledge/` with your documents
2. Update the system prompt in `chat.py` with your context
3. Re-run `python3 ingest.py` to rebuild the vector store

Works for: internal knowledge bases, customer support bots, HR policy assistants, legal document Q&A, product documentation, and more.

