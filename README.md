# Document Chat

A RAG-based document Q&A application that allows users to upload documents and ask questions about their content.

## Overview

Document Q&A Chat Application that uses RAG (Retrieval-Augmented Generation) to answer questions based on uploaded documents. Users can upload various document types (PDF, DOCX, TXT), which are processed to extract text, chunked, and embedded using Gemini embeddings.

The application leverages FastAPI for the backend and Next.js for the frontend, providing a seamless chat interface for users to interact with their documents.

## Quick Start

```bash
# 1. Clone the repository (if needed)
cd document-chat

# 2. Set up backend
cd backend
uv sync
cp .env.example .env
# Add your GEMINI_API_KEY to .env
uvicorn app.main:app --reload

# 3. Set up frontend (in a new terminal)
cd frontend
npm install
npm run dev

# 4. Open http://localhost:3000 in your browser
```

## Features

### Backend
- ✅ **RAG Pipeline**: Complete retrieval-augmented generation workflow
- ✅ **Multi-Format Support**: PDF, DOCX, and TXT documents
- ✅ **Streaming Responses**: Server-Sent Events (SSE) for real-time answers
- ✅ **Source Citations**: Automatic extraction of relevant document chunks
- ✅ **Session Management**: Track document state across requests
- ✅ **Comprehensive Testing**: 199 tests with end-to-end coverage
- ✅ **API Documentation**: Interactive Swagger UI and ReDoc

### Frontend
- ✅ **Modern UI**: Clean, professional interface with Tailwind CSS
- ✅ **Drag-and-Drop Upload**: Easy document upload with validation
- ✅ **Real-time Chat**: Streaming responses with typing indicators
- ✅ **Source Display**: View document excerpts supporting each answer
- ✅ **Responsive Design**: Works on desktop and mobile devices
- ✅ **TypeScript**: Full type safety across the application
- ✅ **Error Handling**: User-friendly error messages

## Tech Stack

- **Backend**: FastAPI with Python 3.12+
- **Frontend**: Next.js 15 with TypeScript & Tailwind CSS
- **Vector Database**: ChromaDB
- **LLM Framework**: LangChain
- **Embeddings & Chat**: Google Gemini API

## Setup

### Prerequisites

- Python 3.12 or higher
- Node.js 18 or higher
- Google Gemini API key

### Backend Installation

1. Navigate to backend and install dependencies:

```bash
cd backend

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

2. Create `.env` file:

```bash
cp .env.example .env
```

3. Add your Google Gemini API key to `.env`:

```
GEMINI_API_KEY=your_actual_api_key_here
```

### Frontend Installation

1. Navigate to frontend and install dependencies:

```bash
cd frontend
npm install
```

2. The `.env.local` file is already configured with the default backend URL:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

If your backend runs on a different port, update this file accordingly.

### Running the Application

You need to run both the backend and frontend servers.

**Backend Server:**

```bash
# From backend directory
cd backend
uvicorn app.main:app --reload

# Or use the convenience script
./scripts/run_dev.sh
```

The backend API will be available at:
- **API**: `http://localhost:8000`
- **Interactive Docs (Swagger UI)**: `http://localhost:8000/docs`
- **Alternative Docs (ReDoc)**: `http://localhost:8000/redoc`

**Frontend Server (in a new terminal):**

```bash
# From frontend directory
cd frontend
npm run dev
```

The frontend will be available at:
- **Web App**: `http://localhost:3000`

## Project Structure

```
document-chat/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app, CORS, lifespan
│   │   ├── config.py               # Settings (env vars)
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes/
│   │   │       ├── __init__.py
│   │   │       ├── documents.py    # POST /documents
│   │   │       ├── query.py        # POST /query (SSE)
│   │   │       └── health.py       # GET /health
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── document_service.py # Process, embed, store
│   │   │   ├── query_service.py    # Retrieve, prompt, stream
│   │   │   ├── embedding_service.py# Gemini embeddings
│   │   │   ├── llm_service.py      # Gemini chat completion
│   │   │   └── vector_store_service.py # ChromaDB operations
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── document_processor.py # LangChain loaders & splitters
│   │   │   ├── session.py          # Global session state
│   │   │   └── exceptions.py       # Custom exceptions
│   │   └── models/
│   │       ├── __init__.py
│   │       └── schemas.py          # Pydantic models
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_*.py               # 199 comprehensive tests
│   │   └── test_e2e.py             # End-to-end integration tests
│   ├── scripts/
│   │   └── run_dev.sh              # Development server script
│   ├── pyproject.toml
│   ├── .env.example
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx          # Root layout with Inter font
│   │   │   ├── page.tsx            # Main chat page
│   │   │   └── globals.css         # Global styles & Tailwind
│   │   ├── components/
│   │   │   ├── ui/
│   │   │   │   ├── Button.tsx      # Reusable button component
│   │   │   │   ├── Card.tsx        # Card components
│   │   │   │   └── Spinner.tsx     # Loading spinner
│   │   │   ├── FileUpload.tsx      # Document upload interface
│   │   │   ├── ChatInterface.tsx   # Main chat UI
│   │   │   ├── MessageBubble.tsx   # Chat message display
│   │   │   └── SourceCard.tsx      # Source citation display
│   │   ├── lib/
│   │   │   ├── api.ts              # API client with SSE support
│   │   │   └── types.ts            # TypeScript types
│   │   └── hooks/
│   │       └── useChat.ts          # Chat state management
│   ├── public/                     # Static assets
│   ├── tailwind.config.ts          # Tailwind configuration
│   ├── tsconfig.json               # TypeScript configuration
│   ├── package.json
│   ├── .env.local                  # Environment variables
│   └── README.md
├── CLAUDE.md
└── README.md
```

## Configuration

See `.env.example` for all available configuration options:

- `GEMINI_API_KEY` - Required. Your Google Gemini API key
- `MAX_PAGE_LIMIT` - Maximum pages per document (default: 50)
- `CHUNK_SIZE_TOKENS` - Chunk size in tokens (default: 500)
- `CHUNK_OVERLAP_TOKENS` - Overlap between chunks (default: 100)
- `TOP_K_CHUNKS` - Number of chunks to retrieve (default: 3)
- `CHROMA_PERSIST_DIR` - ChromaDB storage directory (default: ./chroma_data)
- `ALLOWED_EXTENSIONS` - Allowed file types (default: pdf,txt,docx)
- `MAX_FILE_SIZE_MB` - Max upload size (default: 20)
- `DEBUG` - Enable debug mode with detailed error messages (default: false)

## API Documentation

### API Endpoints

| Endpoint | Method | Description | Status |
|----------|--------|-------------|--------|
| `/health` | GET | Health check and session status | ✅ |
| `/api/v1/documents` | POST | Upload and process a document | ✅ |
| `/api/v1/query` | POST | Query document (streaming SSE) | ✅ |
| `/api/v1/query/sync` | POST | Query document (non-streaming) | ✅ |

### Health Check

Check API health and current session status.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "session_active": true,
  "document_loaded": true,
  "filename": "document.pdf",
  "page_count": 15,
  "chunk_count": 42
}
```

### Upload Document

Upload and process a document for Q&A.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@/path/to/document.pdf"
```

**Response (201 Created):**
```json
{
  "filename": "document.pdf",
  "file_type": "pdf",
  "total_chunks": 42,
  "message": "Document uploaded and processed successfully"
}
```

**Supported File Types:**
- PDF (`.pdf`)
- Microsoft Word (`.docx`)
- Plain Text (`.txt`)

**Error Responses:**
- `400 Bad Request` - Unsupported file type or page limit exceeded
- `413 Request Entity Too Large` - File exceeds size limit
- `422 Unprocessable Entity` - Document processing failed

### Query Document (Streaming)

Query the uploaded document with Server-Sent Events streaming.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"query": "What is the main topic of this document?"}'
```

**Response (SSE Stream):**
```
event: token
data: {"token": "The "}

event: token
data: {"token": "main "}

event: token
data: {"token": "topic "}

event: sources
data: {"sources": [{"content": "...", "page_number": 3, "chunk_index": 7, "relevance_score": 0.89, "filename": "document.pdf"}]}

event: done
data: {"status": "complete"}
```

**Event Types:**
- `token` - Individual tokens as they're generated
- `sources` - Retrieved source chunks with page numbers and scores
- `done` - Completion signal
- `error` - Error information

### Query Document (Non-Streaming)

Synchronous query endpoint for testing.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/query/sync \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "answer": "The main topic of this document is...",
  "sources": [
    {
      "content": "Relevant text from the document...",
      "page_number": 3,
      "chunk_index": 7,
      "relevance_score": 0.89,
      "filename": "document.pdf"
    }
  ]
}
```

**Query Parameters:**
- `query` (required) - The question to ask
- `top_k` (optional) - Number of chunks to retrieve (1-10, default: 3)

### Error Responses

All endpoints use consistent error formatting:

```json
{
  "error": "Error type",
  "detail": "Detailed error message"
}
```

Common error codes:
- `400` - Bad request (invalid input, no document loaded)
- `413` - File too large
- `422` - Validation error or processing failed
- `500` - Internal server error

## Testing

Run the test suite:

```bash
cd backend

# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=app --cov-report=html

# Run specific test file
uv run pytest tests/test_e2e.py -v

# Run end-to-end tests only
uv run pytest tests/test_e2e.py -v --tb=short
```

## Development Status

### ✅ Phase 3: Backend Complete

**Services Implemented:**
- ✅ Embedding Service (Gemini embeddings)
- ✅ Vector Store Service (ChromaDB operations)
- ✅ LLM Service (Gemini chat with streaming)
- ✅ Query Service (RAG pipeline orchestration)
- ✅ Document Service (Upload pipeline)

**API Routes Implemented:**
- ✅ Health check endpoint
- ✅ Document upload endpoint
- ✅ Query endpoint with SSE streaming
- ✅ Synchronous query endpoint

**Backend Features:**
- ✅ Complete RAG pipeline
- ✅ Server-Sent Events streaming
- ✅ Session management
- ✅ Error handling
- ✅ CORS support
- ✅ API documentation (Swagger UI & ReDoc)
- ✅ Comprehensive test coverage (199 tests)
- ✅ End-to-end integration tests

### ✅ Phase 4: Frontend Complete

**Components Implemented:**
- ✅ FileUpload component with drag-and-drop
- ✅ ChatInterface with real-time messaging
- ✅ MessageBubble with source display
- ✅ SourceCard for citation details
- ✅ Reusable UI components (Button, Card, Spinner)

**Frontend Features:**
- ✅ Next.js 15 with App Router
- ✅ TypeScript for type safety
- ✅ Tailwind CSS styling
- ✅ Server-Sent Events integration
- ✅ Custom React hooks (useChat)
- ✅ Responsive design
- ✅ Error handling and validation
- ✅ Production-ready build

### Architecture

The application follows a clean architecture with clear separation of concerns:

1. **API Layer** (`app/api/routes/`) - FastAPI endpoints
2. **Service Layer** (`app/services/`) - Business logic and orchestration
3. **Core Layer** (`app/core/`) - Document processing and session management
4. **Models Layer** (`app/models/`) - Pydantic schemas

**RAG Pipeline Flow:**
```
User Upload → Document Service → Process & Chunk → Embedding Service → Vector Store
                                                                            ↓
User Query → Query Service → Retrieve Chunks → Build Prompt → LLM Service → Stream Response
```

## Example Usage

### Complete Workflow

```bash
# 1. Check API health
curl http://localhost:8000/health

# 2. Upload a document
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@research_paper.pdf"

# 3. Query the document
curl -X POST http://localhost:8000/api/v1/query/sync \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings?"}'

# 4. Check session status
curl http://localhost:8000/health
```

### Using Python

```python
import requests

# Upload document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/documents",
        files={"file": f}
    )
print(response.json())

# Query document
response = requests.post(
    "http://localhost:8000/api/v1/query/sync",
    json={"query": "Summarize the key points"}
)
print(response.json()["answer"])
```

## Troubleshooting

### Common Issues

**1. "GEMINI_API_KEY not set" warning**
- Ensure `.env` file exists in `backend/` directory
- Add your API key: `GEMINI_API_KEY=your_key_here`

**2. Import errors**
- Run `uv sync` or `pip install -e .` from the backend directory
- Ensure Python 3.12+ is being used

**3. Tests failing**
- Check that all dependencies are installed: `uv sync --group dev`
- Ensure mock fixtures are properly configured

**4. Port already in use**
- Change the port: `uvicorn app.main:app --reload --port 8001`

## License

MIT

