# Yoga Wellness RAG Application

A comprehensive Retrieval-Augmented Generation (RAG) application acting as an intelligent Yoga expert. It provides safe, accurate, and context-aware advice by combining a curated yoga knowledge base with Large Language Models (LLMs), featuring a robust safety layer and detailed interaction logging.

## 📂 Project Structure

The repository is organized into the following key directories:

-   **src/**: The core application logic, API, and RAG services.
    -   `api/`: FastAPI application and routes.
    -   `services/`: Modular services for Chunking, Embeddings, Retrieval, Generation, and Safety.
    -   `models/`: Pydantic data schemas.
    -   `core/`: Configuration, logging, caching, and rate limiting.
    -   `frontend/`: Lightweight Vanilla JavaScript Single Page Application (SPA).
-   **scripts/**: Utility scripts for data ingestion and system verification.
-   **tests/**: Comprehensive unit and property-based tests.

## 🏗️ Architecture

The application follows a modular microservices-based architecture (monolith code structure):

1.  **Frontend**: Uses standard web technologies (HTML5, CSS3, ES6 JavaScript) to communicate with the backend via REST APIs.
2.  **Backend API**: Built with **FastAPI**, handling request validation, CORS, rate limiting, and routing to appropriate services.
3.  **RAG Pipeline**:
    -   **Ingestion**: `scripts/process_yoga_knowledge_base.py` parses Markdown files.
    -   **Chunking**: Splits text into semantic chunks with overlap (256-512 tokens).
    -   **Embedding**: Converts text to vectors using **Sentence Transformers** or **NVIDIA NIM API**.
    -   **Vector Database**: Stores embeddings in **ChromaDB** (local) or **Pinecone** (cloud) for semantic search.
    -   **Generation**: Uses **OpenAI GPT-4** (or compatible) to generate answers based on retrieved context.
4.  **Safety Layer**: Intercepts queries to detect sensitive topics (Medical, Emergency, Pregnancy) before processing.
5.  **Logging**: Stores all interactions and safety flags in **MongoDB** for analysis.
6.  **Performance**: Redis caching and rate limiting for production scalability.

## 🚀 Setup & Installation

### 1. Prerequisites

-   Python 3.10+
-   MongoDB (running locally on port 27017 or remote)
-   Redis (optional, for caching and rate limiting)
-   git

### 2. Environment Variables

Create a `.env` file in the root directory (copy from `.env.example`):

```bash
# Quick setup
make setup-env  # This copies .env.example to .env
# Then edit .env with your actual API keys
```

**Required Environment Variables:**
```env
# Application
APP_NAME="Yoga Wellness RAG"
DEBUG=false
LOG_LEVEL=INFO

# API
API_HOST=0.0.0.0
API_PORT=8000

# Database
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=wellness_rag

# Vector Database
CHROMA_PERSIST_DIRECTORY=./data/chroma
CHROMA_COLLECTION_NAME=wellness_chunks
# OR for Pinecone:
# PINECONE_API_KEY=your_key_here
# PINECONE_ENVIRONMENT=your_env

# AI Services
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4-turbo-preview

# NVIDIA (Optional)
# NVIDIA_EMBEDDING_API_KEY=your_nvidia_key
# NVIDIA_LLM_API_KEY=your_nvidia_llm_key

# Performance (Optional)
REDIS_URL=redis://localhost:6379
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

**⚠️ Important**: Never commit your actual `.env` file to GitHub. Only commit `.env.example`.

### 3. Installation
```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Running Locally

Start the backend server:

```bash
python -m src.api.main
```

OR using uvicorn directly:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Access the application at: **http://localhost:8000**

### 5. Ingesting Data (RAG Setup)
Before asking questions, populate the vector database:
```bash
python scripts/process_yoga_knowledge_base.py
```
This script reads `yoga_knowledge_base.md`, chunks the content, embeds it, and stores it in the vector database.

## 🧠 RAG Pipeline Explanation

The Retrieval-Augmented Generation pipeline follows these steps:

1.  **Query Analysis**: The user's question is first passed to the **Safety Service**.
2.  **Safety Check**: If the query contains emergency keywords, it is blocked immediately. If it touches on medical/pregnancy topics, specific warnings are prepared.
3.  **Embedding**: Valid queries are converted into a dense vector representation.
4.  **Retrieval**: The system queries the Vector DB for the top K most similar chunks from the `yoga_knowledge_base.md`.
5.  **Context Construction**: Retrieved chunks are formatted into a context block.
6.  **Prompt Engineering**: A system prompt combines the "Yoga Expert" persona, the user query, and the retrieved context.
7.  **Generation**: The LLM generates a response citing the context where appropriate.

## 🏛️ Architecture Decisions

### **RAG Pipeline Choices**

**Why Semantic Chunking (256-512 tokens)?**
- Balances context preservation with embedding quality
- Prevents token limit issues with NVIDIA API (512 max)
- Maintains semantic coherence within chunks
- Allows for meaningful overlap between chunks

**Why Multiple Embedding Providers?**
- **Sentence Transformers**: Fast, local, no API costs (384 dimensions)
- **NVIDIA NIM**: Higher quality embeddings for production (1024 dimensions)
- Factory pattern allows easy switching based on requirements

**Why ChromaDB + Pinecone Support?**
- **ChromaDB**: Perfect for local development and testing
- **Pinecone**: Production-ready with better performance and persistence
- Abstraction layer allows seamless switching

**Why OpenAI GPT-4 for Generation?**
- Superior reasoning and context understanding
- Excellent at following safety instructions
- Reliable source attribution and citation
- Consistent response quality

### **Safety System Choices**

**Why Deterministic Safety Filter?**
- **Predictable**: Same input always produces same safety assessment
- **Transparent**: Easy to audit and understand decisions
- **Fast**: No additional API calls or ML inference
- **Reliable**: No false negatives on critical safety issues

**Why Keyword-Based Detection?**
- **Comprehensive Coverage**: Covers medical, pregnancy, emergency scenarios
- **Low Latency**: Instant safety assessment
- **Maintainable**: Easy to add new keywords and categories
- **Fail-Safe**: Errs on the side of caution

**Why Multi-Level Risk Assessment?**
- **CRITICAL**: Immediate blocking for emergencies
- **HIGH**: Strong warnings for medical/pregnancy queries
- **MEDIUM/LOW**: Gentle disclaimers for general safety
- **Graduated Response**: Appropriate action for each risk level

### **Technical Architecture Choices**

**Why FastAPI?**
- **Async Support**: Full async/await for high concurrency
- **Type Safety**: Pydantic integration for request/response validation
- **Performance**: One of the fastest Python web frameworks
- **Documentation**: Auto-generated OpenAPI/Swagger docs

**Why MongoDB for Logging?**
- **Flexible Schema**: Easy to evolve log structure
- **JSON Native**: Perfect for storing interaction data
- **Scalable**: Handles high-volume logging efficiently
- **Analytics Ready**: Easy to query and analyze user interactions

**Why Redis for Caching?**
- **Performance**: Sub-millisecond response times
- **Persistence**: Optional data persistence
- **Scalability**: Handles high concurrent load
- **TTL Support**: Automatic cache expiration

## 🛡️ Safety Logic

The project implements a deterministic safety filter in `src/services/safety`:

-   **Emergency**: Keywords like "suicide", "heart attack", "bleeding" trigger a **CRITICAL** risk.
    -   *Action*: Request is blocked. User is directed to emergency services.
-   **Pregnancy**: Keywords like "pregnant", "bump", "trimester" trigger a **HIGH/MEDIUM** risk.
    -   *Action*: Response allowed but appended with: *"Prenatal yoga should be practiced under expert guidance."*
-   **Medical Conditions**: Specific keywords (e.g., "hernia", "surgery") trigger **MEDICAL_ADVICE** flag.
    -   *Action*: Warning added: *"Please consult a doctor or certified yoga therapist..."*

## 📊 Data Models

Key Pydantic models in `src/models/schemas.py`:

-   **`Chunk`**: Represents a piece of knowledge.
    -   `id`: Unique identifier.
    -   `content`: The text content.
    -   `metadata`: Source, category (`YOGA`, `WELLNESS`, etc.), and timestamp.
    -   `embedding`: Vector representation (optional output).
-   **`SafetyAssessment`**: Result of the safety filter.
    -   `risk_level`: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`.
    -   `flags`: List of specific issues found (e.g., `SafetyFlagType.EMERGENCY`).
    -   `required_disclaimers`: List of strings to show the user.
-   **`UserInteractionLog`**: For analytics.
    -   `query`: User input.
    -   `response_content`: System output.
    -   `safety_flags`: What risks were detected.
    -   `processing_time_ms`: Performance metric.

## 🧪 Testing

The project includes comprehensive testing:

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src

# Run specific test categories
python -m pytest tests/unit/          # Unit tests
python -m pytest tests/property/     # Property-based tests
python -m pytest tests/unit/test_api_routes.py  # API tests
```

**Test Coverage**: 64% overall with 95% test success rate (59/62 tests passing)

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run individual services
docker build -t yoga-rag-app .
docker run -p 8000:8000 yoga-rag-app
```

## 🔧 Tech Stack

-   **Backend**: Python 3.11, FastAPI, Uvicorn
-   **Frontend**: HTML5, CSS3, Vanilla JavaScript
-   **AI/ML**: Sentence Transformers, OpenAI GPT-4, NVIDIA NIM (optional)
-   **Vector DB**: ChromaDB (Local) / Pinecone (Cloud)
-   **Database**: MongoDB (Logging)
-   **Caching**: Redis (optional)
-   **Testing**: Pytest, Hypothesis (property-based testing)
-   **Containerization**: Docker, Docker Compose

## 📈 Performance Features

-   **Rate Limiting**: 100 requests per minute per IP
-   **Redis Caching**: Embedding and response caching
-   **Async Processing**: Full async/await implementation
-   **Connection Pooling**: MongoDB and Redis connection pools
-   **Health Checks**: `/api/v1/health` endpoint for monitoring

## 🚦 API Endpoints

-   `GET /` - Frontend application
-   `POST /api/v1/ask` - Submit yoga questions
-   `POST /api/v1/feedback` - Submit user feedback
-   `GET /api/v1/health` - Health check
-   `GET /style.css` - Frontend styles
-   `GET /app.js` - Frontend JavaScript

## 📝 License

This project is licensed under the MIT License.

## 🤖 AI Development Tools Used

This project was developed using **Kiro AI IDE** - an AI-powered development environment. Below is a comprehensive list of key prompts and interactions used during development:

### **Initial Project Setup & Architecture**
- "Create a comprehensive RAG application for yoga wellness with safety filters"
- "Design a modular architecture with FastAPI backend and vanilla JS frontend"
- "Implement semantic chunking for yoga knowledge base processing"
- "Set up MongoDB logging and ChromaDB/Pinecone vector storage"

### **Core RAG Pipeline Development**
- "Implement embedding service with Sentence Transformers and NVIDIA NIM support"
- "Create semantic chunker with 256-512 token chunks and overlap handling"
- "Build retrieval engine with similarity search and metadata validation"
- "Develop response generator with OpenAI GPT-4 integration and source attribution"

### **Safety System Implementation**
- "Design deterministic safety filter for medical, pregnancy, and emergency queries"
- "Implement multi-level risk assessment (CRITICAL, HIGH, MEDIUM, LOW)"
- "Create safety response generation with appropriate warnings and disclaimers"
- "Add comprehensive safety incident logging to MongoDB"

### **Performance & Production Features**
- "Implement Redis caching layer for embeddings and responses"
- "Add rate limiting with Redis backend and in-memory fallback"
- "Create comprehensive API unit tests with proper mocking"
- "Set up Docker containerization with docker-compose"

### **Testing & Quality Assurance**
- "Write property-based tests using Hypothesis framework"
- "Create unit tests for all core services with 95%+ coverage"
- "Implement API endpoint tests with safety and rate limiting scenarios"
- "Add comprehensive error handling and graceful degradation"

### **Documentation & Deployment**
- "Create comprehensive README with architecture decisions and setup instructions"
- "Document RAG pipeline choices and safety system rationale"
- "Add Makefile with developer-friendly commands"
- "Prepare project for GitHub with proper .gitignore and structure"

### **Key AI-Assisted Decisions**
- **Embedding Strategy**: Chose dual-provider approach (Sentence Transformers + NVIDIA) for flexibility
- **Safety Approach**: Implemented deterministic keyword-based filtering for reliability
- **Architecture Pattern**: Used factory pattern for service abstraction and testability
- **Testing Strategy**: Combined unit tests with property-based testing for robustness
- **Performance**: Added Redis caching and rate limiting for production scalability

### **Development Methodology**
- Incremental development with continuous testing
- AI-guided architectural decisions based on best practices
- Comprehensive error handling and logging throughout
- Production-ready deployment configuration
