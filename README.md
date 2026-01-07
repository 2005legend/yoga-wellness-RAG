# Wellness RAG Application

A comprehensive Retrieval-Augmented Generation (RAG) system for wellness and yoga guidance, built with FastAPI, MongoDB, and modern AI technologies.

## 🌟 Features

- **Complete RAG Pipeline**: Document chunking, embedding generation, similarity search, and response generation
- **Safety-First Design**: Medical advice detection, crisis response, and wellness vs medical differentiation
- **Comprehensive Logging**: MongoDB-based interaction and safety incident logging
- **Production-Ready**: Docker containerization, monitoring, performance optimization
- **Property-Based Testing**: 21 correctness properties with comprehensive test coverage

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   RAG Pipeline  │
│   (React/Vue)   │◄──►│   (FastAPI)     │◄──►│   (Python)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   MongoDB       │    │ Vector Database │
                       │   (Logs)        │    │ (Pinecone/Chroma)│
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- MongoDB
- Redis (optional, for caching)
- OpenAI API key
- Vector database (Pinecone or ChromaDB)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd wellness-rag-app
   ```

2. **Set up environment**
   ```bash
   make setup-env
   # Edit .env file with your configuration
   ```

3. **Install dependencies**
   ```bash
   make install-dev
   ```

4. **Run the application**
   ```bash
   make run
   ```

The API will be available at `http://localhost:8000`

## 📋 Configuration

Copy `.env.example` to `.env` and configure the following:

### Required Configuration
- `OPENAI_API_KEY`: Your OpenAI API key
- `MONGODB_URL`: MongoDB connection string
- `PINECONE_API_KEY`: Pinecone API key (if using Pinecone)

### Optional Configuration
- `REDIS_URL`: Redis connection for caching
- `DEBUG`: Enable debug mode
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
make test

# Unit tests only
make test-unit

# Property-based tests only
make test-property

# Integration tests only
make test-integration
```

## 🔧 Development

### Code Quality
```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check
```

### Project Structure
```
src/
├── api/              # FastAPI routes and middleware
├── core/             # Core utilities (logging, exceptions)
├── models/           # Pydantic schemas and data models
├── services/         # Business logic services
│   ├── chunking/     # Document chunking service
│   ├── embedding/    # Embedding generation service
│   ├── retrieval/    # Retrieval engine
│   ├── generation/   # Response generation service
│   ├── safety/       # Safety filtering service
│   └── logging/      # Interaction logging service
└── config.py         # Configuration management

tests/
├── unit/             # Unit tests
├── property/         # Property-based tests
└── integration/      # Integration tests
```

## 🛡️ Safety Features

The application includes comprehensive safety mechanisms:

- **Medical Advice Detection**: Identifies and blocks requests for medical diagnosis, prescription, or treatment
- **Crisis Response**: Detects emergency situations and provides appropriate resources
- **Content Filtering**: Prevents inappropriate or harmful content
- **Disclaimer System**: Automatically adds appropriate disclaimers to responses

## 📊 Monitoring

The application provides comprehensive monitoring:

- **Health Checks**: `/health` endpoint for system status
- **Metrics**: Prometheus metrics on port 9090
- **Structured Logging**: JSON-formatted logs for analysis
- **Safety Incident Tracking**: Automatic logging of safety concerns

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build image
make docker-build

# Run container
make docker-run
```

Or use docker-compose:

```bash
docker-compose up -d
```

## 📚 API Documentation

Once running, visit:
- **Interactive API Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

### Key Endpoints

- `POST /api/v1/query` - Submit wellness queries
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation at `/docs`
- Review the test suite for usage examples

## 🔮 Roadmap

- [ ] Advanced retrieval strategies (hybrid search, re-ranking)
- [ ] Multi-language support
- [ ] Voice interface integration
- [ ] Advanced analytics dashboard
- [ ] Mobile application support