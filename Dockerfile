FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (e.g., for building some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements if exists, else manual install
# Since we don't have a requirements.txt yet, we install manually
RUN pip install --no-cache-dir \
    fastapi uvicorn motor python-multipart structlog \
    sentence-transformers torch numpy \
    pinecone-client chromadb \
    openai pydantic pydantic-settings

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]