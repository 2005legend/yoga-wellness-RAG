# Wellness RAG Application Makefile

.PHONY: help install install-dev test test-unit test-property test-integration lint format type-check clean run docker-build docker-run setup-env

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  test          Run all tests"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-property Run property-based tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black"
	@echo "  type-check    Run type checking with mypy"
	@echo "  clean         Clean up temporary files"
	@echo "  run           Run the application"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run Docker container"
	@echo "  setup-env     Set up environment file"

# Installation
install:
	pip install -r requirements.txt

install-dev: install
	pip install -e .
	pre-commit install

# Testing
test:
	pytest -v --cov=src --cov-report=term-missing --cov-report=html

test-unit:
	pytest -v -m "unit" --cov=src

test-property:
	pytest -v -m "property" --cov=src

test-integration:
	pytest -v -m "integration" --cov=src

# Code quality
lint:
	flake8 src tests
	black --check src tests
	mypy src

format:
	black src tests
	isort src tests

type-check:
	mypy src

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Development
run:
	python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Docker
docker-build:
	docker build -t wellness-rag-app .

docker-run:
	docker run -p 8000:8000 --env-file .env wellness-rag-app

# Environment setup
setup-env:
	cp .env.example .env
	@echo "Please edit .env file with your configuration"