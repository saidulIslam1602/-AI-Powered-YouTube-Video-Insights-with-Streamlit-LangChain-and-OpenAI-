# YouTube Video Insights - Makefile
# Common commands for development and deployment

.PHONY: help install dev-install test lint format clean run docker-build docker-run deploy

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)YouTube Video Insights - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	pip install -r requirements.txt

dev-install: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install -e ".[dev,docs]"
	pre-commit install

test: ## Run all tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest --cov=src --cov-report=html --cov-report=term-missing

test-fast: ## Run tests without coverage (faster)
	@echo "$(BLUE)Running fast tests...$(NC)"
	pytest -x

lint: ## Run all linters
	@echo "$(BLUE)Running linters...$(NC)"
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/
	isort src/ tests/

clean: ## Clean cache files and build artifacts
	@echo "$(BLUE)Cleaning cache and build files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

run: ## Run the Streamlit application
	@echo "$(BLUE)Starting YouTube Video Insights...$(NC)"
	streamlit run app.py

run-dev: ## Run the application in development mode
	@echo "$(BLUE)Starting in development mode...$(NC)"
	streamlit run app.py --server.runOnSave true

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t youtube-insights:latest .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -p 8501:8501 --env-file .env youtube-insights:latest

docker-compose-up: ## Start with docker-compose
	@echo "$(BLUE)Starting with docker-compose...$(NC)"
	docker-compose up --build

docker-compose-down: ## Stop docker-compose services
	@echo "$(BLUE)Stopping docker-compose services...$(NC)"
	docker-compose down

env-setup: ## Set up environment file from example
	@echo "$(BLUE)Setting up environment file...$(NC)"
	@if [ ! -f .env ]; then \
		cp env.example .env; \
		echo "$(YELLOW)Created .env file from template. Please update with your API key.$(NC)"; \
	else \
		echo "$(YELLOW).env file already exists.$(NC)"; \
	fi

check: ## Run all checks (lint, test, security)
	@echo "$(BLUE)Running all checks...$(NC)"
	make lint
	make test
	@echo "$(GREEN)All checks passed!$(NC)"

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	pip-audit
	bandit -r src/

docs-build: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	mkdocs build

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	mkdocs serve

deploy-streamlit: ## Deploy to Streamlit Cloud (requires setup)
	@echo "$(BLUE)Deploying to Streamlit Cloud...$(NC)"
	@echo "$(YELLOW)Make sure you've connected your repository to Streamlit Cloud$(NC)"
	git push origin main

requirements-update: ## Update requirements.txt from current environment
	@echo "$(BLUE)Updating requirements.txt...$(NC)"
	pip freeze > requirements.txt

setup: ## Complete project setup for new developers
	@echo "$(BLUE)Setting up YouTube Video Insights for development...$(NC)"
	make env-setup
	make dev-install
	@echo "$(GREEN)Setup complete! Run 'make run' to start the application.$(NC)"
	@echo "$(YELLOW)Don't forget to add your OpenAI API key to the .env file!$(NC)"

health-check: ## Check application health
	@echo "$(BLUE)Checking application health...$(NC)"
	@curl -f http://localhost:8501/_stcore/health || echo "$(RED)Application is not running$(NC)"

logs: ## Show application logs (Docker)
	@echo "$(BLUE)Showing application logs...$(NC)"
	docker-compose logs -f youtube-insights

monitor: ## Start monitoring stack
	@echo "$(BLUE)Starting monitoring stack...$(NC)"
	docker-compose --profile monitoring up -d

# Development workflow shortcuts
dev: dev-install run-dev ## Quick development setup and run

ci: lint test ## Run CI checks locally

# Deployment shortcuts
prod: docker-build docker-run ## Build and run production Docker image 