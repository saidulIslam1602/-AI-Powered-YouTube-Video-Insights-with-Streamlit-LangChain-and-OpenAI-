#!/bin/bash

# YouTube Insights Platform - Enterprise Deployment Script
# This script provides various deployment options for the platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="youtube-insights"
DOCKER_IMAGE="${APP_NAME}:latest"
NAMESPACE="youtube-insights"

# Functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  YouTube Insights Platform     ${NC}"
    echo -e "${BLUE}  Enterprise Deployment        ${NC}"
    echo -e "${BLUE}================================${NC}"
    echo
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_dependencies() {
    print_header
    echo "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose is installed"
    
    # Check kubectl (optional)
    if command -v kubectl &> /dev/null; then
        print_success "kubectl is installed"
        KUBECTL_AVAILABLE=true
    else
        print_warning "kubectl is not installed. Kubernetes deployment will be skipped."
        KUBECTL_AVAILABLE=false
    fi
}

build_docker_image() {
    echo "Building Docker image..."
    docker build -t $DOCKER_IMAGE .
    print_success "Docker image built successfully"
}

deploy_development() {
    echo "Deploying development environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        echo "Creating .env file..."
        cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo
LOG_LEVEL=INFO
CACHE_ENABLED=true
DEBUG=true
POSTGRES_DB=youtube_insights
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
GRAFANA_PASSWORD=admin
EOF
        print_warning "Please update .env file with your actual API keys"
    fi
    
    # Start services
    docker-compose up -d youtube-insights redis mailhog
    print_success "Development environment deployed"
    echo "Access the application at: http://localhost:8501"
    echo "Access MailHog at: http://localhost:8025"
}

deploy_production() {
    echo "Deploying production environment..."
    
    # Check for required environment variables
    if [ -z "$OPENAI_API_KEY" ]; then
        print_error "OPENAI_API_KEY environment variable is required"
        exit 1
    fi
    
    # Start all services
    docker-compose --profile production up -d
    print_success "Production environment deployed"
    echo "Access the application at: http://localhost:8501"
    echo "Access Grafana at: http://localhost:3000 (admin/admin)"
    echo "Access Prometheus at: http://localhost:9090"
}

deploy_monitoring() {
    echo "Deploying monitoring stack..."
    docker-compose --profile monitoring up -d
    print_success "Monitoring stack deployed"
    echo "Access Grafana at: http://localhost:3000 (admin/admin)"
    echo "Access Prometheus at: http://localhost:9090"
}

deploy_logging() {
    echo "Deploying logging stack..."
    docker-compose --profile logging up -d
    print_success "Logging stack deployed"
    echo "Access Kibana at: http://localhost:5601"
    echo "Access Elasticsearch at: http://localhost:9200"
}

deploy_kubernetes() {
    if [ "$KUBECTL_AVAILABLE" = false ]; then
        print_error "kubectl is not available. Cannot deploy to Kubernetes."
        exit 1
    fi
    
    echo "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Redis
    kubectl apply -f k8s/redis.yaml -n $NAMESPACE
    
    # Deploy main application
    kubectl apply -f k8s/deployment.yaml -n $NAMESPACE
    
    # Deploy monitoring
    kubectl apply -f k8s/monitoring.yaml -n $NAMESPACE
    
    print_success "Kubernetes deployment completed"
    echo "Check deployment status with: kubectl get pods -n $NAMESPACE"
}

setup_demo_clients() {
    echo "Setting up demo clients..."
    python demo_client_setup.py
    print_success "Demo clients created"
}

show_help() {
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  dev         Deploy development environment"
    echo "  prod        Deploy production environment"
    echo "  monitoring  Deploy monitoring stack only"
    echo "  logging     Deploy logging stack only"
    echo "  k8s         Deploy to Kubernetes"
    echo "  build       Build Docker image only"
    echo "  demo        Setup demo clients"
    echo "  stop        Stop all services"
    echo "  logs        Show application logs"
    echo "  status      Show service status"
    echo "  help        Show this help message"
    echo
    echo "Examples:"
    echo "  $0 dev              # Deploy development environment"
    echo "  $0 prod             # Deploy production environment"
    echo "  $0 k8s              # Deploy to Kubernetes"
    echo "  $0 build && $0 dev  # Build and deploy development"
}

stop_services() {
    echo "Stopping all services..."
    docker-compose down
    print_success "All services stopped"
}

show_logs() {
    echo "Showing application logs..."
    docker-compose logs -f youtube-insights
}

show_status() {
    echo "Service Status:"
    docker-compose ps
}

# Main script logic
case "${1:-help}" in
    "dev")
        check_dependencies
        build_docker_image
        deploy_development
        setup_demo_clients
        ;;
    "prod")
        check_dependencies
        build_docker_image
        deploy_production
        ;;
    "monitoring")
        check_dependencies
        deploy_monitoring
        ;;
    "logging")
        check_dependencies
        deploy_logging
        ;;
    "k8s")
        check_dependencies
        build_docker_image
        deploy_kubernetes
        ;;
    "build")
        check_dependencies
        build_docker_image
        ;;
    "demo")
        setup_demo_clients
        ;;
    "stop")
        stop_services
        ;;
    "logs")
        show_logs
        ;;
    "status")
        show_status
        ;;
    "help"|*)
        show_help
        ;;
esac