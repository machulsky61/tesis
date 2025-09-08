#!/bin/bash

# Deployment script for AI Debate Game
# Usage: ./deploy.sh [environment] [region]

set -e  # Exit on any error

# Configuration
ENVIRONMENT=${1:-dev}
AWS_REGION=${2:-us-east-1}
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
TERRAFORM_DIR="$PROJECT_ROOT/deployment/terraform"
MODELS_SOURCE_DIR="$(dirname "$PROJECT_ROOT")/debate_mnist/models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validation function
validate_environment() {
    if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|production)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT. Must be dev, staging, or production."
        exit 1
    fi
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is required but not installed."
        exit 1
    fi
    
    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is required but not installed."
        exit 1
    fi
    
    # Check Node.js (for frontend build)
    if ! command -v node &> /dev/null; then
        log_error "Node.js is required but not installed."
        exit 1
    fi
    
    # Check Python (for backend packaging)
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed."
        exit 1
    fi
    
    log_success "All dependencies are available."
}

# Check AWS authentication
check_aws_auth() {
    log_info "Checking AWS authentication..."
    
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS authentication failed. Please configure AWS CLI."
        exit 1
    fi
    
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    local region=$(aws configure get region)
    
    log_success "AWS authenticated. Account: $account_id, Region: ${region:-default}"
}

# Package backend for Lambda
package_backend() {
    log_info "Packaging backend for Lambda deployment..."
    
    cd "$BACKEND_DIR"
    
    # Create deployment directory
    rm -rf deployment
    mkdir deployment
    
    # Install dependencies in deployment directory
    pip3 install -r requirements.txt -t deployment/ --no-deps
    
    # Add required packages that might be missing
    pip3 install mangum -t deployment/
    
    # Copy application code
    cp -r app/ deployment/
    
    # Create deployment package
    cd deployment
    zip -r ../deployment.zip . -x "*.pyc" "__pycache__/*"
    
    # Create ML inference package (simplified version)
    cd ..
    mkdir -p ml-deployment
    cp -r deployment/* ml-deployment/
    
    # Copy models from main project if available
    if [ -d "$MODELS_SOURCE_DIR" ]; then
        log_info "Copying ML models from main project..."
        mkdir -p ml-deployment/models
        cp -r "$MODELS_SOURCE_DIR"/* ml-deployment/models/ 2>/dev/null || log_warning "Could not copy some model files"
    else
        log_warning "Main project models directory not found: $MODELS_SOURCE_DIR"
        mkdir -p ml-deployment/models
        echo "placeholder" > ml-deployment/models/README.txt
    fi
    
    cd ml-deployment
    zip -r ../ml-deployment.zip . -x "*.pyc" "__pycache__/*"
    
    cd ..
    rm -rf deployment ml-deployment
    
    log_success "Backend packaged successfully."
}

# Build frontend
build_frontend() {
    log_info "Building frontend..."
    
    cd "$FRONTEND_DIR"
    
    # Install dependencies
    npm ci
    
    # Set environment variables for build
    export REACT_APP_API_URL="/api"
    export REACT_APP_ENVIRONMENT="$ENVIRONMENT"
    
    # Build
    npm run build
    
    log_success "Frontend built successfully."
}

# Initialize Terraform
init_terraform() {
    log_info "Initializing Terraform..."
    
    cd "$TERRAFORM_DIR"
    
    # Initialize Terraform
    terraform init
    
    # Create workspace if it doesn't exist
    terraform workspace select "$ENVIRONMENT" 2>/dev/null || terraform workspace new "$ENVIRONMENT"
    
    log_success "Terraform initialized for environment: $ENVIRONMENT"
}

# Plan Terraform deployment
plan_terraform() {
    log_info "Planning Terraform deployment..."
    
    cd "$TERRAFORM_DIR"
    
    # Plan deployment
    terraform plan \
        -var="environment=$ENVIRONMENT" \
        -var="aws_region=$AWS_REGION" \
        -out="tfplan-$ENVIRONMENT"
    
    log_success "Terraform plan completed."
}

# Apply Terraform deployment
apply_terraform() {
    log_info "Applying Terraform deployment..."
    
    cd "$TERRAFORM_DIR"
    
    # Apply deployment
    terraform apply "tfplan-$ENVIRONMENT"
    
    # Get outputs
    local api_url=$(terraform output -raw api_gateway_stage_url)
    local cloudfront_url=$(terraform output -raw cloudfront_url)
    local s3_frontend_bucket=$(terraform output -raw s3_frontend_bucket)
    local s3_models_bucket=$(terraform output -raw s3_models_bucket)
    
    # Store outputs for later use
    cat > ../deployment_outputs.env << EOF
API_URL=$api_url
CLOUDFRONT_URL=$cloudfront_url
S3_FRONTEND_BUCKET=$s3_frontend_bucket
S3_MODELS_BUCKET=$s3_models_bucket
ENVIRONMENT=$ENVIRONMENT
AWS_REGION=$AWS_REGION
EOF
    
    log_success "Terraform deployment applied successfully."
}

# Deploy frontend to S3
deploy_frontend() {
    log_info "Deploying frontend to S3..."
    
    # Source deployment outputs
    source "$PROJECT_ROOT/deployment/deployment_outputs.env"
    
    cd "$FRONTEND_DIR"
    
    # Sync built files to S3
    aws s3 sync build/ "s3://$S3_FRONTEND_BUCKET" --delete --region "$AWS_REGION"
    
    # Invalidate CloudFront cache
    local distribution_id=$(cd "$TERRAFORM_DIR" && terraform output -raw cloudfront_distribution_id)
    aws cloudfront create-invalidation --distribution-id "$distribution_id" --paths "/*" --region "$AWS_REGION"
    
    log_success "Frontend deployed successfully."
}

# Deploy models to S3
deploy_models() {
    log_info "Deploying models to S3..."
    
    # Source deployment outputs
    source "$PROJECT_ROOT/deployment/deployment_outputs.env"
    
    if [ -d "$MODELS_SOURCE_DIR" ]; then
        aws s3 sync "$MODELS_SOURCE_DIR" "s3://$S3_MODELS_BUCKET/judge-models/" --region "$AWS_REGION"
        log_success "Models deployed successfully."
    else
        log_warning "Models directory not found, skipping model deployment."
    fi
}

# Update Lambda functions
update_lambda() {
    log_info "Updating Lambda functions..."
    
    cd "$BACKEND_DIR"
    
    # Update backend function
    aws lambda update-function-code \
        --function-name "debate-game-${ENVIRONMENT}-backend" \
        --zip-file "fileb://deployment.zip" \
        --region "$AWS_REGION"
    
    # Update ML inference function
    aws lambda update-function-code \
        --function-name "debate-game-${ENVIRONMENT}-ml-inference" \
        --zip-file "fileb://ml-deployment.zip" \
        --region "$AWS_REGION"
    
    log_success "Lambda functions updated successfully."
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Source deployment outputs
    source "$PROJECT_ROOT/deployment/deployment_outputs.env"
    
    # Wait a moment for deployment to settle
    sleep 10
    
    # Check API health
    local health_url="${API_URL}/health"
    log_info "Checking API health at: $health_url"
    
    local max_attempts=5
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$health_url" > /dev/null; then
            log_success "API health check passed."
            break
        else
            log_warning "API health check failed (attempt $attempt/$max_attempts). Retrying in 10 seconds..."
            sleep 10
            ((attempt++))
        fi
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log_error "API health check failed after $max_attempts attempts."
        return 1
    fi
    
    # Check frontend
    log_info "Checking frontend at: $CLOUDFRONT_URL"
    if curl -s -f "$CLOUDFRONT_URL" > /dev/null; then
        log_success "Frontend health check passed."
    else
        log_warning "Frontend health check failed (this is normal for new deployments as CloudFront needs time to propagate)."
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    
    # Remove deployment packages
    rm -f "$BACKEND_DIR/deployment.zip"
    rm -f "$BACKEND_DIR/ml-deployment.zip"
    
    # Remove Terraform plan files
    rm -f "$TERRAFORM_DIR/tfplan-$ENVIRONMENT"
    
    log_success "Cleanup completed."
}

# Display deployment summary
show_summary() {
    log_info "Deployment Summary"
    echo "=================================="
    
    if [ -f "$PROJECT_ROOT/deployment/deployment_outputs.env" ]; then
        source "$PROJECT_ROOT/deployment/deployment_outputs.env"
        
        echo "Environment: $ENVIRONMENT"
        echo "Region: $AWS_REGION"
        echo ""
        echo "Frontend URL: $CLOUDFRONT_URL"
        echo "API URL: $API_URL"
        echo ""
        echo "S3 Frontend Bucket: $S3_FRONTEND_BUCKET"
        echo "S3 Models Bucket: $S3_MODELS_BUCKET"
        echo ""
        echo "The application should be accessible at:"
        echo "$CLOUDFRONT_URL"
    else
        log_error "Deployment outputs not found."
    fi
}

# Error handler
handle_error() {
    log_error "Deployment failed at step: $1"
    cleanup
    exit 1
}

# Main deployment function
main() {
    log_info "Starting deployment for environment: $ENVIRONMENT, region: $AWS_REGION"
    
    validate_environment
    check_dependencies || handle_error "dependency_check"
    check_aws_auth || handle_error "aws_auth"
    
    package_backend || handle_error "backend_packaging"
    build_frontend || handle_error "frontend_build"
    
    init_terraform || handle_error "terraform_init"
    plan_terraform || handle_error "terraform_plan"
    apply_terraform || handle_error "terraform_apply"
    
    deploy_frontend || handle_error "frontend_deployment"
    deploy_models || handle_error "model_deployment"
    
    # For initial deployment, Lambda functions are created by Terraform
    # For updates, we would update them separately
    if [ "$ENVIRONMENT" != "dev" ] || [ -f "$PROJECT_ROOT/deployment/deployment_outputs.env" ]; then
        update_lambda || handle_error "lambda_update"
    fi
    
    health_check || log_warning "Health check failed, but deployment may still be successful"
    
    cleanup
    show_summary
    
    log_success "Deployment completed successfully!"
}

# Run main function
main "$@"