#!/bin/bash

# Destroy script for AI Debate Game infrastructure
# Usage: ./destroy.sh [environment] [region]

set -e  # Exit on any error

# Configuration
ENVIRONMENT=${1:-dev}
AWS_REGION=${2:-us-east-1}
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
TERRAFORM_DIR="$PROJECT_ROOT/deployment/terraform"

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

# Confirmation prompt
confirm_destruction() {
    echo ""
    log_warning "⚠️  WARNING: This will PERMANENTLY DESTROY all infrastructure for environment: $ENVIRONMENT"
    echo ""
    echo "This includes:"
    echo "  - Lambda functions"
    echo "  - API Gateway"
    echo "  - S3 buckets and ALL their contents"
    echo "  - CloudFront distribution"
    echo "  - CloudWatch logs"
    echo "  - All other AWS resources"
    echo ""
    
    if [ "$ENVIRONMENT" == "production" ]; then
        log_error "🚨 PRODUCTION ENVIRONMENT DETECTED!"
        echo ""
        echo "You are about to destroy the PRODUCTION environment."
        echo "This action is IRREVERSIBLE and will result in:"
        echo "  - Complete service outage"
        echo "  - Loss of all user data"
        echo "  - Loss of all trained models"
        echo "  - Loss of all game history"
        echo ""
        read -p "Type 'DESTROY PRODUCTION' to confirm: " confirmation
        if [ "$confirmation" != "DESTROY PRODUCTION" ]; then
            log_info "Destruction cancelled."
            exit 0
        fi
    else
        read -p "Type 'yes' to confirm destruction of $ENVIRONMENT environment: " confirmation
        if [ "$confirmation" != "yes" ]; then
            log_info "Destruction cancelled."
            exit 0
        fi
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

# Empty S3 buckets before destruction
empty_s3_buckets() {
    log_info "Emptying S3 buckets..."
    
    cd "$TERRAFORM_DIR"
    
    # Check if terraform state exists
    if ! terraform workspace select "$ENVIRONMENT" 2>/dev/null; then
        log_warning "Terraform workspace '$ENVIRONMENT' does not exist. Skipping S3 cleanup."
        return 0
    fi
    
    # Get bucket names from terraform output
    local frontend_bucket=""
    local models_bucket=""
    local logs_bucket=""
    
    # Try to get bucket names from terraform output
    if terraform output -raw s3_frontend_bucket &> /dev/null; then
        frontend_bucket=$(terraform output -raw s3_frontend_bucket)
    fi
    
    if terraform output -raw s3_models_bucket &> /dev/null; then
        models_bucket=$(terraform output -raw s3_models_bucket)
    fi
    
    # Manually construct likely bucket names if terraform outputs fail
    if [ -z "$frontend_bucket" ]; then
        frontend_bucket="debate-game-${ENVIRONMENT}-frontend"
    fi
    
    if [ -z "$models_bucket" ]; then
        models_bucket="debate-game-${ENVIRONMENT}-models"
    fi
    
    logs_bucket="debate-game-${ENVIRONMENT}-cloudfront-logs"
    
    # Empty buckets
    for bucket in "$frontend_bucket" "$models_bucket" "$logs_bucket"; do
        if aws s3api head-bucket --bucket "$bucket" --region "$AWS_REGION" 2>/dev/null; then
            log_info "Emptying bucket: $bucket"
            
            # Delete all objects including versions
            aws s3api delete-objects \
                --bucket "$bucket" \
                --delete "$(aws s3api list-object-versions \
                    --bucket "$bucket" \
                    --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' \
                    --region "$AWS_REGION")" \
                --region "$AWS_REGION" 2>/dev/null || true
                
            # Delete delete markers
            aws s3api delete-objects \
                --bucket "$bucket" \
                --delete "$(aws s3api list-object-versions \
                    --bucket "$bucket" \
                    --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' \
                    --region "$AWS_REGION")" \
                --region "$AWS_REGION" 2>/dev/null || true
                
            log_success "Emptied bucket: $bucket"
        else
            log_warning "Bucket $bucket does not exist or is not accessible."
        fi
    done
}

# Destroy infrastructure with Terraform
destroy_terraform() {
    log_info "Destroying infrastructure with Terraform..."
    
    cd "$TERRAFORM_DIR"
    
    # Select the correct workspace
    if ! terraform workspace select "$ENVIRONMENT" 2>/dev/null; then
        log_error "Terraform workspace '$ENVIRONMENT' does not exist."
        exit 1
    fi
    
    # Plan destruction
    log_info "Planning destruction..."
    terraform plan -destroy \
        -var="environment=$ENVIRONMENT" \
        -var="aws_region=$AWS_REGION" \
        -out="destroy-plan-$ENVIRONMENT"
    
    # Apply destruction
    log_info "Applying destruction..."
    terraform apply "destroy-plan-$ENVIRONMENT"
    
    # Clean up plan file
    rm -f "destroy-plan-$ENVIRONMENT"
    
    log_success "Infrastructure destroyed successfully."
}

# Clean up Terraform workspace
cleanup_terraform() {
    log_info "Cleaning up Terraform workspace..."
    
    cd "$TERRAFORM_DIR"
    
    # Switch to default workspace
    terraform workspace select default
    
    # Delete the environment workspace
    terraform workspace delete "$ENVIRONMENT" || log_warning "Could not delete workspace $ENVIRONMENT"
    
    log_success "Terraform workspace cleaned up."
}

# Verify destruction
verify_destruction() {
    log_info "Verifying destruction..."
    
    local failed_resources=()
    
    # Check if Lambda functions still exist
    if aws lambda get-function --function-name "debate-game-${ENVIRONMENT}-backend" --region "$AWS_REGION" 2>/dev/null; then
        failed_resources+=("Lambda function: debate-game-${ENVIRONMENT}-backend")
    fi
    
    if aws lambda get-function --function-name "debate-game-${ENVIRONMENT}-ml-inference" --region "$AWS_REGION" 2>/dev/null; then
        failed_resources+=("Lambda function: debate-game-${ENVIRONMENT}-ml-inference")
    fi
    
    # Check if S3 buckets still exist
    local frontend_bucket="debate-game-${ENVIRONMENT}-frontend"
    local models_bucket="debate-game-${ENVIRONMENT}-models"
    
    if aws s3api head-bucket --bucket "$frontend_bucket" --region "$AWS_REGION" 2>/dev/null; then
        failed_resources+=("S3 bucket: $frontend_bucket")
    fi
    
    if aws s3api head-bucket --bucket "$models_bucket" --region "$AWS_REGION" 2>/dev/null; then
        failed_resources+=("S3 bucket: $models_bucket")
    fi
    
    # Check if API Gateway still exists
    local api_name="debate-game-${ENVIRONMENT}-api"
    if aws apigateway get-rest-apis --region "$AWS_REGION" --query "items[?name=='$api_name'].id" --output text | grep -q .; then
        failed_resources+=("API Gateway: $api_name")
    fi
    
    if [ ${#failed_resources[@]} -eq 0 ]; then
        log_success "All resources successfully destroyed."
    else
        log_warning "Some resources may still exist:"
        for resource in "${failed_resources[@]}"; do
            echo "  - $resource"
        done
        log_warning "These may be eventually consistent deletions or require manual cleanup."
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    
    # Remove any temporary files
    rm -f "$TERRAFORM_DIR/destroy-plan-$ENVIRONMENT"
    rm -f "$PROJECT_ROOT/deployment/deployment_outputs.env"
    
    log_success "Cleanup completed."
}

# Main destruction function
main() {
    log_info "Starting infrastructure destruction for environment: $ENVIRONMENT, region: $AWS_REGION"
    
    # Validation
    if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|production)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT. Must be dev, staging, or production."
        exit 1
    fi
    
    confirm_destruction
    check_dependencies
    check_aws_auth
    
    empty_s3_buckets
    destroy_terraform
    cleanup_terraform
    verify_destruction
    cleanup
    
    log_success "Destruction completed successfully!"
    echo ""
    log_info "Environment '$ENVIRONMENT' has been completely destroyed."
    echo "All AWS resources for this environment have been removed."
}

# Error handler
handle_error() {
    log_error "Destruction failed at step: $1"
    cleanup
    exit 1
}

# Trap errors
trap 'handle_error "unknown"' ERR

# Run main function
main "$@"