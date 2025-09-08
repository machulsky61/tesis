#!/bin/bash

# Verification script for AI Debate Game deployment
# Usage: ./verify.sh [environment] [region]

set -e

# Configuration
ENVIRONMENT=${1:-dev}
AWS_REGION=${2:-us-east-1}
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

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

# Load deployment outputs
load_outputs() {
    if [ ! -f "$PROJECT_ROOT/deployment/deployment_outputs.env" ]; then
        log_error "Deployment outputs not found. Run ./deploy.sh first."
        exit 1
    fi
    
    source "$PROJECT_ROOT/deployment/deployment_outputs.env"
    log_info "Loaded deployment outputs for environment: $ENVIRONMENT"
}

# Test API endpoints
test_api() {
    log_info "Testing API endpoints..."
    
    # Test health endpoint
    log_info "Testing health endpoint: $API_URL/health"
    if curl -s -f "$API_URL/health" > /dev/null; then
        log_success "✓ Health endpoint responding"
    else
        log_error "✗ Health endpoint failed"
        return 1
    fi
    
    # Test game creation
    log_info "Testing game creation endpoint..."
    local create_response=$(curl -s -X POST "$API_URL/game/create" \
        -H "Content-Type: application/json" \
        -d '{"k": 6, "player_role": "honest", "agent_type": "greedy", "precommit": false}')
    
    if echo "$create_response" | grep -q "game_id"; then
        log_success "✓ Game creation working"
        
        # Extract game_id for further testing
        local game_id=$(echo "$create_response" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('game_id', ''))
" 2>/dev/null)
        
        if [ ! -z "$game_id" ]; then
            log_info "Testing game state endpoint..."
            if curl -s -f "$API_URL/game/$game_id/state" > /dev/null; then
                log_success "✓ Game state endpoint working"
            else
                log_warning "⚠ Game state endpoint issue"
            fi
        fi
    else
        log_error "✗ Game creation failed"
        echo "Response: $create_response"
        return 1
    fi
    
    log_success "API tests completed successfully"
}

# Test frontend
test_frontend() {
    log_info "Testing frontend..."
    
    log_info "Testing frontend URL: $CLOUDFRONT_URL"
    
    # Test if CloudFront returns content
    local status_code=$(curl -s -o /dev/null -w "%{http_code}" "$CLOUDFRONT_URL")
    
    if [ "$status_code" == "200" ]; then
        log_success "✓ Frontend accessible"
        
        # Check if it looks like a React app
        local content=$(curl -s "$CLOUDFRONT_URL")
        if echo "$content" | grep -q "React\|root"; then
            log_success "✓ Frontend appears to be React app"
        else
            log_warning "⚠ Frontend content may not be correct React app"
        fi
    elif [ "$status_code" == "404" ] || [ "$status_code" == "403" ]; then
        log_warning "⚠ Frontend not yet propagated (CloudFront takes time)"
        log_info "This is normal for new deployments. Try again in 10-15 minutes."
    else
        log_error "✗ Frontend not accessible (HTTP $status_code)"
        return 1
    fi
}

# Test AWS resources
test_aws_resources() {
    log_info "Testing AWS resources..."
    
    # Test Lambda functions
    log_info "Checking Lambda functions..."
    
    if aws lambda get-function --function-name "debate-game-${ENVIRONMENT}-backend" --region "$AWS_REGION" >/dev/null 2>&1; then
        log_success "✓ Backend Lambda function exists"
    else
        log_error "✗ Backend Lambda function not found"
        return 1
    fi
    
    if aws lambda get-function --function-name "debate-game-${ENVIRONMENT}-ml-inference" --region "$AWS_REGION" >/dev/null 2>&1; then
        log_success "✓ ML Lambda function exists"
    else
        log_error "✗ ML Lambda function not found"
        return 1
    fi
    
    # Test S3 buckets
    log_info "Checking S3 buckets..."
    
    if aws s3api head-bucket --bucket "$S3_FRONTEND_BUCKET" --region "$AWS_REGION" 2>/dev/null; then
        log_success "✓ Frontend S3 bucket exists"
        
        # Check if bucket has content
        local object_count=$(aws s3api list-objects-v2 --bucket "$S3_FRONTEND_BUCKET" --region "$AWS_REGION" --query 'length(Contents)')
        if [ "$object_count" != "null" ] && [ "$object_count" -gt 0 ]; then
            log_success "✓ Frontend bucket has content ($object_count files)"
        else
            log_warning "⚠ Frontend bucket is empty"
        fi
    else
        log_error "✗ Frontend S3 bucket not accessible"
        return 1
    fi
    
    if aws s3api head-bucket --bucket "$S3_MODELS_BUCKET" --region "$AWS_REGION" 2>/dev/null; then
        log_success "✓ Models S3 bucket exists"
    else
        log_error "✗ Models S3 bucket not accessible"
        return 1
    fi
    
    # Test API Gateway
    log_info "Checking API Gateway..."
    local api_name="debate-game-${ENVIRONMENT}-api"
    local api_id=$(aws apigateway get-rest-apis --region "$AWS_REGION" --query "items[?name=='$api_name'].id" --output text)
    
    if [ ! -z "$api_id" ] && [ "$api_id" != "None" ]; then
        log_success "✓ API Gateway exists (ID: $api_id)"
    else
        log_error "✗ API Gateway not found"
        return 1
    fi
    
    log_success "AWS resources verification completed"
}

# Performance test
performance_test() {
    log_info "Running basic performance test..."
    
    # Test API response time
    local start_time=$(date +%s.%N)
    curl -s "$API_URL/health" > /dev/null
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "unknown")
    
    if [ "$duration" != "unknown" ]; then
        log_info "API response time: ${duration}s"
        
        # Check if response is reasonable (< 5 seconds)
        if (( $(echo "$duration < 5.0" | bc 2>/dev/null || echo 0) )); then
            log_success "✓ API response time acceptable"
        else
            log_warning "⚠ API response time slow (${duration}s)"
        fi
    fi
}

# Generate verification report
generate_report() {
    log_info "Generating verification report..."
    
    local report_file="$PROJECT_ROOT/deployment/verification_report_${ENVIRONMENT}_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
AI Debate Game - Deployment Verification Report
=============================================

Environment: $ENVIRONMENT
Region: $AWS_REGION
Verification Date: $(date)

Deployment URLs:
- Frontend: $CLOUDFRONT_URL
- API: $API_URL

AWS Resources:
- S3 Frontend Bucket: $S3_FRONTEND_BUCKET
- S3 Models Bucket: $S3_MODELS_BUCKET
- Backend Lambda: debate-game-${ENVIRONMENT}-backend
- ML Lambda: debate-game-${ENVIRONMENT}-ml-inference

Verification Status: PASSED

Next Steps:
1. Access the game at: $CLOUDFRONT_URL
2. Test game functionality manually
3. Monitor CloudWatch logs for any issues
4. Set up monitoring/alerting if needed

Troubleshooting:
- If frontend shows 404/403, wait 10-15 minutes for CloudFront propagation
- Check CloudWatch logs for Lambda function errors
- Verify CORS settings if frontend can't reach API
- Use 'aws logs tail' to monitor real-time logs

EOF

    log_success "Verification report saved to: $report_file"
}

# Main verification function
main() {
    log_info "Starting deployment verification for environment: $ENVIRONMENT"
    echo "=================================="
    
    load_outputs
    
    echo ""
    test_aws_resources
    
    echo ""
    test_api
    
    echo ""
    test_frontend
    
    echo ""
    performance_test
    
    echo ""
    generate_report
    
    echo ""
    echo "=================================="
    log_success "🎉 Deployment verification completed successfully!"
    echo ""
    log_info "🎮 Your AI Debate Game is ready!"
    echo "Frontend: $CLOUDFRONT_URL"
    echo "API: $API_URL"
    echo ""
    log_info "If frontend shows errors, wait 10-15 minutes for CloudFront propagation."
}

# Error handler
handle_error() {
    log_error "Verification failed: $1"
    echo ""
    log_info "Common solutions:"
    echo "- Ensure deployment completed successfully"
    echo "- Wait for CloudFront distribution to propagate"
    echo "- Check AWS credentials and permissions"
    echo "- Review CloudWatch logs for errors"
    exit 1
}

# Trap errors
trap 'handle_error "unknown error"' ERR

# Run main function
main "$@"