# Output values for Terraform deployment

output "api_gateway_url" {
  description = "URL of the API Gateway"
  value       = aws_api_gateway_deployment.main.invoke_url
}

output "api_gateway_stage_url" {
  description = "URL of the API Gateway stage"
  value       = "${aws_api_gateway_deployment.main.invoke_url}/${var.environment}"
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID"
  value       = aws_cloudfront_distribution.frontend.id
}

output "cloudfront_domain_name" {
  description = "CloudFront distribution domain name"
  value       = aws_cloudfront_distribution.frontend.domain_name
}

output "cloudfront_url" {
  description = "CloudFront URL"
  value       = "https://${aws_cloudfront_distribution.frontend.domain_name}"
}

output "custom_domain_url" {
  description = "Custom domain URL (if configured)"
  value       = var.enable_custom_domain && var.domain_name != "" ? "https://${var.domain_name}" : ""
}

output "s3_frontend_bucket" {
  description = "S3 bucket name for frontend"
  value       = aws_s3_bucket.frontend.bucket
}

output "s3_frontend_website_url" {
  description = "S3 website URL for frontend"
  value       = "http://${aws_s3_bucket.frontend.bucket}.s3-website-${var.aws_region}.amazonaws.com"
}

output "s3_models_bucket" {
  description = "S3 bucket name for models"
  value       = aws_s3_bucket.models.bucket
}

output "lambda_backend_function_name" {
  description = "Lambda function name for backend"
  value       = aws_lambda_function.backend.function_name
}

output "lambda_backend_function_arn" {
  description = "Lambda function ARN for backend"
  value       = aws_lambda_function.backend.arn
}

output "lambda_ml_function_name" {
  description = "Lambda function name for ML inference"
  value       = aws_lambda_function.ml_inference.function_name
}

output "lambda_ml_function_arn" {
  description = "Lambda function ARN for ML inference"
  value       = aws_lambda_function.ml_inference.arn
}

# Environment information
output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "account_id" {
  description = "AWS account ID"
  value       = local.account_id
}

# Resource ARNs for external integrations
output "api_gateway_arn" {
  description = "API Gateway ARN"
  value       = aws_api_gateway_rest_api.main.arn
}

output "cloudfront_arn" {
  description = "CloudFront distribution ARN"
  value       = aws_cloudfront_distribution.frontend.arn
}

# Logging information
output "lambda_log_group_name" {
  description = "CloudWatch log group name for Lambda"
  value       = aws_cloudwatch_log_group.lambda_logs.name
}

output "api_gateway_log_group_name" {
  description = "CloudWatch log group name for API Gateway"
  value       = aws_cloudwatch_log_group.api_gateway_logs.name
}

# Cost tracking tags
output "common_tags" {
  description = "Common tags applied to all resources"
  value       = local.common_tags
}

# Deployment information
output "deployment_summary" {
  description = "Summary of deployed resources"
  value = {
    frontend_url           = "https://${aws_cloudfront_distribution.frontend.domain_name}"
    api_url               = "${aws_api_gateway_deployment.main.invoke_url}/${var.environment}"
    custom_domain         = var.enable_custom_domain && var.domain_name != "" ? "https://${var.domain_name}" : "Not configured"
    environment           = var.environment
    region                = var.aws_region
    lambda_memory         = var.lambda_memory_size
    lambda_timeout        = var.lambda_timeout
    api_throttle_rate     = var.api_throttle_rate_limit
    api_throttle_burst    = var.api_throttle_burst_limit
    s3_versioning_enabled = var.enable_s3_versioning
    waf_enabled           = var.enable_waf
    custom_domain_enabled = var.enable_custom_domain
  }
}