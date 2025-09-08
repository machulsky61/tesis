# Lambda function for the backend API

# IAM role for Lambda execution
resource "aws_iam_role" "lambda_execution_role" {
  name = "${local.prefix}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# IAM policy for Lambda basic execution
resource "aws_iam_role_policy" "lambda_basic_execution" {
  name = "${local.prefix}-lambda-basic-execution"
  role = aws_iam_role.lambda_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${local.region}:${local.account_id}:*"
      }
    ]
  })
}

# IAM policy for S3 access (for models)
resource "aws_iam_role_policy" "lambda_s3_access" {
  name = "${local.prefix}-lambda-s3-access"
  role = aws_iam_role.lambda_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = [
          aws_s3_bucket.models.arn,
          "${aws_s3_bucket.models.arn}/*"
        ]
      }
    ]
  })
}

# Lambda function
resource "aws_lambda_function" "backend" {
  filename         = "${path.module}/../../backend/deployment.zip"
  function_name    = "${local.prefix}-backend"
  role            = aws_iam_role.lambda_execution_role.arn
  handler         = "app.main.handler"
  runtime         = var.lambda_runtime
  timeout         = var.lambda_timeout
  memory_size     = var.lambda_memory_size

  # This will be updated by deployment scripts
  source_code_hash = filebase64sha256("${path.module}/../../backend/deployment.zip")

  environment {
    variables = {
      ENVIRONMENT = var.environment
      AWS_REGION  = var.aws_region
      MODEL_BUCKET = aws_s3_bucket.models.bucket
      LOG_LEVEL   = var.environment == "production" ? "INFO" : "DEBUG"
    }
  }

  depends_on = [
    aws_iam_role_policy.lambda_basic_execution,
    aws_cloudwatch_log_group.lambda_logs,
  ]

  tags = local.common_tags
}

# Lambda function for ML inference (if needed)
resource "aws_lambda_function" "ml_inference" {
  filename         = "${path.module}/../../backend/ml-deployment.zip"
  function_name    = "${local.prefix}-ml-inference"
  role            = aws_iam_role.lambda_execution_role.arn
  handler         = "inference.handler"
  runtime         = var.lambda_runtime
  timeout         = 60  # Longer timeout for ML inference
  memory_size     = 2048  # More memory for ML models

  source_code_hash = filebase64sha256("${path.module}/../../backend/ml-deployment.zip")

  environment {
    variables = {
      ENVIRONMENT = var.environment
      MODEL_BUCKET = aws_s3_bucket.models.bucket
    }
  }

  depends_on = [
    aws_iam_role_policy.lambda_basic_execution,
    aws_cloudwatch_log_group.lambda_ml_logs,
  ]

  tags = local.common_tags
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${local.prefix}-backend"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "lambda_ml_logs" {
  name              = "/aws/lambda/${local.prefix}-ml-inference"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

# Lambda provisioned concurrency (for production)
resource "aws_lambda_provisioned_concurrency_config" "backend" {
  count                             = var.enable_reserved_capacity && var.environment == "production" ? 1 : 0
  function_name                     = aws_lambda_function.backend.function_name
  provisioned_concurrent_executions = var.reserved_capacity_amount
  qualifier                         = aws_lambda_function.backend.version

  tags = local.common_tags
}

# Lambda permissions for API Gateway
resource "aws_lambda_permission" "api_gateway_backend" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.backend.function_name
  principal     = "apigateway.amazonaws.com"

  source_arn = "${aws_api_gateway_rest_api.main.execution_arn}/*/*"
}

resource "aws_lambda_permission" "api_gateway_ml" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ml_inference.function_name
  principal     = "apigateway.amazonaws.com"

  source_arn = "${aws_api_gateway_rest_api.main.execution_arn}/*/*"
}