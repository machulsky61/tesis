# API Gateway configuration for the backend

# REST API
resource "aws_api_gateway_rest_api" "main" {
  name        = "${local.prefix}-api"
  description = "AI Debate Game API"

  endpoint_configuration {
    types = ["REGIONAL"]
  }

  tags = local.common_tags
}

# API Gateway deployment
resource "aws_api_gateway_deployment" "main" {
  depends_on = [
    aws_api_gateway_integration.backend_proxy,
    aws_api_gateway_integration.backend_cors,
  ]

  rest_api_id = aws_api_gateway_rest_api.main.id
  stage_name  = var.environment

  # Force redeployment on changes
  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.backend_proxy.id,
      aws_api_gateway_method.backend_proxy.id,
      aws_api_gateway_integration.backend_proxy.id,
      aws_api_gateway_method.backend_cors.id,
      aws_api_gateway_integration.backend_cors.id,
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }

  tags = local.common_tags
}

# Proxy resource for all backend routes
resource "aws_api_gateway_resource" "backend_proxy" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  parent_id   = aws_api_gateway_rest_api.main.root_resource_id
  path_part   = "{proxy+}"
}

# Proxy method
resource "aws_api_gateway_method" "backend_proxy" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  resource_id   = aws_api_gateway_resource.backend_proxy.id
  http_method   = "ANY"
  authorization = "NONE"
}

# Proxy integration
resource "aws_api_gateway_integration" "backend_proxy" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_method.backend_proxy.resource_id
  http_method = aws_api_gateway_method.backend_proxy.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.backend.invoke_arn
}

# CORS for root resource
resource "aws_api_gateway_method" "backend_cors" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  resource_id   = aws_api_gateway_rest_api.main.root_resource_id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "backend_cors" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_method.backend_cors.resource_id
  http_method = aws_api_gateway_method.backend_cors.http_method

  type = "MOCK"

  request_templates = {
    "application/json" = jsonencode({
      statusCode = 200
    })
  }
}

resource "aws_api_gateway_method_response" "backend_cors" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_method.backend_cors.resource_id
  http_method = aws_api_gateway_method.backend_cors.http_method
  status_code = "200"

  response_headers = {
    "Access-Control-Allow-Headers" = true
    "Access-Control-Allow-Methods" = true
    "Access-Control-Allow-Origin"  = true
  }
}

resource "aws_api_gateway_integration_response" "backend_cors" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_method.backend_cors.resource_id
  http_method = aws_api_gateway_method.backend_cors.http_method
  status_code = aws_api_gateway_method_response.backend_cors.status_code

  response_headers = {
    "Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"
    "Access-Control-Allow-Methods" = "'GET,OPTIONS,POST,PUT,DELETE'"
    "Access-Control-Allow-Origin"  = "'*'"
  }
}

# CORS for proxy resource
resource "aws_api_gateway_method" "backend_proxy_cors" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  resource_id   = aws_api_gateway_resource.backend_proxy.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "backend_proxy_cors" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_method.backend_proxy_cors.resource_id
  http_method = aws_api_gateway_method.backend_proxy_cors.http_method

  type = "MOCK"

  request_templates = {
    "application/json" = jsonencode({
      statusCode = 200
    })
  }
}

resource "aws_api_gateway_method_response" "backend_proxy_cors" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_method.backend_proxy_cors.resource_id
  http_method = aws_api_gateway_method.backend_proxy_cors.http_method
  status_code = "200"

  response_headers = {
    "Access-Control-Allow-Headers" = true
    "Access-Control-Allow-Methods" = true
    "Access-Control-Allow-Origin"  = true
  }
}

resource "aws_api_gateway_integration_response" "backend_proxy_cors" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_method.backend_proxy_cors.resource_id
  http_method = aws_api_gateway_method.backend_proxy_cors.http_method
  status_code = aws_api_gateway_method_response.backend_proxy_cors.status_code

  response_headers = {
    "Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"
    "Access-Control-Allow-Methods" = "'GET,OPTIONS,POST,PUT,DELETE'"
    "Access-Control-Allow-Origin"  = "'*'"
  }
}

# API Gateway stage configuration
resource "aws_api_gateway_stage" "main" {
  deployment_id = aws_api_gateway_deployment.main.id
  rest_api_id   = aws_api_gateway_rest_api.main.id
  stage_name    = var.environment

  # Throttling
  throttle_settings {
    rate_limit  = var.api_throttle_rate_limit
    burst_limit = var.api_throttle_burst_limit
  }

  # Access logging
  access_log_destination_arn = aws_cloudwatch_log_group.api_gateway_logs.arn
  access_log_format = jsonencode({
    requestId      = "$context.requestId"
    ip            = "$context.identity.sourceIp"
    caller        = "$context.identity.caller"
    user          = "$context.identity.user"
    requestTime   = "$context.requestTime"
    httpMethod    = "$context.httpMethod"
    resourcePath  = "$context.resourcePath"
    status        = "$context.status"
    protocol      = "$context.protocol"
    responseLength = "$context.responseLength"
  })

  tags = local.common_tags
}

# CloudWatch Log Group for API Gateway
resource "aws_cloudwatch_log_group" "api_gateway_logs" {
  name              = "/aws/apigateway/${local.prefix}"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

# Custom domain (optional)
resource "aws_api_gateway_domain_name" "main" {
  count           = var.enable_custom_domain && var.domain_name != "" ? 1 : 0
  domain_name     = var.domain_name
  certificate_arn = var.certificate_arn
}

resource "aws_api_gateway_base_path_mapping" "main" {
  count       = var.enable_custom_domain && var.domain_name != "" ? 1 : 0
  api_id      = aws_api_gateway_rest_api.main.id
  stage_name  = aws_api_gateway_stage.main.stage_name
  domain_name = aws_api_gateway_domain_name.main[0].domain_name
}