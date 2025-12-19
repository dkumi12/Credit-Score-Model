provider "aws" {
  region = "us-east-1"
}

variable "model_s3_url" {
  description = "The S3 URL of the model-v2.tar.gz"
  type        = string
}

# 1. IAM Role for SageMaker to access S3
resource "aws_iam_role" "sagemaker_role" {
  name = "credit-score-project-role-v3"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "s3_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "sm_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# 2. SageMaker Model Definition
resource "aws_sagemaker_model" "model" {
  name               = "credit-score-model-v3"
  execution_role_arn = aws_iam_role.sagemaker_role.arn

  primary_container {
    image          = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    model_data_url = var.model_s3_url
    environment = {
      SAGEMAKER_PROGRAM          = "sagemaker_entry.py"
      SAGEMAKER_SUBMIT_DIRECTORY = var.model_s3_url
    }
  }
}

# 3. Endpoint Configuration
resource "aws_sagemaker_endpoint_configuration" "config" {
  name = "credit-score-config-v3"
  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.model.name
    initial_instance_count = 1
    instance_type          = "ml.t2.medium"
  }
}

# 4. The Live Endpoint
resource "aws_sagemaker_endpoint" "endpoint" {
  name                 = "credit-score-endpoint-v3"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.config.name
}

output "endpoint_name" {
  value = aws_sagemaker_endpoint.endpoint.name
}
