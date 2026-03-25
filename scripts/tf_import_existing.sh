#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# tf_import_existing.sh
#
# Imports pre-existing AWS resources into Terraform state.
# - Every AWS lookup has a hard timeout so it never hangs.
# - Every terraform import runs with a 60-second timeout.
# - Already-in-state resources are silently skipped.
# - Missing resources are silently skipped (Terraform will create them).
# ---------------------------------------------------------------------------
set -uo pipefail   # no -e so individual failures don't stop the script

PROJECT="${PROJECT_NAME:-credit-score}"
REGION="${AWS_REGION:-us-east-1}"

cd terraform

echo "==> Starting resource import (project=$PROJECT, region=$REGION)"

# ── Helper: skip if already in state, import with 60-second timeout ----------
safe_import() {
  local resource="$1"
  local id="$2"

  if [ -z "$id" ] || [ "$id" = "None" ] || [ "$id" = "null" ]; then
    echo "  [skip] $resource — resource not found in AWS"
    return
  fi

  if terraform state show "$resource" &>/dev/null 2>&1; then
    echo "  [skip] $resource — already tracked in state"
    return
  fi

  echo "  [import] $resource  <--  $id"
  # 60-second hard timeout per import so one stuck call can't block everything
  if timeout 60 terraform import "$resource" "$id" 2>&1; then
    echo "  [ok]    $resource"
  else
    echo "  [warn]  $resource import failed or timed out — Terraform will reconcile on apply"
  fi
}

# ── Lookup helpers (each capped at 10 seconds) --------------------------------
aws_lookup() {
  timeout 10 aws "$@" 2>/dev/null || echo ""
}

ACCOUNT_ID=$(aws_lookup sts get-caller-identity --query Account --output text)
VPC_ID=$(aws_lookup ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text --region "$REGION")
echo "    account=$ACCOUNT_ID  vpc=$VPC_ID"

# ── 1. CloudWatch Log Group ---------------------------------------------------
safe_import "aws_cloudwatch_log_group.api" "/ecs/${PROJECT}-api"

# ── 2. IAM Roles --------------------------------------------------------------
safe_import "aws_iam_role.sagemaker_role"     "${PROJECT}-sagemaker-role"
safe_import "aws_iam_role.ecs_execution_role" "${PROJECT}-ecs-execution-role"
safe_import "aws_iam_role.ecs_task_role"      "${PROJECT}-ecs-task-role"

# ── 3. IAM Policy -------------------------------------------------------------
POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT}-sagemaker-invoke"
POLICY_EXISTS=$(aws_lookup iam get-policy --policy-arn "$POLICY_ARN" \
  --query "Policy.Arn" --output text)
safe_import "aws_iam_policy.sagemaker_invoke" "$POLICY_EXISTS"

# ── 4. Security Groups --------------------------------------------------------
ALB_SG=$(aws_lookup ec2 describe-security-groups \
  --filters "Name=group-name,Values=${PROJECT}-alb-sg" "Name=vpc-id,Values=$VPC_ID" \
  --query "SecurityGroups[0].GroupId" --output text --region "$REGION")
safe_import "aws_security_group.alb" "$ALB_SG"

ECS_SG=$(aws_lookup ec2 describe-security-groups \
  --filters "Name=group-name,Values=${PROJECT}-ecs-sg" "Name=vpc-id,Values=$VPC_ID" \
  --query "SecurityGroups[0].GroupId" --output text --region "$REGION")
safe_import "aws_security_group.ecs" "$ECS_SG"

# ── 5. ALB -------------------------------------------------------------------
ALB_ARN=$(aws_lookup elbv2 describe-load-balancers \
  --names "${PROJECT}-alb" \
  --query "LoadBalancers[0].LoadBalancerArn" --output text --region "$REGION")
safe_import "aws_lb.main" "$ALB_ARN"

# ── 6. Target Group ----------------------------------------------------------
TG_ARN=$(aws_lookup elbv2 describe-target-groups \
  --names "${PROJECT}-tg" \
  --query "TargetGroups[0].TargetGroupArn" --output text --region "$REGION")
safe_import "aws_lb_target_group.api" "$TG_ARN"

# ── 7. ALB Listener (only if ALB found) --------------------------------------
if [ -n "$ALB_ARN" ] && [ "$ALB_ARN" != "None" ]; then
  LISTENER_ARN=$(aws_lookup elbv2 describe-listeners \
    --load-balancer-arn "$ALB_ARN" \
    --query "Listeners[?Port==\`80\`].ListenerArn | [0]" \
    --output text --region "$REGION")
  safe_import "aws_lb_listener.http" "$LISTENER_ARN"
fi

# ── 8. ECS Cluster -----------------------------------------------------------
CLUSTER_ARN=$(aws_lookup ecs describe-clusters \
  --clusters "${PROJECT}-cluster" \
  --query "clusters[?status=='ACTIVE'].clusterArn | [0]" \
  --output text --region "$REGION")
safe_import "aws_ecs_cluster.main" "$CLUSTER_ARN"

# ── 9. ECR Repositories ------------------------------------------------------
ECR_SM=$(aws_lookup ecr describe-repositories \
  --repository-names "${PROJECT}-sagemaker-model" \
  --query "repositories[0].repositoryName" --output text --region "$REGION")
safe_import "aws_ecr_repository.sagemaker_model" "$ECR_SM"

ECR_API=$(aws_lookup ecr describe-repositories \
  --repository-names "${PROJECT}-api" \
  --query "repositories[0].repositoryName" --output text --region "$REGION")
safe_import "aws_ecr_repository.api" "$ECR_API"

echo ""
echo "==> Import step complete — proceeding to terraform apply."
