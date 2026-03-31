# 💳 Credit Score Classifier

[![CI/CD](https://github.com/dkumi12/Credit-Score-Model/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/dkumi12/Credit-Score-Model/actions/workflows/ci-cd.yml)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked%20on%20DagsHub-0194E2?logo=mlflow&logoColor=white)](https://dagshub.com/dkumi12/Credit-Score-Model.mlflow)
[![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](https://python.org)
[![AWS](https://img.shields.io/badge/AWS-SageMaker%20%7C%20ECS%20%7C%20API%20Gateway-FF9900?logo=amazon-aws&logoColor=white)](https://aws.amazon.com)
[![Terraform](https://img.shields.io/badge/IaC-Terraform-7B42BC?logo=terraform&logoColor=white)](https://terraform.io)
[![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> End-to-end MLOps project — Random Forest credit risk classifier trained with scikit-learn and MLflow, containerised with Docker, and deployed on AWS using SageMaker, ECS Fargate, and API Gateway, fully automated via GitHub Actions CI/CD.

**Live demo:** `http://credit-score-alb-<id>.us-east-1.elb.amazonaws.com`
**API endpoint:** `https://4p97tzuzvd.execute-api.us-east-1.amazonaws.com`

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Model](#-model)
- [API Reference](#-api-reference)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Infrastructure](#-infrastructure)
- [Local Development](#-local-development)
- [Tech Stack](#-tech-stack)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  EXPERIMENT LAYER                                                │
│                                                                  │
│  python Src/train.py                                            │
│       │                                                          │
│       ├── logs params, metrics, charts ──▶ DagsHub (MLflow)    │
│       ├── registers model ──────────────▶ MLflow Model Registry │
│       └── saves .pkl ───────────────────▶ Models/              │
└──────────────────────────┬──────────────────────────────────────┘
                           │  model artifact
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  SERVING LAYER                                                   │
│                                                                  │
│  Browser                                                         │
│     │                                                            │
│     ▼                                                            │
│  Streamlit UI (ECS Fargate)                                     │
│     │  POST /predict                                             │
│     ▼                                                            │
│  API Gateway → ALB (path-based routing)                         │
│                  │                                               │
│                  ├── /predict /health /docs                      │
│                  │      ▼                                        │
│                  │   FastAPI (ECS Fargate)                       │
│                  │      │  InvokeEndpoint                        │
│                  │      ▼                                        │
│                  │   SageMaker Endpoint                          │
│                  │   (custom Docker — scikit-learn 1.2.2)        │
│                  │                                               │
│                  └── /* ──▶ Streamlit Frontend (ECS Fargate)    │
└─────────────────────────────────────────────────────────────────┘
```

**Live experiment tracking:** [dagshub.com/dkumi12/Credit-Score-Model.mlflow](https://dagshub.com/dkumi12/Credit-Score-Model.mlflow)

### Why custom Docker for SageMaker?
The built-in SageMaker scikit-learn container caused dependency conflicts. Baking the exact training environment (`scikit-learn==1.2.2`, `numpy<2.0`) into a custom image eliminates version mismatches entirely.

---

## 📁 Project Structure

```
Credit-Score-Model/
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # Full CI/CD pipeline
├── Data/
│   └── Credit Score Classification Dataset.csv
├── Models/
│   ├── credit_scoring_model.pkl  # Trained sklearn Pipeline
│   └── scaler.pkl
├── Src/
│   ├── api.py                  # FastAPI — /predict /health
│   ├── train.py                # Training script — logs to DagsHub via MLflow
│   └── utils.py                # Preprocessing helpers
├── docker/
│   ├── Dockerfile.sagemaker    # Custom SageMaker inference container
│   ├── Dockerfile.api          # FastAPI container (ECS)
│   ├── Dockerfile.frontend     # Streamlit container (ECS)
│   ├── serve.py                # SageMaker Flask server (/ping /invocations)
│   └── requirements-api.txt
├── terraform/
│   ├── main.tf                 # Provider + S3 backend
│   ├── variables.tf
│   ├── ecr.tf                  # ECR repositories
│   ├── iam.tf                  # Roles for SageMaker + ECS
│   ├── sagemaker.tf            # Model, endpoint config, endpoint
│   ├── networking.tf           # VPC, SGs, ALB, target groups
│   ├── ecs.tf                  # Cluster, task defs, services
│   ├── api_gateway.tf          # HTTP API + integration
│   └── outputs.tf
├── scripts/
│   ├── tf_import_existing.sh   # Import pre-existing AWS resources into state
│   └── cleanup_aws.sh          # Tear down AWS resources
├── tests/
│   └── test_utils.py
├── app.py                      # Streamlit frontend
├── requirements.txt            # Frontend deps only
└── requirements-train.txt      # Training deps (MLflow, DagsHub, sklearn)
```

---

## 🤖 Model

| Property | Detail |
|---|---|
| Algorithm | Random Forest Classifier (sklearn Pipeline) |
| Preprocessing | `ColumnTransformer` — `StandardScaler` (numeric) + `OneHotEncoder` (categorical) |
| Target | Credit Score: **Low / Average / High** |
| Training tool | MLflow (experiment tracking) |
| Serialisation | `joblib` → `credit_scoring_model.pkl` |

### Input features

| Feature | Type |
|---|---|
| Age | Numeric |
| Gender | Categorical |
| Annual Income | Numeric |
| Education Level | Categorical |
| Marital Status | Categorical |
| Number of Children | Numeric |
| Home Ownership | Categorical |

---

## 📡 API Reference

**Base URL:** `https://4p97tzuzvd.execute-api.us-east-1.amazonaws.com`

### `GET /health`
```json
{ "status": "healthy", "endpoint": "credit-score-endpoint" }
```

### `POST /predict`

**Request:**
```json
{
  "age": 35,
  "gender": "Male",
  "income": 75000,
  "education": "Bachelor's Degree",
  "marital_status": "Single",
  "num_children": 0,
  "home_ownership": "Rented"
}
```

**Response:**
```json
{ "credit_score": "High" }
```

**Example — curl:**
```bash
curl -X POST https://4p97tzuzvd.execute-api.us-east-1.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"gender":"Male","income":75000,"education":"Bachelor'\''s Degree","marital_status":"Single","num_children":0,"home_ownership":"Rented"}'
```

---

## 🔄 CI/CD Pipeline

Push to `main` triggers a 4-job GitHub Actions workflow:

```
test ──▶ build-and-push ──▶ deploy
 │              │               │
 │    Build 3 Docker images     │
 │    Push to ECR               │
 │                       Terraform apply:
 │                       - SageMaker endpoint (new image)
 │                       - ECS services (API + frontend)
 │                       - API Gateway routes
 │
security (runs in parallel with test)
```

**Required GitHub Secrets:**

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user with SageMaker, ECS, ECR, ALB, API Gateway permissions |
| `AWS_SECRET_ACCESS_KEY` | Corresponding secret key |

**Average pipeline duration:** ~25–35 min (SageMaker endpoint creation dominates)

---

## 🏛️ Infrastructure

All AWS resources are managed by Terraform with state stored in S3.

| Resource | Details |
|---|---|
| ECR | 3 repositories — sagemaker-model, api, frontend |
| SageMaker | Custom Docker endpoint on `ml.m5.large` |
| ECS Cluster | Fargate — 2 services (API + frontend) |
| ALB | Path-based routing — `/predict` → API, `/*` → frontend |
| API Gateway | HTTP API → ALB integration |
| CloudWatch | Log groups for both ECS services |
| S3 | Terraform state bucket (per-account, versioned, encrypted) |

**Deploy from scratch:**
```bash
# 1. Add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to GitHub Secrets
# 2. Push to main — the pipeline handles everything
git push origin main
```

**Tear down:**
```bash
bash scripts/cleanup_aws.sh
cd terraform && terraform destroy
```

---

## 🧪 Training & Experiment Tracking

All training runs are tracked on DagsHub via MLflow:
[dagshub.com/dkumi12/Credit-Score-Model.mlflow](https://dagshub.com/dkumi12/Credit-Score-Model.mlflow)

**What gets logged per run:**
- Parameters: algorithm, n_estimators, max_depth, test_size, preprocessor
- Metrics: accuracy, precision, recall, F1, ROC-AUC
- Artifacts: confusion matrix, feature importance chart, classification report
- Model: registered in MLflow Model Registry as `CreditScoringModel`

**Run training locally:**
```bash
pip install -r requirements-train.txt
python Src/train.py
```

On first run DagsHub will ask you to authenticate — follow the prompt and paste your DagsHub token. After that, open the DagsHub link to see your experiments live.

---

## 💻 Local Development

```bash
git clone https://github.com/dkumi12/Credit-Score-Model.git
cd Credit-Score-Model

# Frontend only (points at live API by default)
pip install -r requirements.txt
streamlit run app.py

# Run tests
pip install pytest
pytest tests/
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn 1.2.2, pandas, numpy, MLflow |
| Model serving | Custom Docker + Flask on SageMaker |
| API | FastAPI + Uvicorn on ECS Fargate |
| Frontend | Streamlit on ECS Fargate |
| Infrastructure | Terraform, AWS (SageMaker, ECS, ECR, ALB, API Gateway) |
| CI/CD | GitHub Actions + Docker BuildX (layer caching) |
| Observability | CloudWatch Logs |

---

## 🤝 Contact

**David Osei Kumi**
[GitHub @dkumi12](https://github.com/dkumi12) · [LinkedIn](https://linkedin.com/in/david-osei-kumi) · 12dkumi@gmail.com

---

*MIT License*
