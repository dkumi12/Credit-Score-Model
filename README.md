# 💳 Credit Score Prediction Model

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-FF9900?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/sagemaker/)
[![Terraform](https://img.shields.io/badge/Terraform-7B42BC?logo=terraform&logoColor=white)](https://terraform.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production ML system for credit risk prediction with AWS SageMaker deployment and Terraform infrastructure-as-code**

Machine learning model predicting credit score classification (Good/Bad) using Random Forest with comprehensive feature engineering and cloud deployment pipeline.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Model Performance](#-model-performance)
- [Architecture](#-architecture)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [AWS SageMaker Deployment](#-aws-sagemaker-deployment)
- [Terraform Infrastructure](#-terraform-infrastructure)
- [API Usage](#-api-usage)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)

---

## 🎯 Overview

Credit Score Model is a production-grade machine learning system that predicts credit risk based on demographic and financial features. Built with MLOps best practices including automated deployment, infrastructure-as-code, and comprehensive testing.

### Key Achievements

- **Random Forest Classifier** with optimized hyperparameters
- **AWS SageMaker Endpoint** deployment for scalable inference
- **Terraform Infrastructure** for reproducible cloud deployments
- **RESTful API** with FastAPI for real-time predictions
- **Comprehensive Testing** with unit tests and integration tests

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Algorithm** | Random Forest Classifier |
| **Features** | 9 demographic + financial variables |
| **Target Classes** | Good / Bad Credit Score |
| **Deployment** | AWS SageMaker Real-time Endpoint |

### Input Features

- Age
- Gender  
- Education Level
- Marital Status
- Number of Children
- Annual Income
- Home Ownership Status
- Employment Status
- Loan Amount (if applicable)

---

## 🏗️ Architecture

### System Design

```
┌──────────────────────────────────────────────────────────────┐
│                    AWS Cloud Environment                     │
│                                                              │
│  ┌────────────┐      ┌──────────────┐      ┌─────────────┐ │
│  │     S3     │─────▶│  SageMaker   │─────▶│  Endpoint   │ │
│  │            │      │    Model     │      │  (Real-time)│ │
│  │ model.pkl  │      │              │      │             │ │
│  │ scaler.pkl │      │  Scikit-learn│      │ ml.t2.medium│ │
│  └────────────┘      │  Container   │      └─────────────┘ │
│                      └──────────────┘              │        │
│                                                    │        │
│                                            ┌───────▼──────┐ │
│                                            │  API Gateway │ │
│                                            │  (Optional)  │ │
│                                            └──────────────┘ │
└──────────────────────────────────────────────────────────────┘
           │
           │ Managed by Terraform
           ▼
    ┌─────────────┐
    │ terraform/  │
    │  main.tf    │
    └─────────────┘
```

### Deployment Flow

1. **Model Training:** Train Random Forest locally with GridSearchCV
2. **Model Packaging:** Serialize model and scaler to .pkl files
3. **S3 Upload:** Upload model artifacts to S3 bucket
4. **Terraform Apply:** Create SageMaker model, endpoint config, and endpoint
5. **API Integration:** Connect endpoint to FastAPI for predictions

---

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/dkumi12/Credit-Score-Model.git
cd Credit-Score-Model
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model**
```bash
python Src/api.py
```

4. **Run API locally**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## ☁️ AWS SageMaker Deployment

### Prerequisites
- AWS CLI configured
- Terraform installed
- S3 bucket for model artifacts

### Deployment Steps

1. **Package the model for SageMaker**
```bash
# Model and dependencies are packaged in Models/
tar -czf model.tar.gz -C Models credit_scoring_model.pkl scaler.pkl
```

2. **Upload to S3**
```bash
aws s3 cp model.tar.gz s3://dkumi12-credit-score-project/
```

3. **Deploy with Terraform**
```bash
cd terraform

# Initialize Terraform
terraform init

# Review deployment plan
terraform plan -var="model_s3_url=s3://dkumi12-credit-score-project/model.tar.gz"

# Deploy infrastructure
terraform apply -var="model_s3_url=s3://dkumi12-credit-score-project/model.tar.gz"
```

4. **Get endpoint details**
```bash
aws sagemaker describe-endpoint --endpoint-name credit-score-project-endpoint
```

---

## 📡 API Usage

### Local Prediction

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "age": 35,
        "gender": "Male",
        "education": "Bachelor's",
        "marital_status": "Married",
        "num_children": 2,
        "income": 75000,
        "home_ownership": "Own",
        "employment_status": "Employed"
    }
)

print(response.json())
# Output: {"prediction": "Good", "confidence": 0.87, "risk_score": 0.13}
```

### SageMaker Endpoint Prediction

```python
import boto3
import json

sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

payload = {
    "instances": [{
        "age": 35,
        "gender": "Male",
        "education": "Bachelor's",
        "marital_status": "Married",
        "num_children": 2,
        "income": 75000,
        "home_ownership": "Own",
        "employment_status": "Employed"
    }]
}

response = sagemaker_runtime.invoke_endpoint(
    EndpointName='credit-score-project-endpoint',
    ContentType='application/json',
    Body=json.dumps(payload)
)

result = json.loads(response['Body'].read().decode())
print(result)
```

---

## 🗂️ Project Structure

```
Credit-Score-Model/
├── README.md
├── requirements.txt
├── app.py                      # FastAPI application
├── Src/
│   ├── api.py                  # Model training pipeline
│   ├── sagemaker_entry.py     # SageMaker inference script
│   ├── utils.py               # Helper functions
│   └── __init__.py
├── Data/
│   └── Credit Score Classification Dataset.csv
├── Models/
│   ├── credit_scoring_model.pkl
│   └── scaler.pkl
├── terraform/
│   └── main.tf                # Infrastructure as Code
├── tests/
│   ├── test_utils.py
│   └── __init__.py
└── credit-score-infographic/  # React visualization dashboard
```

---

## 💻 Tech Stack

**Machine Learning:**
- Scikit-learn (Random Forest)
- Pandas (Data preprocessing)
- NumPy (Numerical operations)
- Joblib (Model serialization)

**Cloud Infrastructure:**
- AWS SageMaker (Model hosting)
- AWS S3 (Model storage)
- Terraform (Infrastructure as Code)

**API & Deployment:**
- FastAPI (REST API)
- Uvicorn (ASGI server)
- Docker (Containerization)

---

## 🔐 Model Details

### Feature Engineering

- **Categorical Encoding:** One-Hot Encoding for gender, education, marital status
- **Numerical Scaling:** StandardScaler for age, income, num_children
- **Feature Selection:** Correlation analysis and feature importance ranking

### Hyperparameter Tuning

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

Optimized using GridSearchCV with 5-fold cross-validation.

---

## 📈 Monitoring & Observability

### SageMaker Metrics
- Invocation count and latency
- Model drift detection (planned)
- Cost tracking per prediction

### Logging
```bash
# View SageMaker endpoint logs
aws logs tail /aws/sagemaker/Endpoints/credit-score-project-endpoint --follow
```

---

## 💰 Cost Optimization

- **Instance Type:** ml.t2.medium (cost-effective for low-traffic)
- **Auto-scaling:** Configure based on invocation metrics
- **Endpoint Lifecycle:** Stop endpoint when not in use to save costs

```bash
# Stop endpoint to save costs
aws sagemaker delete-endpoint --endpoint-name credit-score-project-endpoint
```

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Test SageMaker deployment
python test_deployment.py
```

---

## 🔄 CI/CD Pipeline

Automated deployment workflow:
1. Code push to `main` branch
2. GitHub Actions runs tests
3. Model artifacts uploaded to S3
4. Terraform applies infrastructure changes
5. SageMaker endpoint updated with new model

---

## 📝 Future Enhancements

- [ ] Add model versioning with MLflow
- [ ] Implement A/B testing for model variants
- [ ] Create React dashboard for credit score visualization
- [ ] Add explainability with SHAP values
- [ ] Integrate with loan application workflow

---

## 🤝 Contributing

This is a portfolio project. For questions or collaboration opportunities, reach out via:

- **Email:** 12dkumi@gmail.com
- **LinkedIn:** [david-osei-kumi](https://linkedin.com/in/david-osei-kumi)
- **GitHub:** [@dkumi12](https://github.com/dkumi12)

---

## 📄 License

MIT License - See LICENSE file for details

---

**Built by:** David Osei Kumi  
**Tech:** AWS SageMaker • Random Forest • Terraform • FastAPI  
**Status:** Production-ready ML deployment with cloud infrastructure
