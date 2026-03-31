# Credit Score Classifier — Project Report

**Author:** David Osei Kumi
**Date:** March 2026
**Repo:** https://github.com/dkumi12/Credit-Score-Model
**Discipline:** MLOps / AIOps / CloudOps

---

## 1. What This Project Is

A **production-grade credit risk classifier** that takes a customer's demographic and financial profile and predicts their credit score category: **Low / Average / High**.

The project is not just a model. It is a full MLOps system — covering model serving, containerisation, infrastructure-as-code, CI/CD automation, and cloud deployment. The goal was to demonstrate how a trained ML model moves from a Jupyter notebook to a live, scalable, observable production endpoint on AWS.

**Key disciplines demonstrated:**
- MLOps — model lifecycle, versioning, reproducible deployment
- AIOps — automated inference pipeline, real-time prediction serving
- CloudOps — IaC with Terraform, CI/CD with GitHub Actions, containerisation with Docker

---

## 2. Project History

This project has been through multiple iterations over time. The version documented here represents the most complete rebuild — starting from a working notebook and trained model, and building out the full production deployment stack from scratch.

### State of the repo at the start of this rebuild

- A trained `credit_scoring_model.pkl` (sklearn Pipeline, scikit-learn 1.2.2, Random Forest)
- A `scaler.pkl` for numerical feature standardisation
- A basic `Src/api.py` (FastAPI) and `app.py` (Streamlit)
- A `terraform/main.tf` with a single file attempting to deploy to SageMaker
- A React infographic directory (`credit-score-infographic/`) that was disconnected
- A `sagemaker_entry.py` inference script that was outdated

**The core problem:** Running `terraform apply` crashed immediately with package dependency errors. The built-in SageMaker scikit-learn container used a different library version than the model was trained on, causing pickle deserialisation failures. This was the starting point for the rebuild.

---

## 3. Model — Final Production Version

| Property | Detail |
|---|---|
| Algorithm | Logistic Regression |
| Framework | scikit-learn |
| Pipeline | `StandardScaler` → `LogisticRegression(class_weight='balanced')` |
| Training tracking | MLflow → DagsHub |
| Target variable | Binary default label (0 = No Default, 1 = Default) |
| Dataset | 141,259 rows — Lending Club real loan data |
| Default rate | 19.8% (27,926 defaults / 141,259 total) |
| Serialisation | joblib → `credit_scoring_model.pkl` |

### Input features

| Feature | Description |
|---|---|
| loan_amnt | Loan amount requested |
| term | Loan term in months (36 / 60) |
| int_rate | Interest rate assigned by lender |
| installment | Monthly installment amount |
| grade | Lending Club loan grade (A–G encoded) |
| sub_grade | Sub-grade within grade tier |
| emp_length | Employment length in years |
| home_ownership | Home ownership status |
| annual_inc | Annual income |
| verification_status | Income verification status |
| purpose | Purpose of the loan |
| dti | Debt-to-income ratio |
| delinq_2yrs | Delinquencies in past 2 years |
| inq_last_6mths | Credit inquiries in last 6 months |
| open_acc | Number of open credit accounts |
| pub_rec | Number of public records |
| revol_bal | Revolving credit balance |
| revol_util | Revolving utilisation rate |
| total_acc | Total number of credit accounts |

All features are numeric — preprocessing is a single `StandardScaler` transform.

---

## 4. MLflow Experiment Tracking — The Full Journey

All experiments are publicly visible at:
**https://dagshub.com/dkumi12/Credit-Score-Model.mlflow**

The experiment log tells the honest story of how this model was developed — including the mistakes, the discoveries, and the reasoning behind every decision.

---

### Phase 1 — First Dataset (165 rows, German Credit)

**What I did:** Took the original dataset from the notebook and ran 3 experiments varying `n_estimators` (50, 100, 200) on a Random Forest.

**Results:**
| Run | n_estimators | Accuracy | CV Mean |
|---|---|---|---|
| random-forest | 50 | 96.9% | ~0.97 |
| random-forest | 100 | 96.9% | ~0.97 |
| random-forest | 200 | 96.9% | ~0.97 |

**What I noticed:** All three runs returned identical results. This wasn't a logging error — the model genuinely couldn't tell the difference between 50 and 200 trees.

**Why it happened:** 165 rows is a tiny dataset. With that few samples, the signal is fully captured at 50 trees. More trees don't add information — there isn't enough data to find new patterns. The model had hit the ceiling of what this dataset could offer, not the ceiling of the algorithm.

**Lesson:** More model complexity on small data does not improve results. The bottleneck was data, not algorithm.

---

### Phase 2 — Second Dataset (1,000 rows, German Credit extended) — Target Leakage Discovered

**What I did:** Upgraded to a 1,000-row version of the German Credit dataset. Ran Logistic Regression, Random Forest, and XGBoost as a proper algorithm comparison.

**Results:**
| Algorithm | Accuracy | F1 | CV Mean |
|---|---|---|---|
| LogisticRegression | 98.5% | 0.985 | 0.968 |
| RandomForest | 99.0% | 0.990 | 0.978 |
| XGBoost | **100.0%** | **1.000** | **0.985** |

**Why the 1.000 score was a red flag, not a win:**

A perfect score on a real-world problem is almost never legitimate. It means one of three things: the test set leaked into training, the model is memorising the training data, or — what happened here — the target variable was derived directly from the features.

The target label (`Credit Score: Low / Average / High`) was generated using a rule:

```
if savings + checking_account + credit_burden > threshold → High
elif savings + checking_account + credit_burden > lower_threshold → Average
else → Low
```

Then the model was trained on those same features to predict that label. It wasn't learning creditworthiness — it was learning to reverse-engineer a formula that was written into the data generation step. Of course it got 100%. It had seen the answer key.

This is called **target leakage** — one of the most common and most dangerous mistakes in machine learning because the model appears to work perfectly in training and fails completely in production.

The low ROC-AUC (0.35–0.38) was the honest signal. AUC below 0.5 means the model performs worse than random for actual classification — the only reason accuracy was high was because the majority class dominated and the derived labels were perfectly consistent.

**Lesson:** A perfect score is a warning sign. Always ask where the labels came from. If the target is derived from the same features used for training, you don't have a model — you have a lookup table.

---

### Phase 3 — Final Dataset (141,259 rows, Lending Club) — Real Results

**What I did:** Switched to the Lending Club clean dataset from HuggingFace. 141,259 real loan applications with an actual `default` label (0/1) assigned by historical outcome — did the borrower actually default or not. No derivation, no formula — real ground truth.

**Results:**
| Algorithm | AUC | PR-AUC | F1 | CV Mean |
|---|---|---|---|---|
| LogisticRegression | **0.702** | **0.358** | **0.420** | **0.706 ± 0.006** |
| RandomForest | 0.692 | 0.347 | 0.091 | 0.696 ± 0.004 |
| XGBoost | 0.681 | 0.337 | 0.403 | 0.683 ± 0.004 |

**Winner: Logistic Regression — and why that makes sense:**

Logistic Regression beating Random Forest and XGBoost on credit data is not a surprise to anyone who has worked in credit risk. There are specific reasons:

1. **The features are already curated.** Lending Club's own risk team assigned `grade` and `sub_grade` to each loan — these are already engineered representations of creditworthiness. The relationship between these features and default probability is approximately linear by design. LR captures that directly.

2. **Class imbalance.** With 19.8% default rate, tree-based models tend to over-split on the majority class and produce poor probability calibration. LR with `class_weight='balanced'` adjusts the decision boundary explicitly for imbalanced data.

3. **Overfitting on tabular data.** Random Forest with 200 trees on 141k rows can overfit on noise. LR has no capacity for that kind of overfitting — it finds the single best linear boundary and stops.

4. **Interpretability.** LR produces feature coefficients — you can rank which variables drive default probability most. This matters in credit risk specifically: regulators in many jurisdictions require explainable models. A black box tree ensemble fails regulatory review. A logistic regression passes.

**Why AUC 0.70 is a good result:**

Credit default prediction is a genuinely hard problem. Even commercial credit bureaus with access to thousands of features — full credit report, payment history, tradeline data, inquiry patterns — produce models in the 0.70–0.85 AUC range. With 19 features from a cleaned public dataset, 0.70 AUC is a legitimate, defensible result.

It is a real model. It would give a real bank real signal about which loan applicants are higher risk. That is more valuable than a 99% accuracy number that is meaningless.

---

### Model Registry

The winning Logistic Regression model was registered in MLflow Model Registry on DagsHub as `CreditScoringModel` and promoted to **Production** status — completing the full MLflow lifecycle:

```
Training run logged → Model registered → Version promoted to Production
```

This is the version saved to `Models/credit_scoring_model.pkl` and deployed to SageMaker.

---

## 5. Architecture

### Why This Architecture

The original crash was caused by dependency mismatches between the training environment and the built-in SageMaker container. The solution was to use a **custom Docker container** that bakes in the exact training environment — eliminating version mismatches entirely.

From there, the architecture became:

```
User
 │
 ▼
Streamlit UI  ──────────────────── served from ECS Fargate
 │                                 (containerised, deployed via ALB)
 │  POST /predict
 ▼
API Gateway  (HTTP API)
 │           routes all traffic through a single public URL
 ▼
Application Load Balancer
 │
 ├── /predict, /health, /docs ──▶  FastAPI  (ECS Fargate)
 │                                  │
 │                                  │  InvokeEndpoint (boto3)
 │                                  ▼
 │                           SageMaker Endpoint
 │                           ┌─────────────────────────────────┐
 │                           │  Custom Docker Container        │
 │                           │  python:3.9-slim                │
 │                           │  scikit-learn==1.2.2            │
 │                           │  Flask /ping  /invocations      │
 │                           │  ml.m5.large                    │
 │                           └─────────────────────────────────┘
 │
 └── /*  ──▶  Streamlit Frontend  (ECS Fargate)
```

### AWS Services Used

| Service | Role |
|---|---|
| Amazon SageMaker | Custom Docker model endpoint — real-time inference |
| Amazon ECR | Container registry for all 3 Docker images |
| Amazon ECS Fargate | Serverless container runtime — API + frontend |
| Application Load Balancer | Path-based routing between API and frontend |
| Amazon API Gateway (HTTP API) | Public-facing URL, request routing |
| Amazon CloudWatch | Logs for ECS services and SageMaker endpoint |
| Amazon S3 | Terraform state storage |
| AWS IAM | Least-privilege roles for SageMaker and ECS |

---

## 6. Why SageMaker — Not Just ECS

This is an important architectural question for an MLOps project.

**ECS alone** would mean running the model inside the FastAPI container. It works, but it is generic compute — a container runner with no awareness that it is serving an ML model.

**SageMaker** is a purpose-built ML serving platform. The difference matters because SageMaker gives you:

| Capability | ECS alone | SageMaker |
|---|---|---|
| Model versioning | Manual | Built-in (Model Registry) |
| A/B testing between model versions | Build it yourself | Production Variants — native |
| Canary / blue-green deployment | Build it yourself | Native, with auto-rollback |
| Invocation metrics (latency, error rate) | CloudWatch generic | SageMaker-specific metrics |
| Data drift detection | Not available | SageMaker Model Monitor |
| Model quality monitoring | Not available | SageMaker Clarify |
| Auto-scaling on ML metrics | CPU/memory only | Invocations per minute |
| MLflow / experiment tracking integration | Manual | Native |
| Audit trail of deployed models | None | Full lineage |

For an MLOps project the point is not just to serve predictions — it is to operate the model as a production system. SageMaker provides the infrastructure to do that. ECS provides compute. They are not equivalent.

The custom Docker approach used here keeps all of SageMaker's operational benefits while solving the dependency problem. The model runs in a container you control, on an ML-optimised instance, with all of SageMaker's observability and lifecycle management intact.

---

## 7. Infrastructure as Code

All AWS resources are defined in Terraform, split across purpose-specific files:

```
terraform/
├── main.tf           # Provider + S3 backend (state persistence)
├── variables.tf      # Image URIs, region, project name
├── ecr.tf            # 3 ECR repos (sagemaker-model, api, frontend)
├── iam.tf            # SageMaker execution role, ECS task/execution roles
├── sagemaker.tf      # SageMaker model, endpoint config, endpoint
├── networking.tf     # Security groups, ALB, target groups, listener rules
├── ecs.tf            # Cluster, task definitions, Fargate services
├── api_gateway.tf    # HTTP API, integration, routes, stage
└── outputs.tf        # Live URLs printed at end of deploy
```

**State management:** Terraform state is stored in an S3 bucket (`credit-score-tf-state-<account-id>`) with versioning and AES-256 encryption. This was a critical fix — without it, every CI run started with empty state and tried to recreate existing resources, causing `ResourceAlreadyExistsException` failures.

---

## 8. CI/CD Pipeline

```
git push origin main
        │
        ▼
┌─────────────────────────────────────────────────┐
│  GitHub Actions  .github/workflows/ci-cd.yml    │
│                                                 │
│  [test]  ──────────────────────────────────     │
│    pytest, flake8                         │     │
│                                           ▼     │
│  [security]                         [build-and-push]
│    bandit scan                        Docker BuildX + GHA cache
│                                       3 images → ECR
│                                       (sagemaker, api, frontend)
│                                               │
│                                               ▼
│                                         [deploy]
│                                    Bootstrap S3 state bucket
│                                    terraform init (S3 backend)
│                                    Pre-flight cleanup:
│                                      - Scale ECS to 0
│                                      - Delete SageMaker endpoint
│                                      - Delete SGs, log groups
│                                    terraform apply
│                                    Print live endpoints
└─────────────────────────────────────────────────┘
```

**Why pre-flight cleanup?** SageMaker does not hot-reload a new container image when you update a model in-place. Full deletion and recreation is required to guarantee the new image is running. ECS security groups that changed also need to be deleted first due to ENI attachment dependencies.

**Docker layer caching:** Using Docker BuildX with GitHub Actions cache (`type=gha`). On first run builds take ~15 min. Every subsequent run takes ~2–3 min because unchanged layers are restored from cache.

---

## 9. Errors Encountered and Resolutions

This rebuild involved significant debugging. Every error is documented below.

---

### Error 1 — Deprecated GitHub Actions versions
**Symptom:** Workflow failed immediately before running any jobs
**Message:** `This request has been automatically failed because it uses a deprecated version of actions/upload-artifact: v3`
**Cause:** GitHub deprecated v3 of several official actions in April 2024
**Resolution:** Upgraded `actions/cache`, `codecov/codecov-action`, and `actions/upload-artifact` from v3 to v4

---

### Error 2 — Terraform state not persisting between CI runs
**Symptom:** Every workflow run crashed with `ResourceAlreadyExistsException` on IAM roles, security groups, CloudWatch log groups
**Cause:** No Terraform backend was configured. The state file existed only on the CI runner's ephemeral filesystem and was destroyed at the end of every run. The next run started with empty state and tried to create resources that already existed in AWS
**Resolution:** Added S3 backend to `terraform/main.tf` with the bucket name passed dynamically via `-backend-config` flags in CI. The bucket is created in a bootstrap step if it does not exist. State now persists indefinitely across all runs.

---

### Error 3 — Import script hanging for 35+ minutes
**Symptom:** `Import pre-existing AWS resources` step ran for 35+ minutes with no progress
**Cause:** The import script ran `terraform import` for each resource sequentially. Each invocation re-initialises Terraform, acquires the S3 state lock, makes AWS API calls, and releases the lock. With 10+ resources this was inherently slow. Some resources from a previously cancelled run were in transitional states (SageMaker endpoint still deleting), causing certain imports to block indefinitely
**Resolution:** Replaced the import approach entirely with a **pre-flight deletion** step. Instead of importing what exists, the workflow deletes the conflicting resources before `terraform apply` so Terraform creates them fresh. Simpler, faster, and no hanging.

---

### Error 4 — SageMaker endpoint hung for 52 minutes (three sub-problems)

**4a — Wrong instance type**
`ml.t2.medium` is not available for SageMaker real-time inference endpoints. AWS accepted the API call but silently queued the provisioning request indefinitely.
Resolution: Changed to `ml.m5.large` — the smallest widely-available general-purpose inference instance.

**4b — Container crash loop**
`serve.py` called `joblib.load()` synchronously before starting Flask. SageMaker's health checker called `GET /ping` during container startup, received a connection refused (Flask not yet running), marked the container as failed, and restarted it. This looped indefinitely.
Resolution: Moved model loading to a background daemon thread. Flask starts in milliseconds. `/ping` returns `503 loading` while the model loads, then `200 healthy` once ready. The container never crashes — SageMaker retries health checks until it sees 200.

**4c — No Terraform timeout**
Without an explicit `timeouts` block, Terraform waited the full 30-minute default before reporting failure, making the CI run appear completely stuck.
Resolution: Attempted to add `timeouts { create = "15m" }` — discovered this block is not supported on `aws_sagemaker_endpoint` in the locked provider version. Removed it. The endpoint itself was fixed by 4a and 4b so the timeout became unnecessary.

---

### Error 5 — SageMaker OCI manifest rejection
**Symptom:** `Unsupported manifest media type application/vnd.oci.image.index.v1+json`
**Cause:** Docker BuildX creates multi-architecture OCI image index manifests by default (for multi-platform support). SageMaker only accepts Docker Image Manifest v2 Schema 2 (single platform)
**Resolution:** Added `platforms: linux/amd64`, `provenance: false`, and `sbom: false` to the SageMaker image's `docker/build-push-action` step. This forces a single-platform Docker v2 manifest.

---

### Error 6 — Security group stuck destroying
**Symptom:** `aws_security_group.ecs: Still destroying... [07m20s elapsed]`
**Cause:** The ECS security group `description` field was changed in Terraform. AWS security group descriptions are immutable — the only way to change them is to destroy and recreate. However, ECS Fargate tasks were running and holding Elastic Network Interfaces (ENIs) attached to the security group. AWS blocks SG deletion while any ENI references it.
**Resolution:** Reverted the description to its original value. Terraform then performed an in-place rule update (adding port 8501 for Streamlit) without needing to destroy the SG. Also added an ECS scale-to-0 step in pre-flight so tasks and their ENIs are released before any SG operations.

---

### Error 7 — Feature name mismatch on predictions
**Symptom:** `Feature names seen at fit time, yet now missing: Education_Bachelor's Degree, Gender_Male...`
**Cause:** The model was trained on a DataFrame that had already been one-hot encoded via `pd.get_dummies`. The inference server received raw categorical values (`{"Gender": "Male"}`) and passed them directly to `model.predict()`, which expected the encoded column format (`{"Gender_Male": 1, "Gender_Female": 0, ...}`)
**Resolution:** Added full preprocessing logic to `docker/serve.py` — `pd.get_dummies` on all categorical columns, then `df.reindex(columns=model.feature_names_in_, fill_value=0)` to align column order exactly to training, then scaler transform. The model's `feature_names_in_` attribute (sklearn ≥ 1.0) was used as the ground truth for column order.

---

### Error 8 — New container image not picked up by SageMaker
**Symptom:** After deploying a fixed `serve.py`, the same feature name error persisted
**Cause:** Terraform updated `aws_sagemaker_model` in-place when the ECR image URI changed, but SageMaker does not pull a new container image on in-place model resource updates. The running endpoint continued serving from the cached old image.
**Resolution:** Added SageMaker endpoint, endpoint config, and model deletion to the pre-flight step. Full deletion forces Terraform to create new resources with the new image on every deploy.

---

### Error 9 — `timeouts` block not supported
**Symptom:** `Blocks of type "timeouts" are not expected here` on `aws_sagemaker_endpoint`
**Cause:** The `timeouts` meta-argument is not supported on this resource in the version of the AWS Terraform provider locked in `.terraform.lock.hcl`
**Resolution:** Removed the block. The underlying issues (instance type, crash loop) had already been fixed so the timeout was no longer needed.

---

### Error 10 — Docker COPY with spaces in filename
**Symptom:** `/Score: not found` during API container build
**Cause:** Used backslash escaping (`COPY Data/Credit\ Score\ ...`) which is valid in bash but not in Dockerfile `COPY` instructions
**Resolution:** Used JSON array syntax: `COPY ["Data/Credit Score Classification Dataset.csv", "dest.csv"]`

---

### Error 11 — Quoted specifier in requirements.txt
**Symptom:** `pip install did not complete successfully: exit code 1` in Docker build
**Cause:** `"numpy<2.0"` — quotes are valid in shell (`pip install "numpy<2.0"`) but are treated as part of the package name in a `requirements.txt` file, causing pip to fail
**Resolution:** Removed quotes → `numpy<2.0`

---

## 10. Final Working State

### First Deployment (German Credit, 165 rows)

**Live endpoints at time of teardown:**
- API: `https://4p97tzuzvd.execute-api.us-east-1.amazonaws.com`
- Frontend: `http://credit-score-alb-<id>.us-east-1.elb.amazonaws.com`

**Verified predictions:**
```
{ age: 35, income: $75k, education: "Bachelor's", single, rented }  → Average
{ age: 22, income: $18k, education: "High School", single, 2 kids } → Low
{ age: 52, income: $200k, education: "Master's", married, owned }   → High
```

**Infrastructure destroyed** March 2026 — SageMaker `ml.m5.large` runs 24/7 at ~$0.115/hr (~$83/month) regardless of traffic, which is not justified for a portfolio project between active demo sessions.

---

### Second Deployment (Lending Club, 141k rows)

**Model:** Logistic Regression — AUC 0.702, trained on 141,259 real Lending Club loans

**Frontend form updated** to match Lending Club features — Loan Amount, Interest Rate, Grade, DTI, Annual Income, Employment Length, Purpose, Credit History.

**Verified prediction:**
```
{ loan: $10,000, term: 36mo, int_rate: 12%, grade: A1,
  purpose: Debt Consolidation, dti: 15%, income: $60k }
  → Default probability: 21.0% — HIGH RISK
```

**MLflow tracking:** 13 experiment runs across 3 datasets logged to DagsHub. Best model registered as `CreditScoringModel v9`, promoted to Production.

**The full codebase is preserved on GitHub.** A single push to `main` rebuilds and redeploys the entire stack from scratch via GitHub Actions.

---

## 11. What Would Be Different Next Time

The architecture is sound for an MLOps project. What would be optimised:

**Cost — SageMaker Serverless Inference**
Instead of a persistent `ml.m5.large` endpoint, use SageMaker Serverless Inference. It scales to zero when idle and bills per invocation. For a portfolio project with intermittent traffic this would cost near nothing.

```hcl
# In sagemaker.tf
production_variants {
  serverless_config {
    max_concurrency   = 5
    memory_size_in_mb = 2048
  }
}
```

**Monitoring — SageMaker Model Monitor**
Add a data capture configuration to log predictions and baseline them against training data distribution. This is where AIOps becomes tangible — alerting when the incoming data drifts from what the model was trained on.

**Model Registry**
Register the model in SageMaker Model Registry before deployment. This adds a formal approval step, version lineage, and makes A/B testing between versions straightforward.

These are incremental additions to the existing stack — the foundation built here supports all of them.

---

*Report generated March 2026 · David Osei Kumi*
