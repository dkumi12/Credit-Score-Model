from fastapi import FastAPI, HTTPException
import boto3
import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = FastAPI(title="Credit Score API")

ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "credit-score-endpoint")
AWS_REGION     = os.environ.get("AWS_REGION", "us-east-1")
MODEL_PATH     = os.environ.get("MODEL_PATH", "/app/credit_scoring_model.pkl")
DATA_PATH      = os.environ.get("DATA_PATH",  "/app/Credit_Score_Classification_Dataset.csv")

client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

# Computed once at startup — cached for all /metrics calls
_metrics_cache: dict = {}


def _compute_metrics():
    """Load model + dataset, compute evaluation metrics, cache result."""
    global _metrics_cache
    try:
        model = joblib.load(MODEL_PATH)
        df    = pd.read_csv(DATA_PATH)

        X = df.drop("Credit Score", axis=1)
        y = df["Credit Score"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        cm     = confusion_matrix(y_test, y_pred, labels=["Low", "Average", "High"]).tolist()

        # Feature importances from the pipeline's RandomForest
        rf           = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]
        feat_names   = list(preprocessor.get_feature_names_out())
        importances  = dict(zip(feat_names, [round(float(v), 4) for v in rf.feature_importances_]))
        # Keep top-10 by importance
        importances  = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10])

        _metrics_cache = {
            "accuracy":               round(accuracy_score(y_test, y_pred) * 100, 2),
            "total_samples":          len(df),
            "test_samples":           len(X_test),
            "class_distribution":     {str(k): int(v) for k, v in y.value_counts().items()},
            "per_class":              {
                cls: {
                    "precision": round(report[cls]["precision"] * 100, 1),
                    "recall":    round(report[cls]["recall"]    * 100, 1),
                    "f1":        round(report[cls]["f1-score"]  * 100, 1),
                    "support":   int(report[cls]["support"]),
                }
                for cls in ["Low", "Average", "High"] if cls in report
            },
            "confusion_matrix":       {"labels": ["Low", "Average", "High"], "values": cm},
            "top_features":           importances,
        }
    except Exception as exc:
        _metrics_cache = {"error": str(exc)}


# Compute metrics at startup (non-blocking for small dataset)
try:
    _compute_metrics()
except Exception:
    pass


@app.get("/health")
async def health():
    return {"status": "healthy", "endpoint": ENDPOINT_NAME}


@app.get("/metrics")
async def metrics():
    return _metrics_cache


@app.post("/predict")
async def predict(data: dict):
    payload = {
        "Age":                data.get("age"),
        "Gender":             data.get("gender"),
        "Income":             data.get("income"),
        "Education":          data.get("education"),
        "Marital Status":     data.get("marital_status"),
        "Number of Children": data.get("num_children"),
        "Home Ownership":     data.get("home_ownership"),
    }
    try:
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        return json.loads(response["Body"].read().decode())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
