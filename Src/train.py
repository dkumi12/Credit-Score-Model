"""
Training script for Credit Default Classifier.
Compares Logistic Regression, Random Forest, and XGBoost.
Logs all experiments, metrics, and artifacts to DagsHub via MLflow.

Dataset : Lending Club Clean (150,000 rows) from HuggingFace
Target  : Binary default label (0 = No Default, 1 = Default)

Run:
    pip install -r requirements-train.txt
    python Src/train.py
"""

import os
from dotenv import load_dotenv
load_dotenv()  # loads HF_TOKEN from .env

import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve
)
from mlflow.models.signature import infer_signature

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed — skipping. Run: pip install xgboost")

# ── DagsHub ───────────────────────────────────────────────────────────────────
dagshub.init(repo_owner="dkumi12", repo_name="Credit-Score-Model", mlflow=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
TMP_DIR   = os.path.join(BASE_DIR, "tmp_artifacts")

HF_DATASET = "hf://datasets/RPD123-byte/credit-risk-datasets/lending_club_clean.csv"

LABELS = ["No Default", "Default"]

TARGET   = "default"
FEATURES = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc"
]


def load_and_prepare():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not set. Add it to your .env file.")

    print("Loading dataset from HuggingFace...")
    import huggingface_hub
    huggingface_hub.login(token=token, add_to_git_credential=False)

    df = pd.read_csv(HF_DATASET).dropna()
    rate = df[TARGET].mean()
    print(f"Dataset  : {df.shape[0]:,} rows  |  {df.shape[1]} columns")
    print(f"Default rate: {rate:.1%}  ({df[TARGET].sum():,} defaults / {len(df):,} total)\n")
    return df


# ── Pipeline ──────────────────────────────────────────────────────────────────
def build_pipeline(classifier) -> Pipeline:
    # All features are numeric — just scale them
    return Pipeline(steps=[
        ("scaler",     StandardScaler()),
        ("classifier", classifier),
    ])


# ── Artifact helpers ──────────────────────────────────────────────────────────
def save_confusion_matrix(y_test, y_pred, path: str, title: str):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_roc_curve(y_test, y_prob, path: str, algorithm: str):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, color="steelblue",
             label=f"{algorithm}  (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {algorithm}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_feature_importance(pipeline: Pipeline, path: str, algorithm: str):
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return
    all_features = FEATURES
    importances  = classifier.feature_importances_
    indices      = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances[indices], color="steelblue")
    plt.xticks(range(len(importances)),
               [all_features[i] for i in indices], rotation=45, ha="right")
    plt.title(f"Feature Importance — {algorithm}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ── Experiment runner ─────────────────────────────────────────────────────────
def log_run(run_name, algorithm_label, classifier,
            X_train, X_test, y_train, y_test, test_size):
    with mlflow.start_run(run_name=run_name):
        pipeline  = build_pipeline(classifier)
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=5, scoring="roc_auc")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        params = {"algorithm": algorithm_label, "test_size": test_size,
                  "random_state": 42, "train_rows": len(X_train),
                  "test_rows": len(X_test)}
        params.update({k: str(v) for k, v in classifier.get_params().items()})
        mlflow.log_params(params)

        metrics = {
            "accuracy":    accuracy_score(y_test, y_pred),
            "precision":   precision_score(y_test, y_pred, zero_division=0),
            "recall":      recall_score(y_test, y_pred, zero_division=0),
            "f1_score":    f1_score(y_test, y_pred, zero_division=0),
            "roc_auc":     roc_auc_score(y_test, y_prob),
            "pr_auc":      average_precision_score(y_test, y_prob),
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std":  float(cv_scores.std()),
        }
        mlflow.log_metrics(metrics)

        cm_path = os.path.join(TMP_DIR, f"cm_{algorithm_label}.png")
        save_confusion_matrix(y_test, y_pred, cm_path,
                              f"Confusion Matrix — {algorithm_label}")
        mlflow.log_artifact(cm_path, artifact_path="charts")

        roc_path = os.path.join(TMP_DIR, f"roc_{algorithm_label}.png")
        save_roc_curve(y_test, y_prob, roc_path, algorithm_label)
        mlflow.log_artifact(roc_path, artifact_path="charts")

        fi_path = os.path.join(TMP_DIR, f"fi_{algorithm_label}.png")
        save_feature_importance(pipeline, fi_path, algorithm_label)
        if os.path.exists(fi_path):
            mlflow.log_artifact(fi_path, artifact_path="charts")

        report_path = os.path.join(TMP_DIR, f"report_{algorithm_label}.txt")
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, y_pred, target_names=LABELS))
        mlflow.log_artifact(report_path)

        sig     = infer_signature(X_train, pipeline.predict(X_train))
        example = X_train.head(5)
        mlflow.sklearn.log_model(
            pipeline, artifact_path="model",
            signature=sig, input_example=example,
            registered_model_name="CreditScoringModel",
        )

        run_id = mlflow.active_run().info.run_id
        print(f"  [{algorithm_label:22s}]  "
              f"AUC={metrics['roc_auc']:.3f}  "
              f"PR-AUC={metrics['pr_auc']:.3f}  "
              f"F1={metrics['f1_score']:.3f}  "
              f"CV={metrics['cv_auc_mean']:.3f}±{metrics['cv_auc_std']:.3f}  "
              f"run={run_id[:8]}")

        return pipeline, metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(TMP_DIR,   exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df        = load_and_prepare()
    X         = df[FEATURES]
    y         = df[TARGET]
    test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    mlflow.set_experiment("credit-score-classification")

    print("=" * 75)
    print("  Algorithm comparison — German Credit dataset (binary default)")
    print("=" * 75)

    results   = {}
    pipelines = {}

    # Run 1 — Logistic Regression (baseline)
    pipe, metrics = log_run(
        "logistic-regression", "LogisticRegression",
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        X_train, X_test, y_train, y_test, test_size,
    )
    results["LogisticRegression"]   = metrics
    pipelines["LogisticRegression"] = pipe

    # Run 2 — Random Forest
    pipe, metrics = log_run(
        "random-forest", "RandomForest",
        RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               random_state=42),
        X_train, X_test, y_train, y_test, test_size,
    )
    results["RandomForest"]   = metrics
    pipelines["RandomForest"] = pipe

    # Run 3 — XGBoost
    if XGBOOST_AVAILABLE:
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        pipe, metrics = log_run(
            "xgboost", "XGBoost",
            XGBClassifier(n_estimators=200, scale_pos_weight=pos_weight,
                          eval_metric="auc", verbosity=0, random_state=42),
            X_train, X_test, y_train, y_test, test_size,
        )
        results["XGBoost"]   = metrics
        pipelines["XGBoost"] = pipe

    # ── Summary ───────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\n{'=' * 75}")
    print(f"  Best : {best_name}  "
          f"(AUC={results[best_name]['roc_auc']:.4f}  "
          f"F1={results[best_name]['f1_score']:.4f})")
    print(f"  View : https://dagshub.com/dkumi12/Credit-Score-Model.mlflow")
    print(f"{'=' * 75}\n")

    joblib.dump(pipelines[best_name],
                os.path.join(MODEL_DIR, "credit_scoring_model.pkl"))
    print(f"  Saved {best_name} → Models/credit_scoring_model.pkl")


if __name__ == "__main__":
    main()
