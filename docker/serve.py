"""
SageMaker custom inference server.
Responds to GET /ping (health check) and POST /invocations (prediction).

Dataset : Lending Club (141k rows)
Model   : LogisticRegression (AUC 0.702)
Target  : Binary — 0 = No Default, 1 = Default

All features are pre-encoded numerics — preprocessing is just StandardScaler.
"""

import os
import json
import logging
import threading
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_PATH  = os.environ.get("MODEL_PATH",  "/opt/ml/model/credit_scoring_model.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "/opt/ml/model/scaler.pkl")

FEATURES = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc",
]

DEFAULT_LABELS = {0: "No Default", 1: "Default"}

model            = None
scaler           = None
training_columns = None
model_load_error = None


# ── Background loader ─────────────────────────────────────────────────────────
def _load_model():
    global model, scaler, training_columns, model_load_error
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded")

        if hasattr(model, "feature_names_in_"):
            training_columns = list(model.feature_names_in_)
            logger.info(f"Training columns ({len(training_columns)}): {training_columns}")

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            if training_columns is None and hasattr(scaler, "feature_names_in_"):
                training_columns = list(scaler.feature_names_in_)
            logger.info("Scaler loaded")
        else:
            logger.info("No scaler found — skipping scaling step")

    except Exception as exc:
        model_load_error = str(exc)
        logger.error(f"Failed to load model: {exc}")


threading.Thread(target=_load_model, daemon=True).start()


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(raw: dict) -> np.ndarray:
    """
    All Lending Club features are pre-encoded numerics.
    Steps: build DataFrame → align column order → StandardScaler.transform
    """
    df = pd.DataFrame([raw])

    if training_columns:
        df = df.reindex(columns=training_columns, fill_value=0)
    else:
        df = df.reindex(columns=FEATURES, fill_value=0)

    if scaler is not None:
        return scaler.transform(df)
    return df.values


# ── SageMaker required endpoints ──────────────────────────────────────────────
@app.route("/ping", methods=["GET"])
def ping():
    if model_load_error:
        return Response(json.dumps({"status": "error", "detail": model_load_error}),
                        status=500, mimetype="application/json")
    if model is None:
        return Response(json.dumps({"status": "loading"}),
                        status=503, mimetype="application/json")
    return Response(json.dumps({"status": "healthy"}),
                    status=200, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    if model is None:
        return Response(json.dumps({"error": "Model not ready"}),
                        status=503, mimetype="application/json")
    if request.content_type != "application/json":
        return Response("Unsupported media type", status=415)

    try:
        raw      = json.loads(request.data.decode("utf-8"))
        features = preprocess(raw)
        pred     = int(model.predict(features)[0])
        label    = DEFAULT_LABELS.get(pred, str(pred))

        # Include probability if available
        result = {"prediction": pred, "label": label}
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(features)[0][1])
            result["default_probability"] = round(prob, 4)

        logger.info(f"Prediction: {pred} → {label}")
        return Response(json.dumps(result), status=200, mimetype="application/json")

    except Exception as exc:
        logger.error(f"Prediction error: {exc}", exc_info=True)
        return Response(json.dumps({"error": str(exc)}),
                        status=500, mimetype="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
