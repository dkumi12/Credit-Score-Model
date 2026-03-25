"""
SageMaker custom inference server.
Responds to GET /ping (health check) and POST /invocations (prediction).

Preprocessing matches utils.py exactly:
  1. pd.get_dummies on categorical columns
  2. Reindex to training column order (from model.feature_names_in_)
  3. StandardScaler.transform (scaler.pkl)
  4. model.predict
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

# Categorical columns encoded with pd.get_dummies during training
CATEGORICAL_COLS = ["Gender", "Education", "Marital Status", "Home Ownership"]

# Maps numeric prediction back to human-readable label
CREDIT_LABELS = {0: "Low", 1: "Average", 2: "High"}

model            = None
scaler           = None
training_columns = None   # column order seen at fit time
model_load_error = None


# ---------------------------------------------------------------------------
# Background loader — Flask starts immediately, model loads in parallel
# ---------------------------------------------------------------------------
def _load_model():
    global model, scaler, training_columns, model_load_error
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded")

        # Recover training column order from the model (sklearn ≥ 1.0)
        if hasattr(model, "feature_names_in_"):
            training_columns = list(model.feature_names_in_)
            logger.info(f"Training columns from model ({len(training_columns)}): {training_columns}")

        if os.path.exists(SCALER_PATH):
            logger.info(f"Loading scaler from {SCALER_PATH}")
            scaler = joblib.load(SCALER_PATH)
            # Fall back to scaler column order if model didn't have it
            if training_columns is None and hasattr(scaler, "feature_names_in_"):
                training_columns = list(scaler.feature_names_in_)
                logger.info(f"Training columns from scaler: {training_columns}")
            logger.info("Scaler loaded")
        else:
            logger.info("No scaler found — skipping scaling step")

    except Exception as exc:
        model_load_error = str(exc)
        logger.error(f"Failed to load model: {exc}")


threading.Thread(target=_load_model, daemon=True).start()


# ---------------------------------------------------------------------------
# Preprocessing — mirrors utils.preprocess_credit_data + split_and_scale_data
# ---------------------------------------------------------------------------
def preprocess(raw: dict) -> np.ndarray:
    """
    Converts raw API input dict → scaled numpy array ready for model.predict.

    Steps:
      1. Build single-row DataFrame with original column names
      2. pd.get_dummies on categorical columns (same as training)
      3. Reindex to training column order — fills any unseen category with 0
      4. StandardScaler.transform
    """
    df = pd.DataFrame([raw])

    # Step 1 — one-hot encode categoricals
    present_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=present_cats)

    # Step 2 — align to training column order
    if training_columns:
        df = df.reindex(columns=training_columns, fill_value=0)
        logger.debug(f"Aligned to {len(training_columns)} training columns")
    else:
        logger.warning("Training columns unknown — sending encoded data as-is")

    # Step 3 — scale
    if scaler is not None:
        return scaler.transform(df)
    return df.values


# ---------------------------------------------------------------------------
# SageMaker required endpoints
# ---------------------------------------------------------------------------
@app.route("/ping", methods=["GET"])
def ping():
    if model_load_error:
        body = {"status": "error", "detail": model_load_error}
        return Response(json.dumps(body), status=500, mimetype="application/json")
    if model is None:
        return Response(json.dumps({"status": "loading"}), status=503, mimetype="application/json")
    return Response(json.dumps({"status": "healthy"}), status=200, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    if model is None:
        return Response(json.dumps({"error": "Model not ready"}), status=503, mimetype="application/json")
    if request.content_type != "application/json":
        return Response("Unsupported media type", status=415)

    try:
        raw = json.loads(request.data.decode("utf-8"))
        logger.info(f"Received input: {raw}")

        features = preprocess(raw)
        pred_raw  = model.predict(features)[0]

        # Map numeric prediction → label if applicable
        label = CREDIT_LABELS.get(int(pred_raw), str(pred_raw)) \
                if isinstance(pred_raw, (int, np.integer)) else str(pred_raw)

        logger.info(f"Prediction: {pred_raw} → {label}")
        return Response(
            json.dumps({"credit_score": label}),
            status=200,
            mimetype="application/json",
        )
    except Exception as exc:
        logger.error(f"Prediction error: {exc}", exc_info=True)
        return Response(json.dumps({"error": str(exc)}), status=500, mimetype="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
