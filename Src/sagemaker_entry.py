import joblib
import os
import pandas as pd
import json

def model_fn(model_dir):
    # Loads the full Pipeline (Preprocessor + Classifier)
    return joblib.load(os.path.join(model_dir, "credit_score_model.pkl"))

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return pd.DataFrame([data])
    raise ValueError("Unsupported content type")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, response_content_type):
    return json.dumps({'credit_score': str(prediction[0])})