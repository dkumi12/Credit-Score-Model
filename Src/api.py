from fastapi import FastAPI
import boto3
import json

app = FastAPI()
client = boto3.client('sagemaker-runtime', region_name='us-east-1')

@app.post("/predict")
async def predict(data: dict):
    # Renaming fields to match your training features
    payload = {
        "Age": data.get("age"),
        "Gender": data.get("gender"),
        "Annual Income": data.get("income"), 
        "Education Level": data.get("education"),
        "Marital Status": data.get("marital_status"),
        "Number of Children": data.get("children"),
        "Home Ownership": data.get("home_ownership")
    }
    
    response = client.invoke_endpoint(
        EndpointName="credit-score-endpoint-v2",
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    return json.loads(response['Body'].read().decode())