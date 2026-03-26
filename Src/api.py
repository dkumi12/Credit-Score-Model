from fastapi import FastAPI, HTTPException
import boto3
import json
import os

app = FastAPI(title="Credit Default Prediction API")

ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "credit-score-endpoint")
AWS_REGION    = os.environ.get("AWS_REGION", "us-east-1")

client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)


@app.get("/health")
async def health():
    return {"status": "healthy", "endpoint": ENDPOINT_NAME}


@app.post("/predict")
async def predict(data: dict):
    # Map incoming request fields to Lending Club feature names
    payload = {
        "loan_amnt":            data.get("loan_amnt"),
        "term":                 data.get("term"),
        "int_rate":             data.get("int_rate"),
        "installment":          data.get("installment"),
        "grade":                data.get("grade"),
        "sub_grade":            data.get("sub_grade"),
        "emp_length":           data.get("emp_length"),
        "home_ownership":       data.get("home_ownership"),
        "annual_inc":           data.get("annual_inc"),
        "verification_status":  data.get("verification_status"),
        "purpose":              data.get("purpose"),
        "dti":                  data.get("dti"),
        "delinq_2yrs":          data.get("delinq_2yrs"),
        "inq_last_6mths":       data.get("inq_last_6mths"),
        "open_acc":             data.get("open_acc"),
        "pub_rec":              data.get("pub_rec"),
        "revol_bal":            data.get("revol_bal"),
        "revol_util":           data.get("revol_util"),
        "total_acc":            data.get("total_acc"),
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
