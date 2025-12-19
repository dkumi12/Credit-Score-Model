import boto3
import json

# Initialize the SageMaker Runtime client
# Ensure your local AWS CLI is configured with the correct region
client = boto3.client('sagemaker-runtime', region_name='us-east-1')

# The name of your live endpoint from Terraform output
ENDPOINT_NAME = "credit-score-endpoint-v3"

# Example data matching your model's required fields
# Fields: Age, Gender, Income, Education, Marital Status, Number of Children, Home Ownership
payload = {
    "Age": 30,
    "Gender": "Female",
    "Income": 55000.0,              # Changed from 'Annual Income' to 'Income'
    "Education": "Bachelor's Degree", # Double check if it's 'Education' or 'Education Level'
    "Marital Status": "Single",
    "Number of Children": 1,
    "Home Ownership": "Rented"
}

def test_endpoint():
    print(f"🚀 Sending request to endpoint: {ENDPOINT_NAME}...")
    
    try:
        # Convert payload to JSON bytes
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse the response from SageMaker
        result = json.loads(response['Body'].read().decode())
        
        print("-" * 30)
        print("✅ Prediction Successful!")
        print(f"Result: {result}")
        print("-" * 30)
        
    except Exception as e:
        print("-" * 30)
        print(f"❌ Error invoking endpoint: {e}")
        print("-" * 30)

if __name__ == "__main__":
    test_endpoint()
