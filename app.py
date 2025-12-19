import streamlit as st
import boto3
import json

# Page Configuration
st.set_page_config(page_title="Credit Score Classifier", page_icon="💳")

st.title("💳 Credit Score Classification")
st.markdown("Enter customer details below to predict their credit category.")

# Sidebar for AWS Configuration
ENDPOINT_NAME = "credit-score-endpoint-v3"
region = "us-east-1"

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        income = st.number_input("Annual Income ($)", min_value=0, value=50000)
        education = st.selectbox("Education Level", 
                                ["High School Diploma", "Associate's Degree", 
                                 "Bachelor's Degree", "Master's Degree", "Doctorate"])
    
    with col2:
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        home_ownership = st.selectbox("Home Ownership", ["Rented", "Owned"])
    
    submit = st.form_submit_button("Predict Credit Score")

if submit:
    # Prepare the payload exactly as the model expects
    payload = {
        "Age": age,
        "Gender": gender,
        "Income": float(income),
        "Education": education,
        "Marital Status": marital_status,
        "Number of Children": children,
        "Home Ownership": home_ownership
    }
    
    try:
        # Initialize SageMaker client
        client = boto3.client('sagemaker-runtime', region_name=region)
        
        # Invoke Endpoint
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        result = json.loads(response['Body'].read().decode())
        score = result.get('credit_score', 'Unknown')
        
        # Display Result with Color Coding
        st.subheader(f"Predicted Category: {score}")
        if score == "High":
            st.success("This customer has a High Credit Score profile.")
        elif score == "Average":
            st.info("This customer has an Average Credit Score profile.")
        else:
            st.warning("This customer has a Low Credit Score profile.")
            
    except Exception as e:
        st.error(f"Error connecting to SageMaker: {e}")
