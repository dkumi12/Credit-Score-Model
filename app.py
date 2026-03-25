import os
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Score Classifier", page_icon="💳", layout="centered")

# Read from env var so the same code works locally and on Streamlit Cloud
API_URL = os.environ.get(
    "API_GATEWAY_URL",
    "https://4p97tzuzvd.execute-api.us-east-1.amazonaws.com"
).rstrip("/")

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("💳 Credit Score Classifier")
st.markdown("Enter customer details to predict their credit category.")
st.markdown("---")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age            = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender         = st.selectbox("Gender", ["Male", "Female"])
        income         = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
        education      = st.selectbox("Education Level", [
                            "High School Diploma", "Associate's Degree",
                            "Bachelor's Degree", "Master's Degree", "Doctorate"])

    with col2:
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        children       = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        home_ownership = st.selectbox("Home Ownership", ["Rented", "Owned"])

    submitted = st.form_submit_button("🔍 Predict Credit Score", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    payload = {
        "age":            age,
        "gender":         gender,
        "income":         float(income),
        "education":      education,
        "marital_status": marital_status,
        "num_children":   children,
        "home_ownership": home_ownership,
    }

    with st.spinner("Contacting prediction API..."):
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            score = resp.json().get("credit_score", "Unknown")

            st.markdown("---")
            st.subheader(f"Predicted Credit Score: **{score}**")

            if score == "High":
                st.success("✅ This customer has a **High** credit score profile.")
                st.progress(1.0)
            elif score == "Average":
                st.info("ℹ️ This customer has an **Average** credit score profile.")
                st.progress(0.5)
            else:
                st.warning("⚠️ This customer has a **Low** credit score profile.")
                st.progress(0.2)

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out — the API may be cold-starting. Try again in 30 seconds.")
        except requests.exceptions.HTTPError as e:
            st.error(f"API error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"Powered by AWS SageMaker · ECS · API Gateway | endpoint: `{API_URL}`")
