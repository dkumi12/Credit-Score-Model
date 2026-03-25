import os
import requests
import streamlit as st

st.set_page_config(page_title="Credit Score Classifier", page_icon="💳", layout="centered")

API_URL = os.environ.get(
    "API_GATEWAY_URL",
    "https://4p97tzuzvd.execute-api.us-east-1.amazonaws.com"
).rstrip("/")

st.title("💳 Credit Score Classifier")
st.markdown("Enter customer details below to predict their credit score category.")
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

if submitted:
    payload = {
        "age": age, "gender": gender, "income": float(income),
        "education": education, "marital_status": marital_status,
        "num_children": children, "home_ownership": home_ownership,
    }

    with st.spinner("Analysing customer profile..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            resp.raise_for_status()
            score = resp.json().get("credit_score", "Unknown")

            st.markdown("---")

            # ── Result card ───────────────────────────────────────────────────
            SCORE_CONFIG = {
                "High": {
                    "emoji": "🟢", "color": "#1f7a1f",
                    "bar": 1.0,
                    "badge": "LOW RISK",
                    "summary": "Strong financial profile. Likely to qualify for premium credit products with favourable interest rates.",
                    "tips": [
                        "Eligible for premium credit cards and mortgages",
                        "Likely to receive lowest available interest rates",
                        "Good candidate for high credit limit products",
                    ],
                },
                "Average": {
                    "emoji": "🟡", "color": "#b38600",
                    "bar": 0.55,
                    "badge": "MODERATE RISK",
                    "summary": "Satisfactory financial profile. May qualify for standard credit products with moderate rates.",
                    "tips": [
                        "Qualifies for standard credit cards and personal loans",
                        "Interest rates may be slightly above prime",
                        "Consider reducing debt-to-income ratio to improve score",
                    ],
                },
                "Low": {
                    "emoji": "🔴", "color": "#a10000",
                    "bar": 0.2,
                    "badge": "HIGH RISK",
                    "summary": "Credit profile needs improvement. Limited access to mainstream credit products.",
                    "tips": [
                        "Focus on building a consistent payment history",
                        "Reduce existing debt before applying for new credit",
                        "Consider a secured credit card to rebuild credit",
                    ],
                },
            }

            cfg = SCORE_CONFIG.get(score, {
                "emoji": "❓", "color": "#555", "bar": 0,
                "badge": "UNKNOWN", "summary": score, "tips": [],
            })

            # Big result header
            st.markdown(
                f"<h2 style='color:{cfg['color']}'>"
                f"{cfg['emoji']} Credit Score: {score} &nbsp;"
                f"<span style='font-size:0.55em; background:{cfg['color']}; "
                f"color:white; padding:3px 10px; border-radius:12px;'>"
                f"{cfg['badge']}</span></h2>",
                unsafe_allow_html=True,
            )

            st.progress(cfg["bar"])
            st.markdown(f"**{cfg['summary']}**")
            st.markdown("---")

            # Two columns: customer summary + recommendations
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### 👤 Customer Profile")
                st.markdown(f"""
| Field | Value |
|---|---|
| Age | {age} |
| Gender | {gender} |
| Annual Income | ${income:,.0f} |
| Education | {education} |
| Marital Status | {marital_status} |
| Children | {children} |
| Home Ownership | {home_ownership} |
""")

            with c2:
                st.markdown("#### 💡 Recommendations")
                for tip in cfg["tips"]:
                    st.markdown(f"- {tip}")

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out — the API may be cold-starting. Try again in 30 seconds.")
        except requests.exceptions.HTTPError:
            st.error(f"API error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

st.markdown("---")
st.caption(f"Powered by AWS SageMaker · ECS · API Gateway | `{API_URL}`")
