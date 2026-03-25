import os
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Credit Score Classifier", page_icon="💳", layout="wide")

API_URL = os.environ.get(
    "API_GATEWAY_URL",
    "https://4p97tzuzvd.execute-api.us-east-1.amazonaws.com"
).rstrip("/")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("💳 Credit Score Classifier")
st.sidebar.markdown("Powered by **AWS SageMaker · ECS · API Gateway**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🔍 Predict", "📊 Model Metrics"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Predict":
    st.title("🔍 Credit Score Prediction")
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

    if submitted:
        payload = {
            "age": age, "gender": gender, "income": float(income),
            "education": education, "marital_status": marital_status,
            "num_children": children, "home_ownership": home_ownership,
        }
        with st.spinner("Contacting prediction API..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
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
                st.error("⏱️ Request timed out — try again in 30 seconds.")
            except requests.exceptions.HTTPError:
                st.error(f"API error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("📊 Model Performance Metrics")
    st.markdown("Evaluation results on held-out test set (20% of data, random_state=42).")
    st.markdown("---")

    with st.spinner("Fetching metrics from API..."):
        try:
            resp = requests.get(f"{API_URL}/metrics", timeout=30)
            resp.raise_for_status()
            m = resp.json()
        except Exception as e:
            st.error(f"Could not load metrics: {e}")
            st.stop()

    if "error" in m:
        st.error(f"Metrics computation error: {m['error']}")
        st.stop()

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("🎯 Accuracy", f"{m['accuracy']}%")
    k2.metric("📦 Total Samples", m["total_samples"])
    k3.metric("🧪 Test Samples", m["test_samples"])

    st.markdown("---")

    # ── Per-class metrics ─────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Per-Class Metrics")
        if "per_class" in m:
            rows = []
            for cls, vals in m["per_class"].items():
                rows.append({
                    "Class":     cls,
                    "Precision": f"{vals['precision']}%",
                    "Recall":    f"{vals['recall']}%",
                    "F1 Score":  f"{vals['f1']}%",
                    "Support":   vals["support"],
                })
            st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)

    with col_right:
        st.subheader("Class Distribution (full dataset)")
        if "class_distribution" in m:
            dist_df = pd.DataFrame(
                list(m["class_distribution"].items()),
                columns=["Credit Score", "Count"]
            ).set_index("Credit Score")
            st.bar_chart(dist_df)

    st.markdown("---")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.subheader("Confusion Matrix")
        if "confusion_matrix" in m:
            labels = m["confusion_matrix"]["labels"]
            values = m["confusion_matrix"]["values"]
            cm_df  = pd.DataFrame(values, index=labels, columns=labels)
            cm_df.index.name   = "Actual \\ Predicted"
            st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=True)

    with col_fi:
        st.subheader("Top Feature Importances")
        if "top_features" in m:
            fi_df = pd.DataFrame(
                list(m["top_features"].items()),
                columns=["Feature", "Importance"]
            ).set_index("Feature").sort_values("Importance", ascending=True)
            st.bar_chart(fi_df)

    st.markdown("---")
    st.caption(f"Model: RandomForestClassifier via sklearn Pipeline · endpoint: `{API_URL}`")
