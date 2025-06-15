import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit.components.v1 as components

# Load model & preprocessing tools
model = joblib.load("xgb_churn_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Churn Prediction", layout="wide")

# Header
st.markdown("""
    <div style="display:flex; align-items:center; gap:10px;">
        <img src="https://img.icons8.com/emoji/48/magic-crystal-ball.png" width="40"/>
        <h1 style="color:#3B82F6; font-family:'Fira Code', monospace;">Churn Prediction Dashboard</h1>
    </div>
""", unsafe_allow_html=True)

# Badges
st.markdown("""
    <div style="display:flex; gap: 15px; margin-bottom: 20px;">
        <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
        <img src="https://img.shields.io/badge/Model-XGBoost-orange" />
        <img src="https://img.shields.io/badge/Frontend-Streamlit-ff4b4b?logo=streamlit&logoColor=white" />
        <img src="https://img.shields.io/badge/Explainability-SHAP-informational" />
    </div>
""", unsafe_allow_html=True)

# File upload
st.markdown("### 📤 Upload Customer CSV")
uploaded_file = st.file_uploader("Upload a `.csv` file", type="csv")

def preprocess(df):
    customer_ids = df['customerID'] if 'customerID' in df.columns else None
    df = df.drop(columns=['customerID', 'churned'], errors='ignore')
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    encoded = encoder.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    scaled = scaler.transform(df[num_cols])
    scaled_df = pd.DataFrame(scaled, columns=num_cols)

    return pd.concat([encoded_df, scaled_df], axis=1), customer_ids

def risk_tag(prob):
    if prob >= 80: return "🔴 High"
    elif prob >= 40: return "🟡 Medium"
    else: return "🟢 Low"

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Uploaded Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        X, customer_ids = preprocess(df)
        proba = model.predict_proba(X)[:, 1]
        df['Churn Probability (%)'] = (proba * 100).round(2)

        # Churn Table
        st.subheader("🎯 Customer Churn Predictions")
        result_df = pd.DataFrame({
            'customerID': customer_ids,
            'Churn Probability (%)': df['Churn Probability (%)'],
            'Risk Level': [risk_tag(p) for p in df['Churn Probability (%)']]
        })
        st.dataframe(result_df.style.background_gradient(cmap='Reds', subset=['Churn Probability (%)']),
                     use_container_width=True)

        # Top 10
        st.subheader("🚨 Top 10 At-Risk Customers")
        top10 = result_df.sort_values("Churn Probability (%)", ascending=False).head(10)
        st.dataframe(top10, use_container_width=True)

        # Distribution
        st.subheader("📊 Churn Probability Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Churn Probability (%)'], bins=20, kde=True, color="#3B82F6", ax=ax)
        st.pyplot(fig)

        # SHAP Explainability
        st.subheader("🔍 SHAP Explainability")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Global SHAP
        st.markdown("**🌐 Global Feature Importance**")
        fig_summary = plt.figure()
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(fig_summary)

        # Local SHAP
        if customer_ids is not None:
            selected_id = st.selectbox("Choose a customer to explain", options=customer_ids)
            idx = customer_ids[customer_ids == selected_id].index[0]
            st.markdown(f"**SHAP Force Plot for Customer ID: `{selected_id}`**")
            force_plot_html = shap.force_plot(
                base_value=explainer.expected_value,
                shap_values=shap_values[idx],
                features=X.iloc[idx],
                matplotlib=False
            )
            components.html(f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>", height=400)

        # Download predictions
        st.download_button("📥 Download CSV", data=result_df.to_csv(index=False),
                           file_name="churn_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

# Footer
st.markdown("---")
st.markdown('<p style="font-family:Fira Code; font-size:13px;">Made with ❤️ by Chetan, Tushar & AI</p>', unsafe_allow_html=True)
