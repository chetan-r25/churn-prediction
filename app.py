import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit.components.v1 as components
from io import StringIO

# Initialize SHAP JavaScript


# Load the trained model and tools
model = joblib.load("xgb_churn_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Configure Streamlit page
st.set_page_config(page_title="Churn Prediction", layout="wide")

# --- Logo & Title ---
st.markdown("""
    <div style="display:flex; align-items:center; gap:10px;">
        <img src="https://img.icons8.com/emoji/48/magic-crystal-ball.png" width="40"/>
        <h1 style="color:#3B82F6; font-family:'Fira Code', monospace;">Churn Prediction Dashboard</h1>
    </div>
""", unsafe_allow_html=True)

# --- Badges & Shields ---
st.markdown("""
    <div style="display: flex; align-items: center; gap: 20px; margin-top: -10px; margin-bottom: 20px;">
        <a href="https://github.com/chetan-r25/churn-prediction/actions">
            <img src="https://github.com/chetan-r25/churn-prediction/actions/workflows/deploy.yml/badge.svg" alt="GitHub Actions Status"/>
        </a>
        <img src="https://img.shields.io/badge/AI%20Powered-Streamlit%20XGBoost-orange?style=for-the-badge&logo=python&logoColor=white" />
        <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=2000&pause=1000&color=3B82F6&vCenter=true&width=400&lines=Making+AI+Easy+%F0%9F%A4%96;Powered+by+Love+%26+Data+%E2%9D%A4%EF%B8%8F" />
    </div>
""", unsafe_allow_html=True)

# --- File Upload ---
st.markdown('<h3 style="font-family:\'Fira Code\', monospace;">📤 Upload Customer CSV File</h3>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload test data (.csv)", type=["csv"])

def info(msg):
    st.markdown(f"<span style='color:gray;font-size:13px; font-family:Fira Code;'>{msg}</span>", unsafe_allow_html=True)

# --- Preprocessing ---
def preprocess_data(df):
    customer_ids = df['customerID'] if 'customerID' in df.columns else None
    df = df.drop(columns=['customerID', 'churned'], errors='ignore')
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    encoded = encoder.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    scaled = scaler.transform(df[num_cols])
    scaled_df = pd.DataFrame(scaled, columns=num_cols)
    return pd.concat([encoded_df, scaled_df], axis=1), customer_ids

def risk_label(prob):
    if prob >= 80: return "🔴 High"
    elif prob >= 40: return "🟡 Medium"
    else: return "🟢 Low"

def generate_sparkline():
    return "▁▂▃▄▅▆▇"

# --- Main Execution ---
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.markdown('<h4 style="font-family:\'Fira Code\', monospace;">📄 Uploaded Data Preview</h4>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        info("Showing first 10 rows of uploaded data.")

        X, customer_ids = preprocess_data(df)
        churn_proba = model.predict_proba(X)[:, 1]
        df['Churn Probability (%)'] = (churn_proba * 100).round(2)

        result_df = pd.DataFrame({
            'customerID': customer_ids,
            'Churn Probability (%)': df['Churn Probability (%)']
        })

        # Churn Probabilities Table
        st.markdown('<h4 style="font-family:\'Fira Code\', monospace;">🎯 Churn Probabilities</h4>', unsafe_allow_html=True)
        styled_df = result_df.style.background_gradient(cmap='RdYlGn_r', subset=['Churn Probability (%)'])
        st.dataframe(styled_df, use_container_width=True)

        # Top 10 Risk
        st.markdown('<h4 style="font-family:\'Fira Code\', monospace;">🚨 Top 10 At-Risk Customers</h4>', unsafe_allow_html=True)
        top10_df = result_df.sort_values("Churn Probability (%)", ascending=False).head(10).copy()
        top10_df["Risk Level"] = top10_df["Churn Probability (%)"].apply(risk_label)
        top10_df["Trend"] = [generate_sparkline() for _ in range(len(top10_df))]
        st.dataframe(top10_df[["customerID", "Churn Probability (%)", "Risk Level", "Trend"]], use_container_width=True)

        # Churn Distribution
        st.markdown('<h4 style="font-family:\'Fira Code\', monospace;">📊 Churn Probability Distribution</h4>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df['Churn Probability (%)'], bins=20, kde=True, ax=ax, color="#3B82F6")
        ax.set_title("Churn Probability Distribution", fontsize=14, fontname="Fira Code")
        st.pyplot(fig)

        # --- Explainability (Global + Local SHAP) ---
        st.markdown('<h4 style="font-family:\'Fira Code\', monospace;">🔍 Explainability with SHAP</h4>', unsafe_allow_html=True)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Global Summary Plot
        st.markdown("#### 🌐 Global Feature Importance (SHAP Summary)")
        fig_summary = plt.figure()
        shap.summary_plot(shap_values, X, show=False, plot_type="bar")
        st.pyplot(fig_summary)

        # Local SHAP Explainability - Force Plot
        if customer_ids is not None:
            selected_cust = st.selectbox("Select Customer ID for local explanation", options=customer_ids)
            cust_idx = customer_ids[customer_ids == selected_cust].index[0]
            st.markdown(f"<p style='font-family:Fira Code;'><b>Force Plot for Customer: `{selected_cust}`</b></p>", unsafe_allow_html=True)

            # Create force plot
            force_plot = shap.force_plot(
                base_value=explainer.expected_value,
                shap_values=shap_values[cust_idx],
                features=X.iloc[cust_idx],
                matplotlib=False
            )

            html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            components.html(html, height=400, scrolling=True)

        # Download Button
        st.download_button(
            label="📥 Download CSV Predictions",
            data=result_df.to_csv(index=False),
            file_name="churn_predictions.csv",
            mime="text/csv",
            help="Download the churn predictions with customer IDs and probabilities.",
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.markdown('<p style="font-family:Fira Code; font-style:italic;">💡 Made with ❤️ by Chetan Ramrakhyagit add app.pygi & AI</p>', unsafe_allow_html=True)
