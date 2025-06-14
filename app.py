import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load trained model and tools
model = joblib.load("xgb_churn_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Set up Streamlit page
st.set_page_config(page_title="Premium Churn Prediction", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        color: #2e3b4e;
        font-size: 36px;
        font-weight: 700;
    }
    .subheader {
        color: #4a5a6a;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 20px;
    }
    .stButton>button {
        border-radius: 10px;
        padding: 10px 20px;
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    .reportview-container .markdown-text-container {
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🔮 Customer Churn Prediction - Premium Dashboard</div>', unsafe_allow_html=True)

# Preprocessing function
def preprocess_data(df):
    customer_ids = df['customerID'] if 'customerID' in df.columns else None
    df = df.drop('customerID', axis=1, errors='ignore')

    if 'churned' in df.columns:
        df = df.drop('churned', axis=1)

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    encoded = encoder.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

    scaled = scaler.transform(df[num_cols])
    scaled_df = pd.DataFrame(scaled, columns=num_cols)

    processed_df = pd.concat([encoded_df.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)

    return processed_df, customer_ids

# App logic
uploaded_file = st.file_uploader("📤 Upload your CSV file (telco_test.csv format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown('<div class="subheader">📄 Uploaded Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    try:
        X, customer_ids = preprocess_data(df)
        churn_proba = model.predict_proba(X)[:, 1]
        df["Churn Probability (%)"] = (churn_proba * 100).round(2)

        result_df = pd.DataFrame({
            "customerID": customer_ids,
            "Churn Probability (%)": df["Churn Probability (%)"]
        })

        st.markdown('<div class="subheader">📊 Churn Probabilities</div>', unsafe_allow_html=True)
        st.dataframe(result_df, use_container_width=True)

        st.markdown('<div class="subheader">🚨 Top 10 High-Risk Customers</div>', unsafe_allow_html=True)
        st.dataframe(result_df.sort_values("Churn Probability (%)", ascending=False).head(10), use_container_width=True)

        st.download_button("📥 Download Results as CSV", data=result_df.to_csv(index=False), file_name="churn_predictions.csv")

        # SHAP explanation
        st.markdown('<div class="subheader">📉 SHAP Explainability</div>', unsafe_allow_html=True)
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        plt.clf()

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("---")
st.markdown("Made with ❤️ for the Hackathon.")
