import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load model and tools
model = joblib.load("xgb_churn_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit page config
st.set_page_config(page_title="Churn Prediction", layout="wide")

# Optional Logo (upload a logo named logo.png or change path)
st.markdown("""
    <div style="display:flex; align-items:center; gap:10px;">
        <img src="https://img.icons8.com/emoji/48/magic-crystal-ball.png" width="40"/>
        <h1 style="color:#3B82F6;">Churn Prediction Dashboard</h1>
    </div>
""", unsafe_allow_html=True)

# File upload section
st.markdown("### 📤 Upload Customer CSV File")
uploaded_file = st.file_uploader("Upload test data (.csv)", type=["csv"])

# Tooltip helper
def info(msg):
    st.markdown(f"<span style='color:gray;font-size:13px;'>{msg}</span>", unsafe_allow_html=True)

# Preprocessing
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

# Main logic
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    info("Showing first 10 rows of the uploaded data.")

    try:
        X, customer_ids = preprocess_data(df)
        churn_proba = model.predict_proba(X)[:, 1]
        df['Churn Probability (%)'] = (churn_proba * 100).round(2)

        result_df = pd.DataFrame({
            'customerID': customer_ids,
            'Churn Probability (%)': df['Churn Probability (%)']
        })

        # Color-coded probability display
        st.subheader("🎯 Churn Probabilities")
        info("Color-coded based on churn risk. Red = High risk, Green = Low risk.")
        styled_df = result_df.style.background_gradient(cmap='RdYlGn_r', subset=['Churn Probability (%)'])
        st.dataframe(styled_df, use_container_width=True)

        # Top 10 high-risk with color-coded labels
        st.subheader("🚨 Top 10 At-Risk Customers")
        top10_df = result_df.sort_values("Churn Probability (%)", ascending=False).head(10).copy()

        def risk_label(prob):
            if prob >= 80:
                return "🔴 High"
            elif prob >= 40:
                return "🟡 Medium"
            else:
                return "🟢 Low"

        top10_df["Risk Level"] = top10_df["Churn Probability (%)"].apply(risk_label)
        st.dataframe(top10_df, use_container_width=True)

        # Visualize churn probability distribution
        st.subheader("📊 Churn Probability Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df['Churn Probability (%)'], bins=20, kde=True, ax=ax, color="#3B82F6")
        ax.set_title("Distribution of Churn Probability")
        st.pyplot(fig)

        # Feature importance (bar plot instead of SHAP)
        st.subheader("📌 Feature Importance (Model-based)")
        try:
            importance = model.feature_importances_
            feature_names = X.columns
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False).head(10)

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            sns.barplot(data=importance_df, x="Importance", y="Feature", palette="coolwarm", ax=ax2)
            ax2.set_title("Top 10 Important Features")
            st.pyplot(fig2)
        except:
            st.warning("Feature importance could not be extracted from this model.")

        # Download option
        st.download_button("📥 Download Predictions", result_df.to_csv(index=False), file_name="churn_predictions.csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("---")
st.markdown("💡 *Built with ❤️ by Chetan Ramrakhya , Tushar Vashishth and AI*")
