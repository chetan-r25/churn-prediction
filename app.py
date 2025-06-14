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

# Streamlit config
st.set_page_config(page_title="Churn Prediction", layout="wide")

# Logo and title
st.markdown("""
    <div style="display:flex; align-items:center; gap:10px;">
        <img src="https://img.icons8.com/emoji/48/magic-crystal-ball.png" width="40"/>
        <h1 style="color:#3B82F6;">Churn Prediction Dashboard</h1>
    </div>
""", unsafe_allow_html=True)

# Shields
st.markdown("""
    <div style="display: flex; align-items: center; gap: 20px; margin-top: -10px; margin-bottom: 20px;">
        <a href="https://github.com/chetan-r25/churn-prediction/actions">
            <img src="https://github.com/chetan-r25/churn-prediction/actions/workflows/deploy.yml/badge.svg" alt="GitHub Actions Status"/>
        </a>
        <img src="https://img.shields.io/badge/AI%20Powered-Streamlit%20XGBoost-orange?style=for-the-badge&logo=python&logoColor=white" />
        <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=2000&pause=1000&color=3B82F6&vCenter=true&width=400&lines=Making+AI+Easy+%F0%9F%A4%96;Powered+by+Love+%26+Data+%E2%9D%A4%EF%B8%8F" />
    </div>
""", unsafe_allow_html=True)

st.markdown("### 📤 Upload Customer CSV File")
uploaded_file = st.file_uploader("Upload test data (.csv)", type=["csv"])

def info(msg):
    st.markdown(f"<span style='color:gray;font-size:13px;'>{msg}</span>", unsafe_allow_html=True)

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

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        info("Showing first 10 rows of the uploaded data.")

        X, customer_ids = preprocess_data(df)
        churn_proba = model.predict_proba(X)[:, 1]
        df['Churn Probability (%)'] = (churn_proba * 100).round(2)

        result_df = pd.DataFrame({
            'customerID': customer_ids,
            'Churn Probability (%)': df['Churn Probability (%)']
        })

        st.subheader("🎯 Churn Probabilities")
        info("Color-coded based on churn risk. Red = High risk, Green = Low risk.")
        styled_df = result_df.style.background_gradient(cmap='RdYlGn_r', subset=['Churn Probability (%)'])
        st.dataframe(styled_df, use_container_width=True)

        st.subheader("🚨 Top 10 At-Risk Customers (Tabular View)")
        top10_df = result_df.sort_values("Churn Probability (%)", ascending=False).head(10).copy()
        top10_df["Risk Level"] = top10_df["Churn Probability (%)"].apply(risk_label)
        top10_df["Trend"] = [generate_sparkline() for _ in range(len(top10_df))]
        st.dataframe(top10_df[["customerID", "Churn Probability (%)", "Risk Level", "Trend"]], use_container_width=True)

        st.subheader("📌 Visual Risk Cards for Top 10 Customers")
        info("Compact cards with color-coded progress bars.")
        cols_per_row = 3
        for i in range(0, len(top10_df), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(top10_df):
                    row = top10_df.iloc[i + j]
                    prob = float(row["Churn Probability (%)"])
                    customer = row["customerID"]
                    if prob >= 80:
                        risk_level = "🔴 High Risk"
                    elif prob >= 40:
                        risk_level = "🟡 Moderate Risk"
                    else:
                        risk_level = "🟢 Low Risk"
                    with cols[j]:
                        st.markdown(f"**🧾 `{customer}`**")
                        st.markdown(f"{risk_level} — **{prob:.2f}%**")
                        st.progress(prob / 100)

        st.subheader("📊 Churn Probability Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df['Churn Probability (%)'], bins=20, kde=True, ax=ax, color="#3B82F6")
        ax.set_title("Distribution of Churn Probability")
        st.pyplot(fig)

        st.subheader("📈 Churn Risk Trend Across Customers")
        sorted_probs = df['Churn Probability (%)'].sort_values().reset_index(drop=True)
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(sorted_probs, color="#F97316", linewidth=2)
        ax3.set_title("Sorted Churn Probability Trend", fontsize=14)
        ax3.set_xlabel("Customer Index (sorted by risk)")
        ax3.set_ylabel("Churn Probability (%)")
        st.pyplot(fig3)

        st.subheader("📌 Feature Importance (Radar View)")
        try:
            importances = model.feature_importances_
            feature_names = X.columns
            top_n = 10
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).reset_index(drop=True).head(top_n)

            # Add first row again to close radar loop
            importance_df = pd.concat([importance_df, importance_df.iloc[[0]]], ignore_index=True)

            angles = np.linspace(0, 2 * np.pi, len(importance_df), endpoint=False).tolist()
            importance_df['Angle'] = angles

            fig4 = plt.figure(figsize=(6, 6))
            ax4 = fig4.add_subplot(111, polar=True)
            ax4.plot(importance_df['Angle'], importance_df['Importance'], color='#10B981', linewidth=2)
            ax4.fill(importance_df['Angle'], importance_df['Importance'], alpha=0.3)
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(importance_df['Feature'][:-1], fontsize=9)
            ax4.set_title("Top 10 Feature Importances", fontsize=14)
            st.pyplot(fig4)
        except Exception as e:
            st.warning("⚠️ Could not plot feature importances.")
            st.error(str(e))

        st.download_button(
            label="📥 Download CSV Predictions",
            data=result_df.to_csv(index=False),
            file_name="churn_predictions.csv",
            mime="text/csv",
            help="Download the churn predictions with customer IDs and probabilities.",
            key="unique_download_button"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.markdown("💡 *Built with ❤️ by Chetan Ramrakhya, Tushar Vashishth and AI*")
