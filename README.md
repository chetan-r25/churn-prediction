<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-XGBoost-orange" />
  <img src="https://img.shields.io/badge/Built_with-Streamlit-ff4b4b?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Deployment-In_Progress-yellow" />
  <img src="https://img.shields.io/badge/Hackathon_Beast-%F0%9F%90%BE-blueviolet" />
  <img src="https://img.shields.io/github/last-commit/chetan-r25/churn-prediction" />
</p>

# 🔍 Customer Churn Prediction Tool

This project is a powerful, AI-driven churn prediction tool built during a 48-hour hackathon. It uses machine learning to identify customers at risk of leaving a fintech service and provides insights via a clean, interactive dashboard.

---

## 📦 Features

- ✅ Upload your own `.csv` data
- 📊 Predict customer churn with **XGBoost**
- 🧠 See customer-wise churn probabilities
- 📌 View top 10 high-risk customers
- 📉 AUC-ROC Score for model quality
- 💾 Download predictions as CSV
- 🎯 SHAP-based explainability coming soon!

---

## ⚙️ Tech Stack

- **Frontend**: Streamlit (Python)
- **Model**: XGBoost
- **Preprocessing**: Scikit-learn (OneHotEncoder, StandardScaler)
- **Explainability (upcoming)**: SHAP
- **Deployment**: Streamlit Share / GitHub Pages (TBD)

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/chetan-r25/churn-prediction
cd churn-prediction
pip install -r requirements.txt
streamlit run app.py
