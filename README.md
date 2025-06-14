<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-XGBoost-orange" />
  <img src="https://img.shields.io/badge/Built_with-Streamlit-ff4b4b?logo=streamlit&logoColor=white" />
  <img src="https://github.com/chetan-r25/churn-prediction/actions/workflows/deploy.yml/badge.svg" alt="GitHub Actions Status" />
  <img src="https://img.shields.io/badge/Hackathon_Beast-🚀-blueviolet" />
  <img src="https://img.shields.io/github/last-commit/chetan-r25/churn-prediction" />
</p>

---

# 🔍 Customer Churn Prediction Dashboard

A powerful, AI-powered churn prediction tool crafted during a 48-hour hackathon to help businesses retain their customers intelligently.

> 💡 **Live App**: [Click here to try it](https://churn-prediction-7oengduhpztxudfa56kppu.streamlit.app)

---

## 🎯 Key Features

- 📤 Upload `.csv` customer data
- 🤖 Predict churn probability for every customer
- 🚨 View top 10 at-risk customers (🔴 / 🟡 / 🟢)
- 📈 Visual insights: histogram & trendline
- 📌 Feature importance bar plot (XGBoost-based)
- 📥 Download predictions as `.csv`
- 🔁 Auto-deployed via GitHub Actions + Streamlit Cloud

---

## ⚙️ Tech Stack

| Layer          | Tool                             |
|----------------|----------------------------------|
| ML Model       | XGBoost                          |
| Preprocessing  | OneHotEncoder, StandardScaler    |
| Frontend       | Streamlit                        |
| Visualization  | Seaborn, Matplotlib              |
| Deployment     | GitHub Actions + Streamlit Cloud |
| Explainability | XGBoost Feature Importances      |

---

## 📁 Folder Structure
churn-prediction/
├── app.py # Streamlit dashboard
├── encoder.pkl # Fitted OneHotEncoder
├── scaler.pkl # Fitted StandardScaler
├── xgb_churn_model.pkl # Trained model
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .github/
└── workflows/
└── deploy.yml # GitHub Actions deploy pipeline

🔐 Privacy First
The app runs entirely client-side in your browser.

Uploaded data is never stored or transmitted.

Ideal for secure and fast churn evaluations.

🏆 Hackathon Contribution
Built during Hackathon Beast 2025, a 48-hour challenge to create a real-world AI product.
Our goal: insights, impact, and interpretability in minutes.

<p align="center"><b>Made with ❤️ by Chetan Ramrakhya, Tushar Vashishth, and AI</b></p> <p align="center"><i>“Made with ❤️ for the Hackathon”</i></p> ```


