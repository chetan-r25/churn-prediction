<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-XGBoost-orange" />
  <img src="https://img.shields.io/badge/Built_with-Streamlit-ff4b4b?logo=streamlit&logoColor=white" />
  <img src="https://github.com/chetan-r25/churn-prediction/actions/workflows/deploy.yml/badge.svg" alt="GitHub Actions Status" />
  <img src="https://img.shields.io/badge/Hackathon_Beast-%F0%9F%90%BE-blueviolet" />
  <img src="https://img.shields.io/github/last-commit/chetan-r25/churn-prediction" />
</p>

---

```text
   _____ _                      _____           _     _             
  / ____| |                    |  __ \         (_)   | |            
 | |    | |__   ___  ___ ___   | |__) |_ _ _ __ _  __| | ___  _ __  
 | |    | '_ \ / _ \/ __/ __|  |  ___/ _` | '__| |/ _` |/ _ \| '_ \ 
 | |____| | | |  __/\__ \__ \  | |  | (_| | |  | | (_| | (_) | | | |
  \_____|_| |_|\___||___/___/  |_|   \__,_|_|  |_|\__,_|\___/|_| |_|

     🧠 AI-Driven Churn Prediction Dashboard • Built in 48hrs

🔍 Customer Churn Prediction Dashboard
A powerful, AI-powered churn prediction tool crafted during a 48-hour hackathon to help businesses retain their customers with confidence.

🎯 Live App: Click here to try it 🚀

✨ Features
📤 Upload .csv customer data

📊 Predict churn probability for every customer

🎯 View top 10 at-risk customers with risk levels (🔴 / 🟡 / 🟢)

🧠 Model-based feature importance plot

📈 Visual insights: histogram + trend line of churn probabilities

📥 Download predictions directly as .csv

🔁 Auto-deployed via GitHub Actions + Streamlit Cloud

⚙️ Tech Stack

| Layer          | Tool                             |
| -------------- | -------------------------------- |
| ML Model       | XGBoost                          |
| Encoding       | `OneHotEncoder`                  |
| Scaling        | `StandardScaler`                 |
| Frontend UI    | Streamlit                        |
| Plots & Viz    | Matplotlib, Seaborn              |
| Deployment     | GitHub Actions + Streamlit Cloud |
| Explainability | Feature Importance (XGBoost)     |

📁 Project Structure
churn-prediction/
├── app.py                  # Streamlit dashboard
├── encoder.pkl             # Fitted encoder
├── scaler.pkl              # Fitted scaler
├── xgb_churn_model.pkl     # Trained model
├── requirements.txt        # Dependencies
├── README.md               # This documentation
└── .github/
    └── workflows/
        └── deploy.yml      # GitHub Actions deploy pipeline

🧪 Local Run Instructions
git clone https://github.com/chetan-r25/churn-prediction
cd churn-prediction
pip install -r requirements.txt
streamlit run app.py


🔐 Privacy & Security
The app runs entirely client-side in your browser.

Your uploaded data is never stored or shared.

Ideal for quick, secure churn evaluations in real time.


🚀 Hackathon Contribution
Built for Hackathon Beast 2025 — a 48-hour challenge to create an intelligent AI product from scratch.
Our goal: deliver insights, impact, and interpretability in minutes.

❤️ Made with care by
Chetan Ramrakhya, Tushar Vashishth, and AI.

“Made with ❤️ for the Hackathon”
