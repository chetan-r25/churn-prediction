<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-XGBoost-orange" />
  <img src="https://img.shields.io/badge/Built_with-Streamlit-ff4b4b?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Deployment-Auto_via_GitHub_Actions-success?logo=github" />
  <img src="https://img.shields.io/badge/Hackathon_Beast-%F0%9F%90%BE-blueviolet" />
  <img src="https://img.shields.io/github/last-commit/chetan-r25/churn-prediction" />
</p>

---

# 🔍 Customer Churn Prediction Dashboard

A powerful, AI-driven churn prediction tool designed during a 48-hour hackathon. Built with ❤️ to help fintech companies retain customers through intelligent insights and visual explainability.

🔗 **Live App**: [Click here to try it](https://churn-prediction-7oengduhpztxudfa56kppu.streamlit.app)

---

## 🎯 Features

- ✅ Upload `.csv` customer data  
- 🎓 Predict churn probability for each customer  
- 📊 Color-coded churn scores with top-10 high-risk alerts  
- 📥 Download predictions directly as CSV  
- 📈 Interactive visuals of churn trends  
- 🧠 Feature importance bar plot (XGBoost-based)  
- 🚀 Deployed using **Streamlit Cloud** with GitHub auto-deploy  

---

## 📦 Tech Stack

| Component        | Tool                     |
|------------------|--------------------------|
| ML Model         | XGBoost                  |
| Preprocessing    | OneHotEncoder, StandardScaler |
| Frontend         | Streamlit                |
| Visualization    | Matplotlib, Seaborn      |
| Deployment       | GitHub Actions + Streamlit Cloud |
| Explainability   | Feature Importance Bars (SHAP optional) |

---

## 🖼️ Screenshots

| Prediction View | Churn Distribution |
|-----------------|--------------------|
| ![Prediction Table](assets/pred_table.png) | ![Churn Chart](assets/proba_chart.png) |

---

## 🧪 How to Run Locally

```bash
git clone https://github.com/chetan-r25/churn-prediction
cd churn-prediction
pip install -r requirements.txt
streamlit run app.py
