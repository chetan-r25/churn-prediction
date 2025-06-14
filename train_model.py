import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Load data
df = pd.read_csv("gBqE3R1cmOb0qyAv.csv")
print("✅ Columns in the dataset:", df.columns.tolist())

# Save customerID separately (optional — for tracking, not for training)
if 'customerID' in df.columns:
    customer_ids = df['customerID']
    df = df.drop('customerID', axis=1)

# Check if 'churned' exists
if 'churned' not in df.columns:
    raise ValueError("❌ Expected column 'churned' not found in dataset!")

# Identify categorical and numeric columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove target from feature columns
numerical_cols = [col for col in numerical_cols if col != 'churned']

# Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Scale numeric features
scaler = StandardScaler()
scaled = scaler.fit_transform(df[numerical_cols])
scaled_df = pd.DataFrame(scaled, columns=numerical_cols)

# Combine all features
X = pd.concat([encoded_df.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)
y = df['churned'].reset_index(drop=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"✅ AUC-ROC Score: {auc:.4f}")

# Save model and preprocessors
joblib.dump(model, "xgb_churn_model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Saved: xgb_churn_model.pkl, encoder.pkl, scaler.pkl")
