import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

print("--- STARTING MODEL TRAINING ---")

# 1. Load cleaned dataset
df = pd.read_csv("data/cleaned_loan_data.csv")
print(f"Loaded cleaned dataset: {df.shape}")

# 2. Separate Features (X) and Target (y)
X = df.drop(columns=['is_default'])
y = df['is_default']

# 3. Stratified Train-Test Split (80% Train / 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Logistic Regression Model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Model Evaluation
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n--- MODEL PERFORMANCE ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# 7. Save Model & Scaler for Application Deployment
os.makedirs("app", exist_ok=True)
joblib.dump(model, "app/model.pkl")
joblib.dump(scaler, "app/scaler.pkl")
print("Saved model to 'app/model.pkl' and scaler to 'app/scaler.pkl'")