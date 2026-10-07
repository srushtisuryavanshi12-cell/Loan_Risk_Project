import pandas as pd
import numpy as np

print("--- STARTING DATA PREPROCESSING ---")

# 1. Load dataset
df = pd.read_csv("data/loan_data.csv", low_memory=False)
print(f"Original shape: {df.shape}")

# 2. Define default categories
bad_statuses = [
    'Charged Off',
    'Default',
    'Does not meet the credit policy. Status:Charged Off',
    'Late (31-120 days)'
]

good_statuses = [
    'Fully Paid',
    'Does not meet the credit policy. Status:Fully Paid'
]

# Filter for completed/definitive loan outcomes
df_filtered = df[df['loan_status'].isin(bad_statuses + good_statuses)].copy()

# Create binary target: 1 = Default, 0 = Non-Default
df_filtered['is_default'] = df_filtered['loan_status'].apply(lambda x: 1 if x in bad_statuses else 0)

print(f"\nFiltered shape (completed loans): {df_filtered.shape}")
print("\nTarget Class Distribution:")
print(df_filtered['is_default'].value_counts(normalize=True).rename({0: "Non-Default (0)", 1: "Default (1)"}))

# 3. Select key financial risk indicators
candidate_features = [
    'loan_amnt', 'int_rate', 'installment', 'annual_inc',
    'dti', 'delinq_2yrs', 'open_acc', 'pub_rec', 'revol_bal',
    'revol_util', 'total_acc', 'is_default'
]

# Keep features present in the dataset
selected_features = [f for f in candidate_features if f in df_filtered.columns]
df_clean = df_filtered[selected_features].copy()

# Clean percentage strings if interest rate or revol_util are object types
for col in ['int_rate', 'revol_util']:
    if col in df_clean.columns and df_clean[col].dtype == 'object':
        df_clean[col] = df_clean[col].astype(str).str.rstrip('%').astype('float')

# Remove missing rows for core features
df_clean = df_clean.dropna()

print(f"\nFinal Cleaned Shape: {df_clean.shape}")

# 4. Save processed dataset
output_path = "data/cleaned_loan_data.csv"
df_clean.to_csv(output_path, index=False)
print(f"\nSaved cleaned dataset to: {output_path}")