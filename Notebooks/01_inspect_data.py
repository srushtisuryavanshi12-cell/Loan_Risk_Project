import pandas as pd

print("--- LOADING LOAN DATASET ---")

# 1. Read the CSV file from our data folder
file_path = "data/loan_data.csv"
df = pd.read_csv(file_path, low_memory=False)

# 2. Print total rows and columns
print(f"Total Rows (Loans): {df.shape[0]:,}")
print(f"Total Columns (Features): {df.shape[1]}")

# 3. Display the first 5 rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# 4. Check loan status outcomes
if 'loan_status' in df.columns:
    print("\nLoan Status Breakdown:")
    print(df['loan_status'].value_counts())