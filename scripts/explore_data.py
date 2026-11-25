import pandas as pd

# Load the IBM Telco Customer Churn dataset
file_path = "data/WA_FnUseC_TelcoCustomerChurn.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 customers:")
print(df.head())

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())