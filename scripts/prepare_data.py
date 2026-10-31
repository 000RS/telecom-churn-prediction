import pandas as pd

# Load dataset
file_path = "data/WA_FnUseC_TelcoCustomerChurn.csv"
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# Convert TotalCharges from text to numeric
# Invalid/blank values become NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nMissing values after converting TotalCharges:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

# customerID is only an identifier and should not be used for prediction
df = df.drop(columns=["customerID"])

# Convert target variable:
# Yes = 1 (customer churned)
# No = 0 (customer stayed)
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

print("\nCleaned dataset shape:", df.shape)

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nData types:")
print(df.dtypes)

# Save cleaned version
df.to_csv("data/telco_churn_clean.csv", index=False)

print("\nCleaned dataset saved successfully!")