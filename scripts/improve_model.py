import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Load cleaned dataset
df = pd.read_csv("data/telco_churn_clean.csv")

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Identify categorical columns
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Random Forest
rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ]
)

# Parameters to test
parameters = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [5, 10, 15, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}

# Search for better settings
search = RandomizedSearchCV(
    pipeline,
    param_distributions=parameters,
    n_iter=20,
    scoring="recall",
    cv=5,
    random_state=42,
    n_jobs=-1
)

print("Searching for improved Random Forest parameters...")
search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest parameters:")
print(search.best_params_)

# Test improved model
y_pred = best_model.predict(X_test)
y_probability = best_model.predict_proba(X_test)[:, 1]

print("\n--- IMPROVED MODEL RESULTS ---")

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_probability):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save improved model
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/improved_churn_model.pkl")

print("\nImproved model saved successfully!")