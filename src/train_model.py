# Import required libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("data/winequality-red.csv", sep=',')

print("First 5 rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum())

def quality_category(q):
    if q <= 4:
        return 0      # Poor
    elif q == 5:
        return 1      # Average
    elif q == 6:
        return 2      # Good
    else:
        return 3      # Excellent

data["quality"] = data["quality"].apply(quality_category)
# -----------------------------
# Split features and target
# -----------------------------
X = data.drop("quality", axis=1)
y = data["quality"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Train Logistic Regression Model
# -----------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save Model and Scaler
# -----------------------------
joblib.dump(model, "models/logistic_regression_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel saved successfully!")