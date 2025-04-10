import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("data/diabetes_012_health_indicators_BRFSS2015.csv")

# Basic Info
print("Shape of dataset:", df.shape)
print(df["Diabetes_012"].value_counts())

# ------------------ Data Visualization ------------------

# Set style
sns.set(style="whitegrid")

# 1. Distribution of target classes
plt.figure(figsize=(6, 4))
sns.countplot(x="Diabetes_012", data=df, palette="Set2")
plt.title("Distribution of Diabetes Classes")
plt.xlabel("Diabetes Class (0 = No, 1 = Pre, 2 = Yes)")
plt.ylabel("Count")
plt.show()

# 2. Correlation heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# ------------------ Machine Learning Model ------------------

# Sample data to avoid memory issues
df_sampled = df.sample(n=30000, random_state=42)

# Features and target
X = df_sampled.drop(columns=["Diabetes_012"])
y = df_sampled["Diabetes_012"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))


#Save trained model
joblib.dump(model, "model/model.pkl")
