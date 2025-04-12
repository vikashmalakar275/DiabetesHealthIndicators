import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('data/diabetes_012_health_indicators_BRFSS2015.csv')

# Data Exploration
print("Dataset shape:", data.shape)
print("\nClass distribution:")
print(data['Diabetes_012'].value_counts())

# Separate features and target variable
X = data.drop('Diabetes_012', axis=1)
y = data['Diabetes_012']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,  # Number of trees in the forest
    max_depth=10,      # Maximum depth of each tree
    min_samples_split=5,  # Minimum samples required to split a node
    min_samples_leaf=2,   # Minimum samples required at each leaf node
    random_state=42,      # For reproducibility
    class_weight='balanced'  # Adjust weights inversely proportional to class frequencies
)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Save the model (optional)
import joblib
joblib.dump(rf_model, 'model/model.pkl')