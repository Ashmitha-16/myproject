# Experiment No 7
# Build and Evaluate a Simple ANN using scikit-learn (MLPClassifier)

# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load dataset
iris = load_iris()
X = iris.data        # Features
y = iris.target      # Labels

# Step 3: Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Data preprocessing (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Define ANN model using MLPClassifier
# Architecture: 4 inputs -> 10 hidden -> 8 hidden -> 3 outputs
model = MLPClassifier(
    hidden_layer_sizes=(10, 8),  # Two hidden layers: 10 neurons, then 8 neurons
    activation='relu',            # ReLU activation function
    solver='adam',                # Adam optimizer
    max_iter=1000,                # Maximum iterations
    random_state=42,              # For reproducibility
    learning_rate_init=0.001      # Learning rate
)

# Step 6: Train model
print("Training the ANN model...")
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Testing Accuracy: {accuracy * 100:.2f}%")

# Training accuracy (on training set)
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Step 9: Classification Report and Confusion Matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Step 10: Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix - ANN (MLPClassifier)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nANN Experiment completed successfully!")