import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set the global parameters
windowSize = 5
numPCAcomponents = 30
testRatio = 0.25

# Load the data
PATH = os.getcwd()
X_train = np.load(
    PATH
    + "/trainingData/"
    + "XtrainWindowSize"
    + str(windowSize)
    + "PCA"
    + str(numPCAcomponents)
    + "testRatio"
    + str(testRatio)
    + ".npy"
)

y_train = np.load(
    PATH
    + "/trainingData/"
    + "ytrainWindowSize"
    + str(windowSize)
    + "PCA"
    + str(numPCAcomponents)
    + "testRatio"
    + str(testRatio)
    + ".npy"
)

# Reshape the data for Random Forest
X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten the data

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=testRatio, random_state=42
)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
import joblib

joblib.dump(rf_model, "randomForestModel.pkl")
