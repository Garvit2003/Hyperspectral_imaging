"""Python script to classify the image using Random Forest."""

# Import the necessary libraries
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import os
import scipy.io as sio
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import spectral
import cv2

# Global Variables
windowSize = 5
numPCAcomponents = 30
testRatio = 0.25

PATH = os.getcwd()
print(PATH)


# Function to load the Indian Pines Dataset
def loadIndianPinesData():
    """Method to load IndianPines."""
    data_path = os.path.join(os.getcwd(), "data")
    data = sio.loadmat(os.path.join(data_path, "Indian_pines_corrected.mat"))[
        "indian_pines_corrected"
    ]
    labels = sio.loadmat(os.path.join(data_path, "Indian_pines_gt.mat"))[
        "indian_pines_gt"
    ]

    return data, labels


# Function to apply PCA for dimensionality reduction
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


# Function to create patches for classification
def Patch(data, height_index, width_index):
    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    return patch


# Load the Random Forest model
rf_model = joblib.load("randomForestModel.pkl")

# Load test data
X_test = np.load(
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

y_test = np.load(
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

# Reshape X_test for Random Forest
X_test = X_test.reshape(X_test.shape[0], -1)

# Make predictions using Random Forest
y_pred = rf_model.predict(X_test)

# Evaluate the model's performance
classification = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

# Save the classification report and confusion matrix
filename = (
    "reportWindowSize"
    + str(windowSize)
    + "PCA"
    + str(numPCAcomponents)
    + "testRatio"
    + str(testRatio)
    + ".txt"
)
with open(filename, "w") as x_file:
    x_file.write("Classification Report:\n")
    x_file.write("{}\n".format(classification))
    x_file.write("Confusion Matrix:\n")
    x_file.write("{}\n".format(confusion))

# Load the original image and apply PCA
X, y = loadIndianPinesData()
X, pca = applyPCA(X, numComponents=numPCAcomponents)

# Variables for image size and patching
height = y.shape[0]
width = y.shape[1]
PATCH_SIZE = 5

# Initialize output image for predictions
outputs = np.zeros((height, width))

# Predict the mineral class for each patch
for i in range(height - PATCH_SIZE + 1):
    for j in range(width - PATCH_SIZE + 1):
        target = int(y[i + PATCH_SIZE // 2, j + PATCH_SIZE // 2])
        if target == 0:
            continue
        else:
            image_patch = Patch(X, i, j)
            X_test_image = image_patch.reshape(1, -1).astype(
                "float32"
            )  # Flatten the patch
            prediction = rf_model.predict(X_test_image)
            outputs[i + PATCH_SIZE // 2][j + PATCH_SIZE // 2] = prediction

# Visualize the ground truth and prediction results
ground_truth = spectral.imshow(classes=y, figsize=(5, 5))
spectral.save_rgb("ground_truth.png", y, colors=spectral.spy_colors)
predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(5, 5))
spectral.save_rgb("predict_image.png", outputs.astype(int), colors=spectral.spy_colors)

# Display images using OpenCV
ground = cv2.imread("ground_truth.png")
cv2.imshow("Ground Truth Image", ground)

predict = cv2.imread("predict_image.png")
cv2.imshow("Classified Image", predict)

cv2.waitKey(0)
cv2.destroyAllWindows()
