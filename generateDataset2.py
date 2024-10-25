"""Python script to generate dataset for Random Forest from .mat files."""

import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os

# Global Variables
numComponents = 30
windowSize = 5
testRatio = 0.25

PATH = os.getcwd()
print(PATH)


# Function to load the Indian Pines Dataset
def loadIndianPinesData():
    """Method to load the IndianPines Dataset."""
    data_path = os.path.join(os.getcwd(), "data")
    data = sio.loadmat(os.path.join(data_path, "Indian_pines_corrected.mat"))[
        "indian_pines_corrected"
    ]
    labels = sio.loadmat(os.path.join(data_path, "Indian_pines_gt.mat"))[
        "indian_pines_gt"
    ]

    return data, labels


# Function to split data into training and testing sets
def splitTrainTestSet(X, y, testRatio=0.10):
    """Method to split data into train and test set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testRatio, random_state=345, stratify=y
    )
    return X_train, X_test, y_train, y_test


# Standardize the data
def standartizeData(X):
    newX = np.reshape(X, (-1, X.shape[2]))
    scaler = preprocessing.StandardScaler().fit(newX)
    newX = scaler.transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], X.shape[2]))
    return newX, scaler


# Apply PCA for dimensionality reduction
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


# Zero padding for creating patches
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset : X.shape[0] + x_offset, y_offset : X.shape[1] + y_offset, :] = X
    return newX


# Create patches from the data
def createPatches(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) // 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.zeros(
        (X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2])
    )
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[
                r - margin : r + margin + 1, c - margin : c + margin + 1
            ]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1

    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1

    # Flatten the patches for Random Forest
    patchesData = patchesData.reshape(patchesData.shape[0], -1)

    return patchesData, patchesLabels


# Save the preprocessed data
def savePreprocessedData(
    X_trainPatches,
    X_testPatches,
    y_trainPatches,
    y_testPatches,
    windowSize,
    wasPCAapplied=False,
    numPCAComponents=0,
    testRatio=0.25,
):
    if wasPCAapplied:
        with open(
            PATH
            + "/trainingData/"
            + "XtrainWindowSize"
            + str(windowSize)
            + "PCA"
            + str(numPCAComponents)
            + "testRatio"
            + str(testRatio)
            + ".npy",
            "bw",
        ) as outfile:
            np.save(outfile, X_trainPatches)
        with open(
            PATH
            + "/trainingData/"
            + "XtestWindowSize"
            + str(windowSize)
            + "PCA"
            + str(numPCAComponents)
            + "testRatio"
            + str(testRatio)
            + ".npy",
            "bw",
        ) as outfile:
            np.save(outfile, X_testPatches)
        with open(
            PATH
            + "/trainingData/"
            + "ytrainWindowSize"
            + str(windowSize)
            + "PCA"
            + str(numPCAComponents)
            + "testRatio"
            + str(testRatio)
            + ".npy",
            "bw",
        ) as outfile:
            np.save(outfile, y_trainPatches)
        with open(
            PATH
            + "/trainingData/"
            + "ytestWindowSize"
            + str(windowSize)
            + "PCA"
            + str(numPCAComponents)
            + "testRatio"
            + str(testRatio)
            + ".npy",
            "bw",
        ) as outfile:
            np.save(outfile, y_testPatches)


# Load the dataset
X, y = loadIndianPinesData()

# Apply PCA for dimensionality reduction
X, pca = applyPCA(X, numComponents=numComponents)

# Create patches and flatten them for Random Forest
XPatches, yPatches = createPatches(X, y, windowSize=windowSize)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = splitTrainTestSet(XPatches, yPatches, testRatio)

# Save the preprocessed data
savePreprocessedData(
    X_train,
    X_test,
    y_train,
    y_test,
    windowSize=windowSize,
    wasPCAapplied=True,
    numPCAComponents=numComponents,
    testRatio=testRatio,
)
