"""
This file demonstrates the implementation and visualization of the
K-Nearest Neighbours (KNN) algorithm using the Iris dataset.

Steps:
1. Load Iris dataset
2. Select two features for visualization
3. Split data into train and test sets
4. Train custom KNN model
5. Predict test labels
6. Calculate accuracy
7. Plot decision boundary and data points
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from K_Nearest_Neighbours import K_Nearest_Neighbours



# Load Dataset
iris = datasets.load_iris()

# Use petal length & petal width (2D for visualization)
X = iris.data[:, 2:4]
y = iris.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

# Initialize and Train KNN
clf = K_Nearest_Neighbours(k=5)
clf.fit(X_train, y_train)

# Predictions and Accuracy
predictions = clf.predict(X_test)
accuracy = np.sum(predictions == y_test) / len(y_test)

print("Predictions:", predictions)
print("Accuracy:", accuracy)

# Decision Boundary Plot
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.1),
    np.arange(y_min, y_max, 0.1))

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(grid_points)
Z = np.array(Z).reshape(xx.shape)


plt.figure(figsize=(8, 6))

# Plot decision regions
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

# Plot training points
plt.scatter(
    X_train[:, 0], X_train[:, 1],
    c=y_train, cmap=cmap,
    edgecolor="k", label="Training Data"
)

# Plot test points
plt.scatter(
    X_test[:, 0], X_test[:, 1],
    c=y_test, cmap=cmap,
    marker="x", s=100, label="Test Data"
)

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("K-Nearest Neighbours Decision Boundary (k=5)")
plt.legend()
plt.show()
