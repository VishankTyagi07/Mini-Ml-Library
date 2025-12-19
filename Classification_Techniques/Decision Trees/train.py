"""
This file demonstrates the implementation and visualization of the
Decision Tree algorithm using the Iris dataset.

Steps:
1. Load Breast Cancer dataset
2. Select two features for visualization
3. Split data into train and test sets
4. Train custom Decision Tree model
5. Predict test labels
6. Calculate accuracy
7. Plot decision boundary and data points
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(acc)

plt.figure()
plt.imshow(cm)
plt.title("Decision Tree – Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

# Add text values inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()
