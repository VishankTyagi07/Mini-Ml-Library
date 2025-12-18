'''
In this File The use of Logistic Regression algorithm is demonstrated

First i Have taken X and y arrays with iris dataset from sklearn for 2 features

Then i have used train test split function to split the data in a 80 /20 split, also with random state for reproducibility

Then the Logistic Regression class is called and the data is then predicted using the algorithm 

The acuuracy of the algorithm  is shown.

Then prediction gives the lables for the new data points by displaying it in a plot.
'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from LogisticsRegression import LogisticRegression

# Load dataset
iris = load_iris()
X = iris.data[:100,:2]     
y = iris.target[:100]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
clf = LogisticRegression(lr=0.1, n_iters=1000)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Predict for grid points
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(grid_points)
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3)

# Plot training points
plt.scatter(
    X_train[:, 0], X_train[:, 1],
    c=y_train, edgecolors="k", s=50)

plt.xlabel("Sepal Length (standardized)")
plt.ylabel("Sepal Width (standardized)")
plt.title("Logistic Regression Decision Boundary (Iris Dataset)")
plt.show()
