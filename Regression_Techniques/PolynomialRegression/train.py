"""
This file demonstrates the implementation and visualization of the
Polynomial Regression algorithm using a synthetic dataset generated
with sklearn.

Steps:
1. Generate random dataset
2. Select features for visualization
3. Split data into train and test sets
4. Train custom Polynomial Regression model
5. Predict output values
6. Plot regression line and data points
"""
import numpy as np
import matplotlib.pyplot as plt
from PolynomialRegression import PolynomialRegressionFromScratch
from sklearn.model_selection import train_test_split


# Dataset
np.random.seed(0)
X = np.linspace(-10, 10, 100)           # 100 points
y = 2*X**2 + 3*X + 5 + np.random.randn(100)*10  # Quadratic with noise

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Raw data
plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color="blue", label="Training Data")
plt.scatter(X_test, y_test, color="green", label="Test Data")
plt.title("Raw Data")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()


# Train Polynomial Regression
model = PolynomialRegressionFromScratch(degree=2, lr=0.001, iters=5000)
model.fit(X_train, y_train)

# Generate smooth X for plotting
X_curve = np.linspace(X.min(), X.max(), 500)  # dense X for smooth line
y_curve = model.predict(X_curve)


#  Plotting Fit
plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color="blue", label="Training Data")
plt.scatter(X_test, y_test, color="green", label="Test Data")
plt.plot(X_curve, y_curve, color="red", linewidth=2, label=f"Polynomial Fit (degree=2)")
plt.title("Polynomial Regression Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()



print("Learned weights:", model.weights)
