import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_output = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_output)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = -(1/n_samples) * np.sum(
                y * np.log(predictions + 1e-9) +
                (1 - y) * np.log(1 - predictions + 1e-9)
            )
            self.losses.append(loss)

    def predict(self, X):
        X = np.array(X)
        linear_output = np.dot(X, self.weights) + self.bias
        probs = sigmoid(linear_output)
        return (probs >= 0.5).astype(int)
