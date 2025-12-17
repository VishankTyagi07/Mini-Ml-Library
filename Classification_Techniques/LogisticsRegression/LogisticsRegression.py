'''
This Class file is my work of creating a Logistic Regression algorithm from scratch without using scikit learn
This algorithm is made by using only numpy library and mathematics 

Formulas used in the algorithms are

1- Sigmoid Formula
    1/1+e^-x

2- Linear line formmula
    Y = weight*X + bias

3- gradient formulas - 
    dw= 1/N * ∑ (X.T*(y_pred-y))
    db= 1/N * ∑ (y_pred-y)
    then 
    w=w-lr(learning rate of the algortihm)*dw
    b=b-lr(learning rate of the algorithm)*db

3- loss formula:
    -1 / N * ∑ (y * log (p(y) )) + (1 - y) * log(1 - ( p(y) ))

In the algorithm there are two main functions 
fit() and predict()
in fit()
The Data set is fitted in the algorithm using the forumlas.

in predict
I have made the predictions in the predict function using the condition 
which says that if the data point lies below 0.5 it is classified as 0 and if it is more than 0.5 is it classified 1

(The use of the algorithm is in the train.py file)
'''
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
