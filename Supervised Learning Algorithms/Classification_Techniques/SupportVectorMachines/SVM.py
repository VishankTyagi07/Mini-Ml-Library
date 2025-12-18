'''
This Class file is my work of creating a Support Vector Machine algorithm from scratch without using scikit learn
This algorithm is made by using only numpy library and mathematics 

Formulas used in the algorithms are

1- Formula for centre line and hyperplanes-- y*(w*x-b)>=1
where,
y in this formula is the index of the y array
w is the weight  
x in this forumla is the index of x array
b is the bias  

The centre line of the Support vector is when the equaltion is zero 
that means where the equation y*(w*x-b) = 0 there the entre line is created,
And
From that centre line for where the equation have -1 or 1value there the  hyperplanes are made.


In the algorithm there are two main functions 
fit() and predict()
in fit()
The Data set is fitted in the algorithm using the formulas for weights, lines and bias

in predict()
I have made the predictions in the predict function using the approprite formula which will predict the data point in the SVM.

(The use of the algorithm is in the train.py file)
'''
import numpy as np

class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        # init weights
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
