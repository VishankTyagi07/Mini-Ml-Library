'''
This Class file is my work of creating a linear regression algorithm from scratch without using scikit learn
This algorithm is made by using only numpy library and mathematics 

Formulas used in the algorithms are

1- line formula - y= wx+b , 
where,
y= dependent variable
x= independent variable
w= weight of the data point
b= bias of the data point

2- gradient formulas - 
dw= 1/N * ∑ (X.T*(y_pred-y))
db= 1/N * ∑ (y_pred-y)
then 
w=w-lr(learning rate of the algortihm)*dw
b=b-lr(learning rate of the algorithm)*db

3- loss or Mean Squared Error formula:
loss= 1/N * ∑ (y_pred-y)^2

In the algorithm there are two main functions 
fit() and predict()
in fit()
all the formulas are used to train the data set given by the user

in predict

The test data is used to show the prediction made by the algorithm 

these functions are used in plots for regression visualization 

(The use of the algorithm is in the train.py file)
'''
import numpy as np

class LinearRegression:

    def __init__(self,lr=0.001,n_iters=1000)->None:
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None
        self.losses = []

    def fit(self,X,y):
        n_samples,n_features= X.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iters):
            y_pred= np.dot(X,self.weights) + self.bias

            dw= (1/n_samples) * np.dot(X.T, (y_pred-y))
            db= (1/n_samples) * np.sum(y_pred-y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db


            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)
    def predict(self,X):
        y_pred=np.dot(X,self.weights) + self.bias
        return y_pred
    