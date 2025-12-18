'''
This Class file is my work of creating a Polynomial regression algorithm from scratch without using scikit learn
This algorithm is made by using only numpy library and mathematics 

Formulas used in the algorithms are

1- formula - y= w*x+b, 
where,
y= dependent variable
x= independent variable
w= weight of the data point
b= bias of the data point

2- gradient formulas - 
dw= 2/N * ∑ (X.T*(y_pred-y))
then 
w=w-lr(learning rate of the algortihm)*dw

In the algorithm there are three main classes 
Polynomial Features , Standard Scaler and polynomial regression

In Polynomial features class
The X array shape of the data set is transformed in another array which is filled by ones and
then by using the degree a matrix with new values is formed

In Standard Scaler class
in fit()
The X is fitted and then the mean and standard deviation of the columns is found out for normalization
in transform()
The X matrix is copied and a normalized matrix is created using normalization formula

In polynomial regression class
in fit()
The dataset is fitted using the the previous classes functions
then by using the gradient funtion on the normalized matrix  the loop is iterated till the epoches are finished for training the data.

in Predict()
The test data is used to make predictions

these functions are used in plots for regression visualization
(The use of the algorithm is in the train.py file)
'''
import numpy as np
class Polynomialfeatures:
    def __init__(self,degree):
        self.degree=degree

    def transform(self,X):
        
        X=X.reshape(-1,1)
        m=X.shape[0]
        #here i have initialized complete matrix with 1 
        X_poly=np.ones((m,self.degree + 1))
        # use power function on other values except the biases
        for j in range(1,self.degree+1):
            X_poly[:,j]=X[:,0]**j

        return X_poly
    
class StandardScaler:

    def fit(self,X):
        
        # In this the 0th column is skipped as it contains all the biases
        self.mean=np.mean(X[:,1:],axis=0)
        self.std=np.std(X[:,1:],axis=0)
        self.std[self.std==0]=1

    def transform(self,X):
        X=X.copy()

        #Normalizing the matrix except the bias column
        X[:,1:]=(X[:,1:]-self.mean)/self.std

        return X

class PolynomialRegressionFromScratch:
    def __init__(self,degree,lr=0.001,iters=1000):
        self.degree=degree
        self.lr=lr
        self.iters=iters

        self.poly=Polynomialfeatures(degree)
        self.scaler=StandardScaler()

    def fit(self,X,y):

        X_poly=self.poly.transform(X)
        self.scaler.fit(X_poly)
        X_poly=self.scaler.transform(X_poly)

        m,n= X_poly.shape
        self.weights=np.zeros(n)

        for _ in range(self.iters):
            y_pred= np.dot(X_poly,self.weights)

            dw= (2/ m) * np.dot(X_poly.T, (y_pred-y))
            self.weights -= self.lr * dw
        
    def predict(self,X):
        X_poly=self.poly.transform(X)
        X_poly=self.scaler.transform(X_poly)
        return np.dot(X_poly,self.weights)
    