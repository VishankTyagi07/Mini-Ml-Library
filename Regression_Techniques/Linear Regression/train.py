'''
In this File The use of  Linear regression algorithm is demonstrated

First uing sklearn i have imported datasets class which gives me a random dataset values
and by using make regression function i have used 200 data samples and 2 featues ,
with some noise to make the model robust in training and given a random state for reproducibility.

Then i have used train test split function to split the data in a 80 /20 split, also with random state for reproducibility

Then the linear regression class is called and the data is then predicted using the algorithm 
and the data is then shown using matplotlib.
'''
from LinearRegression import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X,y=datasets.make_regression(n_samples=200,n_features=2,noise=20,random_state=4)

X_train,X_test,y_train ,y_test=train_test_split(X,y, test_size=0.2,random_state=1234)
#Actual Data plot
fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
plt.show()

reg=LinearRegression(lr=0.01)
reg.fit(X_train,y_train)

#iteration to loss plot
plt.figure(figsize=(8,5))
plt.plot(reg.losses)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("Training Loss Curve")
plt.show()


predictions=reg.predict(X_test)

y_pred = reg.predict(X_test)

#actual y to predicted y plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color="blue", s=20)
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="black", linestyle="--")
plt.show()

