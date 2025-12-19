'''
In this File The use of Decision Tree algorithm is demonstrated

First i Have taken X and y arrays with iris dataset from sklearn for 2 features

Then i have used train test split function to split the data in a 80 /20 split, also with random state for reproducibility

Then the Decision Tree class is called and the data is then predicted using the algorithm 

The acuuracy of the algorithm  is shown.

'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(acc)