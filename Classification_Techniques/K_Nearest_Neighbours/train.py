'''
In this File The use of  K_ Nearest_ Neighbours algorithm is demonstrated

First i Have taken X and y arrays with iris dataset from sklearn

Then i have used train test split function to split the data in a 80 /20 split, also with random state for reproducibility
Then the raw data is plotted for data visualization
Then the K_Nearest_neighbour class is called and the data is then predicted using the algorithm 

The prediction gives the lables for the new data points 
then the acuuracy of the algorithm is shown.
'''
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from K_Nearest_Neighbours import K_Nearest_Neighbours

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()


clf = K_Nearest_Neighbours(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)
