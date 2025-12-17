'''
This Class file is my work of creating a K_nearest_neighbors algorithm from scratch without using scikit learn
This algorithm is made by using only numpy library and mathematics 

Formulas used in the algorithms are

1- Euclidean Distance Formula
   √∑(x1-x2)^2 


In the algorithm there are two main functions 
fit() and predict()
in fit()
The Data set is fitted in the algorithm

in predict
I have made the predictions in the predict function using the _predict helper function

in _predict helper function
I have used the distance of the x point in the distances(variable which gets the distances between all the x and X_train data)
Then those diatances are used by sorting them with the k points
Then using Counter class most common lable is found for the new data points which is the answer of the algorithm


(The use of the algorithm is in the train.py file)
'''
import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    distance= np.sqrt(np.sum((x1-x2)**2))
    return distance

class K_Nearest_Neighbours:

    def __init__(self, k=int):
        self.k=k

    def fit(self,X,y):
        self.X_Train=X
        self.y_train=y

    def predict(self, X):
        predictions=[self._predict(x) for x in X ]
        return predictions

    def _predict(self,x):
        #distances
        distances= [euclidean_distance(x, x_train) for x_train in self.X_Train]
        #k closest indices
        k_indices=np.argsort(distances)[:self.k]
        K_Nearest_lables=[int(self.y_train[i]) for i in k_indices]
        # most common lables
        most_common = Counter(K_Nearest_lables).most_common()
        return most_common[0][0]