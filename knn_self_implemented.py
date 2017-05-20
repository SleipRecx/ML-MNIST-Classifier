import os
import numpy as np
import time
import math
import operator
import matplotlib.pyplot as plt

from collections import Counter
from scipy.spatial import distance
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

class KNearestNeighbors():
    def fit(self, X_train, y_train, k):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.k_closest(row)
            predictions.append(label)
        return predictions

    def k_closest(self, row):
        distances = []
        for i in range(len(self.X_train)):
            current_dist = distance.euclidean(row, self.X_train[i])
            distances.append((current_dist, self.y_train[i]))
        distances.sort(key=operator.itemgetter(0))
        common_list = distances[:self.k]
        counter = Counter(element[1] for element in common_list)
        return counter.most_common()[0][0]



# Import dataset and place it in current dir
mnist = fetch_mldata('MNIST original', data_home=os.path.dirname(os.path.realpath(__file__)))
mnist.data, mnist.target = resample(mnist.data, mnist.target)
X = mnist.data[:2000]
y = mnist.target[:2000]

# Split data into test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# Create classifier
classifier = KNearestNeighbors()


# Choose k value based on training examples
k = int(math.sqrt(len(X_train)))

# Train classifier
print 'Training classifier with', len(X_train), 'training examples and k =', k
start = time.time()
classifier.fit(X_train, y_train, k)
print 'Training took', (time.time() - start), 'Seconds'

# Use classifier to predict numbers from images in test set
print 'Predicting labels for test set with', len(y_test), 'tests'
start = time.time()
predictions = classifier.predict(X_test)
print 'Predicting tests took', (time.time() - start), 'Seconds'

# print classifier accuracy
print accuracy_score(y_test, predictions) * 100, '% accuracy'
