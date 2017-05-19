import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import time

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

class NearestNeighbor():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_distance = distance.euclidean(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            current_dist = distance.euclidean(row, self.X_train[i])
            if current_dist < best_distance:
                best_distance = current_dist
                best_index = i
        return self.y_train[best_index]


# Import dataset and place it in current dir
mnist = fetch_mldata('MNIST original', data_home=os.path.dirname(os.path.realpath(__file__)))
mnist.data, mnist.target = shuffle(mnist.data, mnist.target)
X = mnist.data[:2000]
y = mnist.target[:2000]

# Split data into test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# Create classifier
classifier = NearestNeighbor()

# Train classifier
print 'Training classifier with', len(X_train), 'training examples'
start = time.time()
classifier.fit(X_train, y_train)
print 'Training took', (time.time() - start), 'Seconds'

# Use classifier to predict numbers from images in test set
print 'Predicting labels for test set with', len(y_test), 'tests'
start = time.time()
predictions = classifier.predict(X_test)
print 'Predicting tests took', (time.time() - start), 'Seconds'

# print classifier accuracy
print accuracy_score(y_test, predictions) * 100, '% accuracy'
