import numpy
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Dataset and split into train and test
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reduce training set (kNN is slooow)
X_train = X_train[:200]
y_train = y_train[:200]
X_test_original = X_test

print X_train[0]

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Normalize input to a value between 0 and 1
X_train = X_train / 255
X_test = X_test / 255

# Encode output labels as a binary matrix
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create classifier
classifier = KNeighborsClassifier()

# Train classifier
print 'Training classifier with', len(X_train), 'training examples'
classifier.fit(X_train, y_train)

# Use classifier to predict numbers from images in test set
predictions = classifier.predict(X_test)

# print classifier accuracy
print accuracy_score(y_test, predictions) * 100, '% accuracy'

# Print some predictions to console
for i in range(0,4):
	print predictions[i].argmax(axis=0)

# Create image-grid using matplotlib
plt.subplot(221)
plt.imshow(X_test_original[0])
plt.subplot(222)
plt.imshow(X_test_original[1])
plt.subplot(223)
plt.imshow(X_test_original[2])
plt.subplot(224)
plt.imshow(X_test_original[3])

# Display image-grid
plt.show()
