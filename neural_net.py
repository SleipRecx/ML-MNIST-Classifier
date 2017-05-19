import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt


def baseline_model(num_pixels, num_classes):
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def main():
    # Set random seed so results of your script are reproducible.
    seed = 7
    numpy.random.seed(seed)

    # Load Dataset and split into train and test
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_test_original = X_test

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

    # build the model
    model = baseline_model(num_pixels, num_classes)

    # Fit the model, aka train that shit
    #model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200, verbose=2)

    predictions = model.predict_on_batch(X_test)
    print predictions[20].argmax(axis=0)
    print predictions[21].argmax(axis=0)
    print predictions[22].argmax(axis=0)
    print predictions[23].argmax(axis=0)

    plt.subplot(221)
    plt.imshow(X_test_original[20])
    plt.subplot(222)
    plt.imshow(X_test_original[21])
    plt.subplot(223)
    plt.imshow(X_test_original[22])
    plt.subplot(224)
    plt.imshow(X_test_original[23])
    plt.show()



main()
