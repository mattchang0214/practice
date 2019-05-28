import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

_HIDDEN_NEURONS = 6
_OUT_NEURONS = 3


def scale_data(arr):
    arr[:, 0] = (arr[:, 0] - 4.3) / 3.6
    arr[:, 1] = (arr[:, 1] - 2.0) / 2.4
    arr[:, 2] = (arr[:, 2] - 1.0) / 5.9
    arr[:, 3] = (arr[:, 3] - 0.1) / 2.4


if __name__ == '__main__':
    # preallocate matrix for iris data and labels
    data = np.empty((150, 4))
    labels = np.empty(150)
    for i in range(3):
        labels[50*i:50*(i+1)].fill(i)

    # define classes
    iris_classes = ['Setosa', 'Versicolour', 'Virginica']

    # fill in matrix with data
    with open('A4 - Iris data.txt') as f:
        num = 0
        for line in f:
            data[num, :] = line.split(',')[:-1]
            num += 1

    # preprocess data: normalize data to fall between 0 and 1
    scale_data(data)

    # mix up data
    for _ in range(100):
        # only swap within each class to ensure same # of samples in training
        for j in range(3):
            a, b = random.sample(range(j * 50, (j + 1) * 50), 2)
            # # slower swap - advanced indexing: extracts rows a & b as 2D arr
            # data[[a, b]] = data[[b, a]]

            # faster swapping method
            copy = data[a, :].copy()
            data[a, :], data[b, :] = data[b, :], copy
            labels[a], labels[b] = labels[b], labels[a]

    # portion data for different purposes
    training_data = np.vstack((data[0:40], data[50:90], data[100:140]))
    training_labels = np.concatenate((labels[0:40], labels[50:90], labels[100:140]))
    test_data = np.vstack((data[40:50], data[90:100], data[140:150]))
    test_labels = np.concatenate((labels[40:50], labels[90:100], labels[140:150]))

    model = keras.Sequential([
        keras.layers.Dense(_HIDDEN_NEURONS, input_shape=(4,), activation=tf.nn.relu),
        keras.layers.Dense(_OUT_NEURONS, activation=tf.nn.softmax),
    ])

    # optimizer: how model is updated based on data and cost function
    # loss: cost function
    # metrics: used to monitor training and testing
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # training stage
    # epochs: number of iterations through the training data
    model.fit(training_data, training_labels, epochs=500)

    # testing stage
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print('Test Accuracy:', test_acc)

    predictions = model.predict(test_data)
