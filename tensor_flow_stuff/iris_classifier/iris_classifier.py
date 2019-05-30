import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

_HIDDEN_NEURONS = 6
_OUT_NEURONS = 3


def scale_data(arr):
    # normalize data to fall between 0 and 1
    data_min = np.amin(arr, axis=0)
    data_max = np.amax(arr, axis=0)
    arr = (arr - data_min) / (data_max - data_min)


# fastest way to shuffle 150 samples - 0.04758 ms per loop
def shuffle_data(arr):
    # only swap within each class to ensure same # of samples in training
    for i in range(3):
        # generate random numbers and sort the indices
        idx = np.argsort(np.random.random(50)) + 50 * i
        arr[i * 50:(i + 1) * 50, :] = arr[idx, :]


# preallocate matrix for iris data and labels
data = np.empty((150, 4))
labels = np.empty(150)
for n in range(3):
    labels[50*n:50*(n+1)].fill(n)

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

shuffle_data(data)

# portion data for different purposes
training_data = np.vstack((data[0:40], data[50:90], data[100:140]))
training_labels = np.concatenate((labels[0:40], labels[50:90], labels[100:140]))
test_data = np.vstack((data[40:50], data[90:100], data[140:150]))
test_labels = np.concatenate((labels[40:50], labels[90:100], labels[140:150]))

model = keras.Sequential([
    keras.layers.Dense(_HIDDEN_NEURONS, input_shape=(4,), activation=tf.nn.relu),
    keras.layers.Dense(_OUT_NEURONS, activation=tf.nn.softmax),
])

model.summary()

# optimizer: how model is updated based on data and cost function
# loss: cost function
# metrics: used to monitor training and testing
# sparse_categorical_crossentropy: for integer encoded outputs rather than
# one-hot encoded outputs
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


# --- APPENDIX ---

# Fisher-Yates shuffle: intuitive but slowest - 2.58197 ms per loop
def shuffle_data_v1(arr):
    for _ in range(100):
        # only swap within each class to ensure same # of samples in training
        for j in range(3):
            a, b = random.sample(range(j * 50, (j + 1) * 50), 2)
            # # slower swap - advanced indexing: extracts rows a & b as 2D arr
            # data[[a, b]] = data[[b, a]]

            # faster swapping method
            copy = arr[a, :].copy()
            arr[a, :], arr[b, :] = arr[b, :], copy


# shuffle data directly: slower than shuffling index - 0.21279 ms per loop
def shuffle_data_v2(arr):
    for i in range(3):
        np.random.shuffle(arr[i * 50:(i + 1) * 50])


# shuffle indices and reassign: slower than argsort index - 0.17388 ms per loop
def shuffle_data_v3(arr):
    for i in range(3):
        idx = np.arange(i * 50, (i + 1) * 50)
        np.random.shuffle(idx)
        arr[i * 50:(i + 1) * 50, :] = arr[idx, :]


# permutate indices and reassign: slower than argsort index - 0.18144 ms per loop
def shuffle_data_v4(arr):
    for i in range(3):
        idx = np.random.permutation(50) + 50 * i
        arr[i * 50:(i + 1) * 50, :] = arr[idx, :]
