import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
# train_img: 60,000 28x28 images, test_img: 10,000 28x28 images
(train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()

# define label/class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
               'Ankle boot']

# preprocess data: normalize data to fall between 0 and 1
train_img = train_img / 255.0
test_img = test_img / 255.0
print(train_img.shape)
# input layer is flattened images
# hidden layer has 128 neurons using relu as activation function
# output layer has 10 neurons using softmax (output a probability distribution)
# dense (fully-connected): a neuron is connected to all neurons of previous layer
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])

# # optimizer: how model is updated based on data and cost function
# # loss: cost function
# # metrics: used to monitor training and testing
# model.compile(optimizer=tf.train.AdamOptimizer(),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# # training stage
# # epochs: number of iterations through the training data
# model.fit(train_img, train_labels, epochs=5)
#
# # testing stage
# test_loss, test_acc = model.evaluate(test_img, test_labels)
# print('Test Accuracy:', test_acc)
#
# predictions = model.predict(test_img)
#
# # show predicted and correct labels for first 25 test images
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid('off')
#     plt.imshow(test_img[i], cmap='binary')
#     predicted_label = np.argmax(predictions[i])
#     true_label = test_labels[i]
#     color = 'red'
#     if predicted_label == true_label:
#         color = 'green'
#     plt.xlabel('{} ({})'.format(class_names[predicted_label],
#                                 class_names[true_label],), color=color)
# plt.show()
