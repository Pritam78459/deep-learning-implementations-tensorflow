from __future__ import absolute_import, print_function, unicode_literals, division
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.keras import models

(training_images, training_labels), (testing_images, testing_labels) = ks.datasets.fashion_mnist.load_data()

"""print(f"Train shape: {training_images.shape}")
print(f"Train Labels shape: {training_labels.shape}")
print(f"Test Shape: {testing_images.shape}")
print(f"Test labels: {testing_labels.shape}")"""

# image preprocessing 

training_images = training_images / 255.0
testing_images = testing_images / 255.0

input_data_shape = (28,28)
hidden_activation_function = 'relu'
output_activation_function = 'softmax'

# building model

nn_model = models.Sequential()
nn_model.add(ks.layers.Flatten(input_shape = input_data_shape, name='Input_layer'))
nn_model.add(ks.layers.Dense(32, activation = hidden_activation_function, name='Hidden_layer'))
nn_model.add(ks.layers.Dense(10, activation = output_activation_function, name='Output_layer'))
print(nn_model.summary())

optimizer = 'adam'
loss_function = 'sparse_categorical_crossentropy'
metric = ['accuracy']
nn_model.compile(optimizer = optimizer, loss = loss_function, metrics = metric)
nn_model.fit(training_images, training_labels, epochs=10)

training_loss, training_accuracy = nn_model.evaluate(training_images, training_labels)
print(f"Training Accuracy: {training_accuracy} Training Loss: {training_loss}")

test_loss, test_accuracy = nn_model.evaluate(testing_images, testing_labels)
print(f"Test Accuracy: {test_accuracy} Test Loss: {test_loss}")