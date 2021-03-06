from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, layers
from tensorflow.keras.datasets import mnist
import random

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train, X_test = np.array(X_train, np.float32), np.array(X_test, np.float32)
X_train, X_test = X_train / 255., X_test / 255.

# training parameters.
classes = 10
learning_rate = 0.001
train_steps = 200
batch_size = 128
display_step = 10

# slice data
train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# network parameters.
conv1_filters = 32 # number of filters for 1st conv layer.
conv2_filters = 64 # number of filters for 2nd conv layer.
fc1_units = 1024 # number of neurons for 1st fully-connected layer.


class ConvNet(Model):
    # Set layers.
    def __init__(self):
        super(ConvNet, self).__init__()
        # convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = layers.Conv2D(32, kernel_size=5, activation=tf.nn.relu)
        # max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        # convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)
        # max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        # flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()
        # fully connected layer.
        self.fc1 = layers.Dense(1024)
        # apply Dropout (if if_training is False, dropout is not applicable).
        self.dropout = layers.Dropout(rate=0.5)
        # output layer, class prediction.
        self.out = layers.Dense(classes)

    # Set forward pass.
    def call(self, x, if_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=if_training)
        x = self.out(x)
        if not if_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x

# network model
conv_net = ConvNet()

def cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)

# Accuracy metric.
def accuracy(Y_predict, Y_test_mnist):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(Y_predict, 1), tf.cast(Y_test_mnist, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.Adam(learning_rate)

def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = conv_net(x, if_training=True)
        # Compute loss.
        loss = cross_entropy_loss(pred, y)

    # Variables to update, i.e. trainable variables.
    trainable_variables = conv_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))

# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(train_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    if step % display_step == 0 or step == 1:
        pred = conv_net(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

pred = conv_net(X_test)
print("Test Accuracy: %f" % accuracy(pred, Y_test))

n_images = 20
rand = random.randint(1, 201)
test_images = X_test[rand*n_images:n_images*(1+rand)]
predictions = conv_net(test_images)

# Display image and model prediction.
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % np.argmax(predictions.numpy()[i]))