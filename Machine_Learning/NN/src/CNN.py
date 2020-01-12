from patient.patientgroup import PatientGroup
from config import config 
from config import hyper_cnn
from dataCNN import generate
import pandas as pd
import numpy as np

# from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

import itertools
import os

patient_groups = [
    PatientGroup(config.basepath.format(groupid=1)),
    # PatientGroup(config.basepath.format(groupid=4)),
    PatientGroup(config.basepath.format(groupid=2)),
    PatientGroup(config.basepath.format(groupid=3)),
]


# Retreiving the data like we know it (5 exercise * column-count * frame count)
X, np_test_x, y, np_test_y = generate(patient_groups)


X = tf.convert_to_tensor(X)
y = tf.convert_to_tensor(y)

# Make random permutation
perm = tf.random.shuffle(tf.range(tf.shape(X)[0]), seed = 420)

# Reorder according to permutation
X = tf.gather(X, perm, axis=0)
y = tf.gather(y, perm, axis=0)

def CNN2(train_x, train_y, test_x, test_y):
#TODO 
#get better dropouts
#tune hypers


    # preview
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_x[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        test = train_y[0] 
        plt.xlabel(str(train_y[i].numpy()))

    plt.show()
    
    frame_x = 100
    frame_y = 40
    channels = 3
    output_size = 3

    model = models.Sequential()
    model.add(layers.Conv2D(hyper_cnn.h_layer_size_s, hyper_cnn.conv2D_size, activation='relu', input_shape=(frame_x, frame_y, channels)))
    model.add(layers.MaxPooling2D(hyper_cnn.maxpool_2D))
    model.add(layers.Conv2D(hyper_cnn.h_layer_size_b, hyper_cnn.conv2D_size, activation='relu'))
    model.add(layers.MaxPooling2D(hyper_cnn.maxpool_2D))
    model.add(layers.Conv2D(hyper_cnn.h_layer_size_b, hyper_cnn.conv2D_size, activation='relu'))


    model.add(layers.Flatten())
    model.add(layers.Dense(hyper_cnn.h_layer_size_b, activation='relu'))
    model.add(layers.Dense(output_size, activation='softmax'))
    
    model.summary()

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    history = model.fit(train_x, train_y, epochs=hyper_cnn.epochs, 
                    validation_data=(test_x, test_y), 
                    callbacks=[hyper_cnn.tensorboard]).history

    
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=hyper_cnn.evaluate_verbose)
    print(test_acc)
    plt.show()

CNN2(X, y, np_test_x, np_test_y)
