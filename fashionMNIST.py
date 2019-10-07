# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:52:49 2019

@author: msouf
"""

import tensorflow as tf
import keras as K

import numpy as np
import matplotlib.pyplot as plt

fashion_MNIST= K.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test)=fashion_MNIST.load_data()

classes=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(x_train[1])
plt.colorbar()
plt.grid(False)
plt.show()

x_train=x_train/255
x_test=x_test/255

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i],cmap=plt.cm.binary)
    plt.xlabel(classes[y_train[i]])
plt.show()

model=K.Sequential()
model.add(K.layers.Flatten(input_shape=(28,28)))
model.add(K.layers.Dense(units=128, activation='relu'))
model.add(K.layers.Dense(units=10, activation='softmax'))
model.compile(optimizer='adam',metrics=['accuracy'],loss='sparse_categorical_crossentropy')

model.fit(x_train,y_train,epochs=5)

test_loss,test_acc=model.evaluate(x_test,y_test)

print('Test accuracy',test_acc)