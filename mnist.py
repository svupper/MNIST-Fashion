# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 19:17:43 2019

@author: msouf
"""

import matplotlib.pyplot as plt
import numpy as np

import keras as k
import tensorflow as tf
from keras import Sequential

data=k.datasets.mnist
(x_train, y_train), (x_test, y_test)=data.load_data()

plt.figure()
plt.imshow(x_train[0])
plt.xlabel(y_train[0])
plt.colorbar()
plt.grid(False)
plt.show

classes=['0','1','2','3','4','5','6','7','8','9']


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i],cmap='binary')
    plt.xlabel(classes[y_train[i]])   
plt.show()

model=Sequential()
model.add(k.layers.Flatten(input_shape=(28,28)))
model.add(k.layers.Dense(units=128,activation='relu'))
model.add(k.layers.Dense(units=10,activation='softmax'))
model.compile(optimizer='adam',metrics=['accuracy'],loss='sparse_categorical_crossentropy')
hist=model.fit(x_train,y_train,epochs=50)
test_loss,test_acc=model.evaluate(x_test,y_test)

print('Test Accuracy',test_acc)
#%%

predictions=model.predict(x_test)

#%%

plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(1,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i],cmap='binary')
    plt.xlabel(np.argmax(predictions[i]))   
plt.show()

#%%
dict=hist.history
acc=dict['acc']
loss=dict['loss']

epochs=len(acc)
#%%
plt.figure()
plt.plot()