# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:10:03 2019

@author: Dell
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from model_2 import create_model
from dataset_tf import dataset
from math import exp

'''Hypertuning'''
learning_rate=1e-10
batch_size=64
momentum=0.9
nesterov=True
metrics=[keras.metrics.categorical_accuracy]
epochs=1
valid_step=int(20e3/batch_size)
train_step=5000
epoch_step=1    #no of epoch after which lr decays
lr_step=exp(1)    #factor by which lr decays
activation_fn="elu"
Batch_norm=False
kernel_init="he_normal"
dropout_rate=0.5
target_class=38
loss_list=np.array([])


'''Function to create model in Keras'''
model=create_model(input_shape=[256,256,3],target_class=target_class)



'''Dataset Create'''
'''
obj=dataset("train",batch_size=batch_size)
train_ds=obj.dataset_ready()
obj=dataset("valid",batch_size=batch_size)
valid_ds=obj.dataset_ready()


def loss(model,x,y):
    y_=model(x)
    loss_=tf.losses.categorical_crossentropy(y,y_)
    return loss_

def grad(model,x,y):
    with tf.GradientTape() as tape:
        loss_=loss(model,x,y)
    return loss_,tape.gradient(loss_,model.trainable_variables)

lr=1e-10
steps=0
optimizer=tf.optimizers.Adam(learning_rate=lr,beta_1=0.9,beta_2=0.999)

def train_step(model,x,y):
    loss_,grad_=grad(model,x,y)
    optimizer.apply_gradients(zip(grad_,model.trainable_variables))



for i in range(epochs):
    for x,y in train_ds:
        print("step-",steps)
        loss_,grad_=grad(model,x,y)
        optimizer.apply_gradients(zip(grad_,model.trainable_variables))
        loss_list=np.append(loss_list,[loss_])
        learning_rate=learning_rate*((1.3)**steps)
        steps=steps+1
        if(steps==train_step):
            break
        print("loss-",loss)
        
        


'''


















