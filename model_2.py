# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 00:56:30 2019

@author: Dell
"""

import tensorflow as tf
from tensorflow import keras

def create_model(input_shape,target_class):
    fine_tune_layer=290
    base_model=tf.keras.applications.inception_v3.InceptionV3(input_shape=input_shape,weights="imagenet",include_top=False)
    base_model.trainable=True;
    for layer in base_model.layers[:fine_tune_layer]:
        layer.trainable=False
    model=keras.Sequential([
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1024,activation="relu"),
            keras.layers.Dropout(rate=0.4),
            keras.layers.Dense(256,activation="relu"),
            keras.layers.Dropout(rate=0.4),
            keras.layers.Dense(38,activation="softmax")
        
            ])
    return model