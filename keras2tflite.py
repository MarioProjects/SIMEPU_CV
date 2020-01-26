#!/usr/bin/env python
# coding: utf-8
# !/usr/bin/env python
# coding: utf-8

# https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e
# https://www.tensorflow.org/lite/guide/get_started
# https://www.youtube.com/watch?v=MZx1fhbL2q4
# https://www.tensorflow.org/guide/keras/overview

import os
import keras
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

if os.environ.get('SIMEPU_DATA') is not None:
    data_dir = os.environ.get('SIMEPU_DATA')
else:
    assert False, "Please set SIMEPU_DATA environment variable!"

batch_size = 128
img_height, img_width = 224, 224
nb_epochs = 50

# ------------------------------------------------------------

model = tf.keras.Sequential()

model.add(tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))

model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(9, activation='softmax'))  # final layer with softmax activation

# ------------------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # set validation split
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')  # set as training data

validation_generator = train_datagen.flow_from_directory(
    data_dir,  # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')  # set as validation data

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=nb_epochs)

# Save weights to a TensorFlow Checkpoint file
model.save_weights('./keras_weights/my_model')

# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights('./keras_weights/my_model')

# ------------------------------------------------------------

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

open("model.tflite", "wb").write(tflite_model)
