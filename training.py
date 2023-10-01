#importing libraries
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Dense, concatenate
from keras.models import Model
#!pip install git+https://github.com/qubvel/classification_models.git
from classification_models.keras import Classifiers
from tensorflow.keras.models import load_model, save_model
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


#IMAGE GENERATOR 
train_datagen = ImageDataGenerator(horizontal_flip=True,
    width_shift_range=[-0.1, 0.2],
    height_shift_range=[-0.1, 0.2],
    brightness_range = [0.5, 1.5],
    zoom_range=0.2,
    rotation_range=10,
    rescale=1/255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1/255)

#Batch Size = 20
BATCH_SIZE = 20
train_generator = train_datagen.flow_from_directory(directory='/Users/saijena/Desktop/Assign_images3/',
                                                      target_size = (224,224),
                                                      class_mode = 'categorical',
                                                      shuffle = False,
                                                      subset = 'training',
                                                      batch_size = BATCH_SIZE)

valid_generator = train_datagen.flow_from_directory(directory='/Users/saijena/Desktop/Assign_images3/',
                                                      target_size = (224,224),
                                                      class_mode = 'categorical',
                                                      shuffle = False,
                                                      subset = 'validation',
                                                      batch_size = BATCH_SIZE)



IMG_SHAPE = (224,224,3)
base_model = tf.keras.applications.ResNetRS101(input_shape=IMG_SHAPE, include_top=False, weights="imagenet") #ResNetRS101
#keeping all the layers as trainable
for layer in base_model.layers:
    layer.trainable =  True
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
#dropout_layer = tf.keras.layers.Dropout(0.3)(global_average_layer)
outputs = tf.keras.layers.Dense(5, activation='softmax')(global_average_layer)
model = tf.keras.models.Model(inputs = base_model.input, outputs = outputs)
best_model_checkpoint = tf.keras.callbacks.ModelCheckpoint('./delivery_weight.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, ema_momentum=0.95) , loss = "categorical_crossentropy", metrics = ["accuracy"]) #tf.keras.optimizers.Adam(learning_rate=1e-4) 
history = model.fit_generator(train_generator, epochs=10, validation_data=valid_generator, verbose=1, callbacks=[best_model_checkpoint, ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=2, min_lr=0.000001), tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)])
save_model(model, "./delivery_model.h5")