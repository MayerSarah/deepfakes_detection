import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras import regularizers, optimizers


class DeepfakeModel():
    def __init__(self,
                 num_classes=2,
                 img_height=150,
                 img_width=150,
                 optimizer='adam', 
                 activation='relu', 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'],
                 batch_size=32,
                 epochs=5
                ):
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        self.optimizer = optimizer
        self.activation = activation
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs

    def create_model(self):

        """
        Create the neural network structure and compile the model.
        """
        
        model = Sequential([
          layers.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)),
          layers.Conv2D(16, 3, padding='same', activation=self.activation),
          layers.MaxPooling2D(),
          layers.Conv2D(32, 3, padding='same', activation=self.activation),
          layers.MaxPooling2D(),
          layers.Conv2D(64, 3, padding='same', activation=self.activation),
          layers.MaxPooling2D(),
          layers.Flatten(),
          layers.Dense(128, activation=self.activation),
          layers.Dense(self.num_classes)
        ])

        model.compile(optimizer=self.optimizer,
              loss=self.loss,
              metrics=self.metrics)
        return model

    
    def train_model(self, model, train_set, validation_set):

        """
        Fit e model on the train set.
        """

        model.fit(
        train_set,
        steps_per_epoch=train_set.samples // self.batch_size,
        validation_data = validation_set,
        validation_steps = validation_set.samples // self.batch_size,
        epochs=self.epochs
        )


    def save_model(self, model, model_name: str):

        """
        Predict on a test set.
        """
        model.save(f"saved_models/{model_name}")
