from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import numpy as np


class KerasNeuralNetwork(object):
    def __init__(self, input_shape, num_classes, regularizer=regularizers.l2(0.01), activation='relu', optimizer='adam'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.regularizer = regularizer
        self.activation = activation
        self.optimizer = optimizer
        self.model = Sequential()

    def compile_model(self):
        self.model.add(Conv2D(64, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation,
                              input_shape=self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def train(self, X, y, validation_data=None, batch_size=64, epochs=3000):
        datagen = ImageDataGenerator(
            rotation_range=np.pi / 4,
            shear_range=np.pi / 5,
            width_shift_range=0.4,
            height_shift_range=0.4,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.5,
            fill_mode='nearest')

        return self.model.fit_generator(datagen.flow(X, y, batch_size=batch_size), validation_data=validation_data,
                                        steps_per_epoch=len(X) / batch_size, epochs=epochs)

    def predict(self, X, batch_size=64):
        y = self.model.predict(X, batch_size=batch_size)
        return y


