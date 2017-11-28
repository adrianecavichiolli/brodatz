from keras import regularizers, applications
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import Activation, BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from keras.layers import Merge, Concatenate

import numpy as np


class KerasNeuralNetwork(object):
    def __init__(self, input_shape, num_classes, regularizer=regularizers.l2(0.01), activation='relu', optimizer='adam'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.regularizer = regularizer
        self.activation = activation
        self.optimizer = optimizer
        self.model = Sequential()

    def compile_model_gpu(self):
        self.model.add(
            Conv2D(32, (5, 5), kernel_regularizer=self.regularizer, input_shape=self.input_shape))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (5, 5), kernel_regularizer=self.regularizer))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model = multi_gpu_model(self.model, gpus=2)

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def compile_model_finetuning_2(self):
        initial_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)

        for layer in initial_model.layers:
            layer.trainable = False

        x = Flatten()(initial_model.output)
        x = Dense(512, activation=self.activation)(x)
        preds = Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax')(x)

        self.model = Model(initial_model.input, preds)

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def compile_model_finetuning(self):
        initial_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)

        for layer in initial_model.layers:
            layer.trainable = False

        x = Flatten()(initial_model.output)
        x = Dense(256, activation=self.activation)(x)
        preds = Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax')(x)

        self.model = Model(initial_model.input, preds)

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def compile_model_11(self):
        self.model.add(
            Conv2D(128, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation, input_shape=self.input_shape))
        self.model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(512, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])


    def compile_model_10(self):
        shift_model = Sequential()
        shift_model.add(Conv2D(32, (5, 5), kernel_regularizer=self.regularizer, input_shape=self.input_shape))
        shift_model.add(Activation(self.activation))
        shift_model.add(MaxPooling2D(pool_size=(2, 2)))

        shift_model.add(Conv2D(64, (3, 3), kernel_regularizer=self.regularizer))
        shift_model.add(Activation(self.activation))
        shift_model.add(MaxPooling2D(pool_size=(2, 2)))

        shift_model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer))
        shift_model.add(Activation(self.activation))
        shift_model.add(MaxPooling2D(pool_size=(2, 2)))

        zoom_in_model = Sequential()
        zoom_in_model.add(Conv2D(32, (5, 5), kernel_regularizer=self.regularizer, input_shape=self.input_shape))
        zoom_in_model.add(Activation(self.activation))
        zoom_in_model.add(MaxPooling2D(pool_size=(2, 2)))

        zoom_in_model.add(Conv2D(64, (3, 3), kernel_regularizer=self.regularizer))
        zoom_in_model.add(Activation(self.activation))
        zoom_in_model.add(MaxPooling2D(pool_size=(2, 2)))

        zoom_in_model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer))
        zoom_in_model.add(Activation(self.activation))
        zoom_in_model.add(MaxPooling2D(pool_size=(2, 2)))

        zoom_out_model = Sequential()
        zoom_out_model.add(Conv2D(32, (5, 5), kernel_regularizer=self.regularizer, input_shape=self.input_shape))
        zoom_out_model.add(Activation(self.activation))
        zoom_out_model.add(MaxPooling2D(pool_size=(2, 2)))

        zoom_out_model.add(Conv2D(64, (3, 3), kernel_regularizer=self.regularizer))
        zoom_out_model.add(Activation(self.activation))
        zoom_out_model.add(MaxPooling2D(pool_size=(2, 2)))

        zoom_out_model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer))
        zoom_out_model.add(Activation(self.activation))
        zoom_out_model.add(MaxPooling2D(pool_size=(2, 2)))

        rotation_model = Sequential()
        rotation_model.add(Conv2D(32, (5, 5), kernel_regularizer=self.regularizer, input_shape=self.input_shape))
        rotation_model.add(Activation(self.activation))
        rotation_model.add(MaxPooling2D(pool_size=(2, 2)))

        rotation_model.add(Conv2D(64, (3, 3), kernel_regularizer=self.regularizer))
        rotation_model.add(Activation(self.activation))
        rotation_model.add(MaxPooling2D(pool_size=(2, 2)))

        rotation_model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer))
        rotation_model.add(Activation(self.activation))
        rotation_model.add(MaxPooling2D(pool_size=(2, 2)))

        merged = Merge([shift_model, zoom_in_model, zoom_out_model, rotation_model])

        self.model.add(merged)

        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])


    def compile_model_9(self):
        self.model.add(
            Conv2D(32, (7, 7), kernel_regularizer=self.regularizer, activation=self.activation, input_shape=self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def compile_model_8(self):
        self.model.add(
            Conv2D(32, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation, input_shape=self.input_shape))
        self.model.add(Conv2D(32, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def compile_model_7(self):
        self.model.add(
            Conv2D(64, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation, input_shape=self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def compile_model_6(self):
        self.model.add(
            Conv2D(64, (5, 5), kernel_regularizer=self.regularizer, activation=self.activation, input_shape=self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation,))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation,))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation,))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(256, kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(Dense(128, kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def compile_model_5(self):
        self.model.add(
            Conv2D(32, (5, 5), kernel_regularizer=self.regularizer, activation=self.activation, input_shape=self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation,))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation,))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(256, kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def compile_model_4(self):
        self.model.add(
            Conv2D(64, (5, 5), kernel_regularizer=self.regularizer, activation=self.activation, input_shape=self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), kernel_regularizer=self.regularizer, activation=self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def compile_model_3_4(self):
        self.model.add(
            Conv2D(32, (3, 3), kernel_regularizer=self.regularizer, input_shape=self.input_shape))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), kernel_regularizer=self.regularizer))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(384))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])


    def compile_model_3_3(self):
        self.model.add(
            Conv2D(32, (3, 3), kernel_regularizer=self.regularizer, input_shape=self.input_shape))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), kernel_regularizer=self.regularizer))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])


    def compile_model_3_2(self):
        self.model.add(Cropping2D(cropping=35, input_shape=self.input_shape))
        self.model.add(
            Conv2D(32, (5, 5), kernel_regularizer=self.regularizer))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), kernel_regularizer=self.regularizer))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def compile_model_3_1(self):
        self.model.add(
            Conv2D(32, (5, 5), kernel_regularizer=self.regularizer, input_shape=self.input_shape, padding='same',
                   strides=1))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), kernel_regularizer=self.regularizer, padding='same'))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer, padding='same'))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def compile_model_3(self):
        self.model.add(
            Conv2D(32, (5, 5), kernel_regularizer=self.regularizer, input_shape=self.input_shape))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), kernel_regularizer=self.regularizer))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def compile_model_2(self):
        self.model.add(
            Conv2D(32, (3, 3), kernel_regularizer=self.regularizer, input_shape=self.input_shape))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), kernel_regularizer=self.regularizer))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), kernel_regularizer=self.regularizer))
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def compile_model(self):
        self.model.add(Conv2D(64, (3, 3), kernel_regularizer=self.regularizer, use_bias=False,
                              input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), use_bias=False, kernel_regularizer=self.regularizer))
        self.model.add(BatchNormalization())
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), use_bias=False, kernel_regularizer=self.regularizer))
        self.model.add(BatchNormalization())
        self.model.add(Activation(self.activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, kernel_regularizer=self.regularizer, activation='softmax'), )

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def train(self, datagen, X, y, validation_data=None, batch_size=64, epochs=3000, callbacks=None):
        # datagen = ImageDataGenerator(
        #     rotation_range=np.pi / 4,
        #     shear_range=np.pi / 5,
        #     width_shift_range=0.4,
        #     height_shift_range=0.4,
        #     horizontal_flip=True,
        #     vertical_flip=True,
        #     zoom_range=0.5,
        #     fill_mode='nearest')

        return self.model.fit_generator(datagen.flow(X, y, batch_size=batch_size), validation_data=validation_data,
                                        steps_per_epoch=len(X) / batch_size, epochs=epochs, callbacks=callbacks,
                                        use_multiprocessing=True, workers=3, max_queue_size=30
                                        )

    def train_generator(self, generator, steps_per_epoch, validation_data=None, epochs=3000, callbacks=None):
        # datagen = ImageDataGenerator(
        #     rotation_range=np.pi / 4,
        #     shear_range=np.pi / 5,
        #     width_shift_range=0.4,
        #     height_shift_range=0.4,
        #     horizontal_flip=True,
        #     vertical_flip=True,
        #     zoom_range=0.5,
        #     fill_mode='nearest')

        return self.model.fit_generator(generator, validation_data=validation_data,
                                        steps_per_epoch=steps_per_epoch, epochs=epochs,
                                        max_queue_size=100, callbacks=callbacks)

    def predict(self, X, batch_size=64):
        y = self.model.predict(X, batch_size=batch_size)
        return y

    def predict_classes(self, X, batch_size=64):
        y = self.model.predict_classes(X, batch_size=batch_size) + 1
        return y

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)
        return np.mean(y_pred == y)

