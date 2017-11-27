import numpy as np
import cv2
import pandas as pd
import keras
import os
from matplotlib import pyplot as plt


def process_file(path):
    df = pd.read_csv(path)

    x = []
    y = []

    for index, row in df.iterrows():
        x.append(cv2.imread(row[0], cv2.IMREAD_GRAYSCALE))
        y.append(float(row[1]))

    x = np.array(x)
    y = np.array(y)

    num_classes = len(np.unique(y))

    x = x.astype('float32') / 255
    x -= np.mean(x, axis=1, keepdims=True)
    x /= (np.std(x, axis=1, keepdims=True) + 1e-7)
    x = x.reshape(x.shape + (-1,))

    y -= 1
    y = keras.utils.to_categorical(y, num_classes=num_classes)

    return x, y, num_classes


def process_file_some_classes(path, classes):
    df = pd.read_csv(path)

    x = []
    y = []

    for index, row in df.iterrows():
        x.append(cv2.imread(row['FilePath'], cv2.IMREAD_GRAYSCALE))
        y.append(float(row['ClassId']))

    x = np.array(x)
    y = np.array(y) - 1

    num_classes = len(classes)

    x = x.astype('float32') / 255
    x -= np.mean(x, axis=1, keepdims=True)
    x /= (np.std(x, axis=1, keepdims=True) + 1e-7)
    x = x.reshape(x.shape + (-1,))

    x = x[np.in1d(y, classes)]
    y = y[np.in1d(y, classes)]

    y -= np.min(y)

    y = keras.utils.to_categorical(y, num_classes=num_classes)

    return x, y, num_classes


def read_some_classes(path, classes):
    df = pd.read_csv(path)

    x = np.array(df[df.columns[0]])
    y = np.array(df[df.columns[1]])
    y -= np.min(y)

    mask = np.in1d(y, classes)
    x = np.array([cv2.imread(filename, cv2.IMREAD_GRAYSCALE) for filename in x[mask]])
    y = y[mask]

    num_classes = len(np.unique(y))
    assert num_classes == len(classes), 'Error during file reading'

    return x, y, num_classes


def read_train_test_sets_2(train_filename, test_filenames, classes_range):
    X_train, y_train, num_classes = read_some_classes(train_filename, classes_range)

    X_train = X_train.astype('float') / 255
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train = (X_train - X_mean) / X_std

    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    X_train = X_train[..., np.newaxis]

    test_list = [list(read_some_classes(test_filename, classes_range)) for test_filename in test_filenames]
    for test in test_list:
        test[0] = test[0].astype('float') / 255
        test[0] = (test[0] - X_mean) / X_std
        test[0] = test[0][..., np.newaxis]
        test[1] = keras.utils.to_categorical(test[1], num_classes=num_classes)

    num_test = round(len(test_list[0][0]) * 0.8)
    X_val = test_list[0][0][num_test:]
    y_val = test_list[0][1][num_test:]

    return X_train, y_train, num_classes, X_val, y_val, test_list



def read_train_test_sets(train_filename, test_filenames, classes_range):
    X_train, y_train, num_classes = read_some_classes(train_filename, classes_range)

    X_train = X_train.astype('float') / 255
    X_mean = np.mean(X_train, axis=0)
    X_train -= X_mean

    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    X_train = X_train[..., np.newaxis]

    test_list = [list(read_some_classes(test_filename, classes_range)) for test_filename in test_filenames]
    for test in test_list:
        test[0] = test[0].astype('float') / 255
        test[0] -= X_mean
        test[0] = test[0][..., np.newaxis]
        test[1] = keras.utils.to_categorical(test[1], num_classes=num_classes)

    num_test = round(len(test_list[0][0]) * 0.8)
    X_val = test_list[0][0][num_test:]
    y_val = test_list[0][1][num_test:]

    return X_train, y_train, num_classes, X_val, y_val, test_list


def save_model(model, path, filename):
    path = '/media/stanislau/82db778e-0496-450c-9b25-d1e50a90e476/data/data4stas/brodaz'
    current_dir = os.getcwd()
    os.chdir(path)

    input_filename = 'brodatz_dataset_test_submit.csv'
    output_filename = 'brodatz_dataset_test_submit_0'

    x_test, y_test, num_classes = process_file(input_filename)
    df = pd.read_csv(input_filename)
    y_pred = model.predict_classes(x_test) + 1
    df[df.columns[1]] = y_pred
    df.to_csv(output_filename, index=False)

    os.chdir(current_dir)


def save_history(loss_filename, accuracy_filename, history):
    acc = history.history['acc']
    val_acc = history.history.get('val_acc')
    loss = history.history['loss']
    val_loss = history.history.get('val_loss')

    epochs = range(1, len(loss) + 1)

    plt.clf()
    plt.plot(epochs, loss)
    if val_loss is not None:
        plt.plot(epochs, val_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(loss_filename)
    plt.clf()

    plt.plot(epochs, acc)
    if val_acc is not None:
        plt.plot(epochs, val_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(accuracy_filename)
