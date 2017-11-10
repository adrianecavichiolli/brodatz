import numpy as np
import cv2
import pandas as pd
import keras
import os


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


def save_model(path, filename, model):
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
