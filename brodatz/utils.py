import numpy as np
import datetime


def probas_to_classes(y_prob):
    return np.argmax(y_prob, axis=1)


def accuracy(y_test, y_predict):
    return np.mean(np.argmax(y_predict, axis=1) == np.argmax(y_test, axis=1))


def curr_date():
    now = datetime.datetime.now()
    return '{}.{}.{} {}:{}:{}'.format(now.day, now.month, now.year, now.hour, now.minute, now.second)