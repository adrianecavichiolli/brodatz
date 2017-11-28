import numpy as np
import datetime
import os
import json

import pandas as pd

from keras import regularizers, optimizers, utils, applications

from brodatz.data_utils import save_history


def probas_to_classes(y_prob):
    return np.argmax(y_prob, axis=1)


def accuracy(y_test, y_predict):
    return np.mean(np.argmax(y_predict, axis=1) == np.argmax(y_test, axis=1))


def curr_date():
    now = datetime.datetime.now()
    return '{}.{}.{} {}:{}:{}'.format(now.day, now.month, now.year, now.hour, now.minute, now.second)


def save_predict_classes(y_pred):
    path = '/mnt/82db778e-0496-450c-9b25-d1e50a90e476/data/data4stas/brodaz/'
    current_dir = os.getcwd()
    os.chdir(path)

    input_filename = 'brodatz_dataset_test_submit.csv'
    output_filename = 'brodatz_dataset_test_submit'

    df = pd.read_csv(input_filename)
    df[df.columns[1]] = y_pred
    df.to_csv(output_filename, index=False)

    os.chdir(current_dir)

def save_predict_results(net, X):
    path = '/mnt/82db778e-0496-450c-9b25-d1e50a90e476/data/data4stas/brodaz/'
    current_dir = os.getcwd()
    os.chdir(path)

    input_filename = 'brodatz_dataset_test_submit.csv'
    output_filename = 'brodatz_dataset_test_submit'

    df = pd.read_csv(input_filename)
    y_pred = net.predict_classes(X)
    df[df.columns[1]] = y_pred
    df.to_csv(output_filename, index=False)

    os.chdir(current_dir)


def base_try_args(X_train, y_train, num_classes, test_list, net, optimizer, history, train_duration, rotation_range,
                  shear_range, shift_range, horizontal_flip, vertical_flip, zoom_range, fill_mode, batch_size, epochs,
                  learning_rate, regularization_strength, save_model, save_results):
    model_name = curr_date()  # name of the directory for model
    root_directory = '/home/stanislau/repository/machine-learning/brodatz/results/'
    directory_name = root_directory + curr_date()
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)  # create directory

    save_history(directory_name + '/loss.png', directory_name + '/accuracy.png',
                 history)  # save plots of loss and accuracy

    model_json = net.model.to_json()
    with open(directory_name + "/model.json", "w") as json_file:
        json_file.write(model_json)  # save model to file

    utils.plot_model(net.model, to_file=directory_name + "/model.png",
                     show_shapes=True)  # save model png overview to file

    number_of_params = net.model.count_params()  # number of model parameters
    optimizer_json = json.dumps(optimizers.serialize(optimizer))
    with open(directory_name + "/optimizer.json", "w") as json_file:
        json_file.write(optimizer_json)  # save optimizer to file

    if save_model:
        net.model.save_weights(directory_name + '/model.h5')

    models_battles_filename = root_directory + 'models_battles.csv'
    header = ['Model name', 'X_train', 'X_test', 'X_test1', 'X_test2', 'X_test3', 'X_test4',
              'Epochs', 'Number of parameters', 'Batches', 'Learning rate', 'Train duration',
              'Rotation range', 'Shear range', 'Shift range', 'Hor flip', 'Ver flip', 'Zoom range', 'Fill mode',
              'Number of classes',
              'Regularization Strength']
    accuracies = [net.accuracy(X_test, y_test) for X_test, y_test, _ in [(X_train, y_train, num_classes)] + test_list]
    data = [model_name]
    data.extend(accuracies)
    data.extend([epochs, number_of_params, batch_size, learning_rate, train_duration])
    data.extend([rotation_range, shear_range, shift_range, horizontal_flip, vertical_flip, zoom_range, fill_mode])
    data.append(num_classes)
    data.append(regularization_strength)
    if not os.path.isfile(models_battles_filename):
        df = pd.DataFrame([data], columns=header)
        df.to_csv(models_battles_filename)
    else:
        with open(models_battles_filename, 'a') as f:
            df = pd.DataFrame([data])
            df.to_csv(f, header=False)

    if save_results:
        save_predict_results(net, test_list[0][0])

def try_args_fit(X_train, y_train, num_classes, X_val, y_val, test_list, net, optimizer, callbacks, rotation_range,
             shear_range, shift_range, horizontal_flip, vertical_flip, zoom_range, fill_mode, batch_size, epochs, learning_rate,
             regularization_strength, save_model=False, save_results=False):
    train_start = datetime.datetime.now()
    history = net.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                              callbacks=callbacks)
    train_finish = datetime.datetime.now()
    train_duration = (train_finish - train_start).seconds

    base_try_args(X_train, y_train, num_classes, test_list, net, optimizer, history, train_duration, rotation_range,
                  shear_range, shift_range, horizontal_flip, vertical_flip, zoom_range, fill_mode, batch_size, epochs,
                  learning_rate, regularization_strength, save_model, save_results)


def try_args(X_train, y_train, num_classes, X_val, y_val, test_list, net, optimizer, datagen, callbacks, rotation_range,
             shear_range, shift_range, horizontal_flip, vertical_flip, zoom_range, fill_mode, batch_size, epochs, learning_rate,
             regularization_strength, save_model=False, save_results=False):
    train_start = datetime.datetime.now()
    history = net.train(datagen, X_train, y_train, (X_val, y_val), epochs=epochs, batch_size=batch_size,
                        callbacks=callbacks)
    train_finish = datetime.datetime.now()
    train_duration = (train_finish - train_start).seconds

    base_try_args(X_train, y_train, num_classes, test_list, net, optimizer, history, train_duration, rotation_range,
                  shear_range, shift_range, horizontal_flip, vertical_flip, zoom_range, fill_mode, batch_size, epochs,
                  learning_rate, regularization_strength, save_model, save_results)


def try_args_generator(X_train, y_train, num_classes, X_val, y_val, test_list, net, optimizer, datagen, steps_per_epoch,
                       callbacks, rotation_range, shear_range, shift_range, horizontal_flip, vertical_flip,
                       zoom_range, fill_mode, batch_size, epochs, learning_rate, regularization_strength,
                       save_model=False, save_results=False):
    train_start = datetime.datetime.now()
    history = net.train_generator(datagen, validation_data=(X_val, y_val),
                                  steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)
    train_finish = datetime.datetime.now()
    train_duration = (train_finish - train_start).seconds

    base_try_args(X_train, y_train, num_classes, test_list, net, optimizer, history, train_duration, rotation_range,
                  shear_range, shift_range, horizontal_flip, vertical_flip, zoom_range, fill_mode, batch_size, epochs,
                  learning_rate, regularization_strength, save_model, save_results)


def get_bottlebeck_features(train_generator, val_generator, batch_size, nb_train_samples, nb_validation_samples):
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    bottleneck_features_train = model.predict_generator(train_generator, nb_train_samples // batch_size)
    bottleneck_features_validation = model.predict_generator(val_generator, nb_validation_samples // batch_size)

    return bottleneck_features_train, bottleneck_features_validation
