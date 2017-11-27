from matplotlib import pyplot as plt
import numpy as np
import itertools


def plot_image(X):
    low, high = np.min(X), np.max(X)
    plt.imshow(255.0 * (X - low) / (high - low), cmap=plt.cm.gray)
    plt.show()


def plot(x):
    plt.imshow(x.reshape(x.shape[:-1]), cmap=plt.cm.gray)
    plt.show()


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history.get('val_acc')
    loss = history.history['loss']
    val_loss = history.history.get('val_loss')

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss)
    if val_loss is not None:
        plt.plot(epochs, val_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(epochs, acc)
    if val_acc is not None:
        plt.plot(epochs, val_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_inputs(image_list):
    cols = 5
    rows = 5

    fig, ax = plt.subplots(rows, cols, figsize=(15, 15))
    for i in range(rows):
        for j in range(cols):
            x = image_list[j][i]
            ax[i][j].imshow(x.reshape(x.shape[:-1]), cmap=plt.cm.gray)
    # plt.setp(ax, xticks=[], yticks=[])
    plt.setp(ax, xticks=[])
    plt.show()