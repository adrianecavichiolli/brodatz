from keras.preprocessing.image import ImageDataGenerator
import numpy as np

shift_generator = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.5,
    height_shift_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0,
    fill_mode='reflect')

zoom_in_generator = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=[0.55, 0.65],
    fill_mode='reflect')

zoom_out_generator = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=[1.45, 1.55],
    fill_mode='reflect')

rotation_generator = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0,
    fill_mode='reflect')


def brodatz_generator(X, y, batch_size=64):
    generator_batch_size = batch_size // 4
    shift = shift_generator.flow(X, y, batch_size=generator_batch_size)
    zoom_in = zoom_in_generator.flow(X, y, batch_size=generator_batch_size)
    zoom_out = zoom_out_generator.flow(X, y, batch_size=generator_batch_size)
    rotation = rotation_generator.flow(X, y, batch_size=generator_batch_size)

    additional_zoom_in = False
    i = 0
    while True:
        # X_shift, y_shift = shift.next() if additional_zoom_in else zoom_in.next()  # Alternate between shift and zoom in
        # additional_zoom_in = not additional_zoom_in

        X_shift, y_shift = shift.next()
        X_zoom_in, y_zoom_in = zoom_in.next()
        X_zoom_out, y_zoom_out = zoom_out.next()
        X_rotation, y_rotation = rotation.next()

        # yield [X_shift, X_zoom_in, X_zoom_out, X_rotation], y_shift

        X_b = np.concatenate((X_shift, X_zoom_in, X_zoom_out, X_rotation))
        y_b = np.concatenate((y_shift, y_zoom_in, y_zoom_out, y_rotation))

        mask = np.random.permutation(len(X_b))
        X_b = X_b[mask]
        y_b = y_b[mask]

        yield X_b, y_b


def brodatz_generator_2(X, y, batch_size=64):
    generator_batch_size = batch_size
    shift = shift_generator.flow(X, y, batch_size=generator_batch_size)
    zoom_in = zoom_in_generator.flow(X, y, batch_size=generator_batch_size)
    zoom_out = zoom_out_generator.flow(X, y, batch_size=generator_batch_size)
    rotation = rotation_generator.flow(X, y, batch_size=generator_batch_size)

    iterators = [shift, zoom_in, zoom_out, rotation]
    iterator_index = 0

    while True:
        batch = iterators[iterator_index].next()
        iterator_index = (iterator_index + 1) % len(iterators)
        yield batch


def brodatz_generator_3(X, y, batch_size=64):
    generator_batch_size = batch_size
    shift = shift_generator.flow(X, y, batch_size=generator_batch_size)
    zoom_in = zoom_in_generator.flow(X, y, batch_size=generator_batch_size)
    zoom_out = zoom_out_generator.flow(X, y, batch_size=generator_batch_size)
    rotation = rotation_generator.flow(X, y, batch_size=generator_batch_size)

    iterators = [shift, zoom_in, zoom_out, rotation]
    iterator_index = 0

    i = 0
    while True:
        if i < 200:
            batch = zoom_in.next()
            i += 1
        else:
            batch = iterators[iterator_index].next()
            iterator_index = (iterator_index + 1) % len(iterators)
        yield batch
