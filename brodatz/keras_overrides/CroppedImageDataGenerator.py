from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator


class CroppedNumpyArrayIterator(NumpyArrayIterator):
    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 cropped_x=0,
                 cropped_y=0):
        super(CroppedNumpyArrayIterator, self).__init__(
            x=x, y=y, image_data_generator=image_data_generator,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            data_format=data_format,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

        self.cropped_x = cropped_x
        self.cropped_y = cropped_y

    def next(self):
        x, y = super(CroppedNumpyArrayIterator, self).next()
        return x[:, self.cropped_x: x.shape[1] - self.cropped_x, self.cropped_y: x.shape[2] - self.cropped_y, :], y


class CroppedImageDataGenerator(ImageDataGenerator):
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 **kwargs):
        super(CroppedImageDataGenerator, self).__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            zca_epsilon=zca_epsilon,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format,
            **kwargs
            )

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        return CroppedNumpyArrayIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)
