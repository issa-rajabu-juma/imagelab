from keras import models
from keras import layers
from keras.regularizers import l2
from keras import backend as K


class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        # initialize the model along with the input shape to be channels last
        model = models.Sequential()
        input_shape = (width, height, depth)
        chan_dims = -1

        # configure the input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, width, height)
            chan_dims = 1

        # add layers to your model
        # Block #1: first CONV => RELU => POOL layer set
        model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), padding='same', input_shape=input_shape,
                                kernel_regularizer=l2(reg)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization(axis=chan_dims))
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(layers.Dropout(0.25))

        # Block #2: second CONV => RELU => POOL layer set
        model.add(layers.Conv2D(256, (5, 5), padding='same', kernel_regularizer=l2(reg)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization(axis=chan_dims))
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(layers.Dropout(0.25))

        # Block #3: CONV => RELU => CONV => RELU => CONV => RELU
        model.add(layers.Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(reg)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization(axis=chan_dims))
        model.add(layers.Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(reg)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization(axis=chan_dims))
        model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(reg)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization(axis=chan_dims))
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(layers.Dropout(0.25))

        # Block #4: first set of FC => RELU layers
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, kernel_regularizer=l2(reg)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization(axis=chan_dims))
        model.add(layers.Dropout(0.5))

        # Block #5: second set of FC => RELU layers
        model.add(layers.Dense(4096, kernel_regularizer=l2(reg)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization(axis=chan_dims))
        model.add(layers.Dropout(0.5))

        # softmax classifier
        model.add(layers.Dense(classes, kernel_regularizer=l2(reg)))
        model.add(layers.Activation('softmax'))

        # return the constructed network architecture
        return model