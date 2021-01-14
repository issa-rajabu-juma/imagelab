from keras.models import Model
from keras.layers import Input
from keras import backend as K
from keras import layers


class MiniGoogleNet:
    @staticmethod
    def conv_module(input_layer, filters, kernel_size, strides, chan_dim, padding='same'):
        # define a CONV => BN => RELU pattern
        input_layer = layers.Conv2D(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding)(input_layer)
        input_layer = layers.BatchNormalization(axis=chan_dim)(input_layer)
        input_layer = layers.Activation('relu')(input_layer)

        # return the block
        return input_layer

    @staticmethod
    def inception_module(input_layer, num11filters, num33filters, chan_dim):
        # define two CONV modules, then concatenate across the channel dimension
        conv11 = MiniGoogleNet.conv_module(input_layer=input_layer,
                                           filters=num11filters,
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           chan_dim=chan_dim)
        conv33 = MiniGoogleNet.conv_module(input_layer=input_layer,
                                           filters=num33filters,
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           chan_dim=chan_dim)
        input_layer = layers.concatenate([conv11, conv33], axis=chan_dim)

        # return the block
        return input_layer

    # this method is responsible for reducing the spatial dimensions of an input volume
    @staticmethod
    def downsample_module(input_layer, filters, chan_dim):
        # define the CONV module and POOL, then concatenate across the channel dimension
        conv33 = MiniGoogleNet.conv_module(input_layer=input_layer,
                                           filters=filters,
                                           kernel_size=(3, 3),
                                           strides=(2, 2),
                                           chan_dim=chan_dim,
                                           padding='valid')
        pool = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(input_layer)
        input_layer = layers.concatenate([conv33, pool], axis=chan_dim)

        # return the block
        return input_layer

    # build a minigoolenet
    @staticmethod
    def build(width, height, depth, classes):
        # initialize input shape and channel dimension
        input_shape = (width, height, depth)
        chan_dim = -1

        # configure input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, width, height)
            chan_dim = 1

        # define the model input and first CONV module
        inputs = Input(shape=input_shape)
        layer = MiniGoogleNet.conv_module(input_layer=inputs, filters=96, kernel_size=(3, 3), strides=(1, 1),
                                          chan_dim=chan_dim)

        # two inception modules followed by downsample module
        layer = MiniGoogleNet.inception_module(input_layer=layer, num11filters=32, num33filters=32, chan_dim=chan_dim)
        layer = MiniGoogleNet.inception_module(input_layer=layer, num11filters=32, num33filters=48, chan_dim=chan_dim)
        layer = MiniGoogleNet.downsample_module(input_layer=layer, filters=80, chan_dim=chan_dim)

        # four inception modules followed by downsample module
        layer = MiniGoogleNet.inception_module(input_layer=layer, num11filters=112, num33filters=48, chan_dim=chan_dim)
        layer = MiniGoogleNet.inception_module(input_layer=layer, num11filters=96, num33filters=64, chan_dim=chan_dim)
        layer = MiniGoogleNet.inception_module(input_layer=layer, num11filters=80, num33filters=80, chan_dim=chan_dim)
        layer = MiniGoogleNet.inception_module(input_layer=layer, num11filters=48, num33filters=96, chan_dim=chan_dim)
        layer = MiniGoogleNet.downsample_module(input_layer=layer, filters=96, chan_dim=chan_dim)

        # two inception modules followed by average pooling and full connected softmax classifier
        layer = MiniGoogleNet.inception_module(input_layer=layer, num11filters=176, num33filters=160, chan_dim=chan_dim)
        layer = MiniGoogleNet.inception_module(input_layer=layer, num11filters=176, num33filters=160, chan_dim=chan_dim)
        layer = layers.AveragePooling2D(pool_size=(7, 7))(layer)
        layer = layers.Dropout(0.5)(layer)
        layer = layers.Flatten()(layer)
        layer = layers.Dense(classes)(layer)
        layer = layers.Activation('softmax')(layer)

        # create the model
        model = Model(inputs, layer, name='googlenet')

        # return the model
        return model