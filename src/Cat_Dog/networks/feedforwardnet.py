from keras import models
from keras import layers


# feed forward net
class FeedForwardNet:
    @staticmethod
    def build(data, classes):
        # config input shape
        input_shape = (data.shape[1],)

        # construct a model
        model = models.Sequential()

        # add layers
        model.add(layers.Dense(256, activation='relu', input_shape=input_shape))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(classes, activation='softmax'))

        # return constructed model
        return model