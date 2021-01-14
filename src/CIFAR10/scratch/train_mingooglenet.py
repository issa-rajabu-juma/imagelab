from src.CIFAR10.networks.minigooglenet import MiniGoogleNet
from keras.datasets import cifar10
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from callbacks.trainingmonitor import TrainMonitor
import os
from keras.optimizers import SGD

# working directories
BASE_DIR = 'C:/Users/Tajr/Desktop/imagelab/src/CIFAR10/'

# learning rate Scheduling
NUM_EPOCHS = 70
INIT_LR = 5e-3


def poly_decay(epoch):
    # initialize the max epoch. base lr and the power of polynomial
    maxEpoch = NUM_EPOCHS
    baseLr = INIT_LR
    power = 0.1

    alpha = baseLr * (1 - (epoch / float(maxEpoch))) ** power

    # return the new learning rate
    return alpha


# working with dataset
((trainX, trainY), (testX, testY)) = cifar10.load_data()

# convert pixels intensity to float
trainX = trainX.astype('float')
testX = testX.astype('float')

# apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert labels to vector from integer
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)
class_names = lb.classes_

# define data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')

# construct the set of callbacks
path = os.path.join(BASE_DIR, 'serialized/plots/{}.png'.format(os.getpid()))
jsonPath = os.path.join(BASE_DIR, 'serialized/json/{}.json'.format(os.getpid()))

callbacks = [TrainMonitor(figure_path=path, json_path=jsonPath), LearningRateScheduler(poly_decay)]

# compile a model
print('[INFO] compiling...')
opt = SGD(lr=INIT_LR, momentum=0.9)
model = MiniGoogleNet.build(32, 32, 3, 10)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

# train a model
print('[INFO] training...')
model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
                    validation_data=(testX, testY),
                    steps_per_epoch=len(trainX) // 64,
                    epochs=NUM_EPOCHS,
                    callbacks=callbacks)

# save the network to the disk
print('[INFO] model serialization...')
model.save(os.path.join(BASE_DIR, 'serialized/model/minigooglenet.hdf5'), overwrite=True)