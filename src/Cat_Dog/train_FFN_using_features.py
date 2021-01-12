from src.Cat_Dog.feedforwardnet.feedforwardnet import FeedForwardNet
import h5py as h5
from config import cat_dog_config as config
from keras.optimizers import SGD
import os
from callbacks.trainingmonitor import TrainMonitor

# load dataset
dataset = h5.File(config.CAT_DOG_FEATURES, 'r')

# initialize model
model = FeedForwardNet.build(dataset['Features'], config.NUM_CLASSES)

# data splitting
index = 0
train_index = int(dataset['Features'].shape[0] * 0.5)
val_index = int(dataset['Features'].shape[0] * 0.75)

# model compilation
learning_rate = 0.001
opt = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

# apply Train monitor callback
path = os.path.sep.join([config.PLOT_PATH, '{}.png'.format(os.getpid())])
callbacks = [TrainMonitor(figure_path=path)]

# training
print('[INFO] training')
history = model.fit(dataset['Features'][index:index + train_index],
                    dataset['Labels'][index:index + train_index],
                    epochs=100,
                    batch_size=50,
                    callbacks=callbacks,
                    validation_data=(
                        dataset['Features'][index + train_index:val_index],
                        dataset['Labels'][index + train_index:val_index]
                    ))

# serialize a model
print('[INFO] model serialization')
model.save(config.FFN_MODEL_PATH, overwrite=True)

print('[INFO] finished')