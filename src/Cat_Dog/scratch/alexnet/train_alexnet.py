# set the matplotlib backend so figures can be saved inthe background
import matplotlib
matplotlib.use('Agg')
from keras.preprocessing.image import ImageDataGenerator
from config import cat_dog_config as config
from preprocessing import preprocessor
from hdf5_datasets.dataset import HDF5DatasetGenerator
from keras.optimizers import Adam
import json
from src.Cat_Dog.networks.alexnet import AlexNet
import os
from callbacks.trainingmonitor import TrainMonitor

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.15,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the preprocessor
sp = preprocessor.SimplePreprocessor(171, 171)
pp = preprocessor.PatchPreprocessor(171, 171)
mp = preprocessor.MeanPreprocessor(means['R'], means['G'], means['B'])
iap = preprocessor.ImageToArrayPreprocessor()

# initialize the training and validation dataset generator
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, batch_size=128, aug=aug, preprocessors=[pp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, batch_size=128, preprocessors=[sp, mp, iap], classes=2)

# initialize the optimizer, model and then compile the model
print('[INFO] compiling model')
opt = Adam(lr=1e-3)
model = AlexNet.build(width=171, height=171, depth=3, classes=2, reg=0.0002)
model.summary()
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

# construct the set of callbacks
path = os.path.sep.join([config.PLOT_PATH, '{}.png'.format(os.getpid())])
callbacks = [TrainMonitor(figure_path=path)]

# train the network
model.fit_generator(trainGen.generator(),
                    steps_per_epoch=trainGen.num_images // 128,
                    validation_data=valGen.generator(),
                    validation_steps=valGen.num_images // 128,
                    epochs=100,
                    max_queue_size=128 * 2,
                    callbacks=callbacks,
                    verbose=1
                    )

# save the model to file
print('[INFO] serializing model...')
model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 dataset
trainGen.close()
valGen.close()