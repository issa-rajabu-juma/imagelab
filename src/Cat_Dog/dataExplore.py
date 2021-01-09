import h5py as h5
from config import cat_dog_config as config


train_data = h5.File(config.TRAIN_HDF5, 'r')
print(train_data['Label Names'])
