import os

# define the project directory
BASE_DIR = 'C:/Users/Tajr/Desktop/imagelab/'


# define the images path directories
IMAGES_PATH = os.path.join(BASE_DIR, 'datasets/raw/Catdog/')


# data split variables
NUM_CLASSES = 2
NUM_VAL_IMAGES = 3125 * NUM_CLASSES
NUM_TEST_IMAGES = 3125 * NUM_CLASSES

# define the path to the output training, validation and testing HDF5 files
TRAIN_HDF5 = os.path.join(BASE_DIR, 'datasets/hdf5/Catdog/train.hdf5')
VAL_HDF5 = os.path.join(BASE_DIR, 'datasets/hdf5/Catdog/val.hdf5')
TEST_HDF5 = os.path.join(BASE_DIR, 'datasets/hdf5/Catdog/test.hdf5')

# extracted features path
CAT_DOG_FEATURES = os.path.join(BASE_DIR, 'datasets/hdf5/CatDog/features.hdf5')

# path to the output model file
MODEL_PATH = os.path.join(BASE_DIR, 'src/Cat_Dog/serialized/model/alexnet_cat_dog.model')
FFN_MODEL_PATH = os.path.join(BASE_DIR, 'src/Cat_Dog/serialized/model/ffn_feature_extraction.model')

# path to the dataset mean
DATASET_MEAN = os.path.join(BASE_DIR, 'src/Cat_Dog/serialized/json/cat_dog_mean.json')

# classification report
OUTPUT_PATH = os.path.join(BASE_DIR, 'src/Cat_Dog/serialized/others/')

# plots
PLOT_PATH = os.path.join(BASE_DIR, 'src/Cat_Dog/serialized/plots/')