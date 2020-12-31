from hdf5writer.HDF5DatasetWriter import HDF5DatasetWriter
from preprocessing.Preprocessing import load_path_and_labels
from keras.applications import VGG16
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import progressbar
import numpy as np
import os


# Specify working directories
base_dir = 'C:/Users/Tajr/Desktop/imagelab/'
malaria_dataset = os.path.join(base_dir, 'datasets/raw/Malaria/')
malaria_hdf5 = os.path.join(base_dir, 'datasets/hdf5/malaria.hdf5')


# Get images path
(images_path, labels) = load_path_and_labels(malaria_dataset)

# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(labels)
class_names = le.classes_

# Load the network
model = VGG16(weights='imagenet', include_top=False)

# Instantiate the dataset writer and store the class/label names
dataset = HDF5DatasetWriter((len(labels), 512 * 7 * 7), malaria_hdf5)
dataset.storeClassLabels(class_names)

print('[INFO] extracting...')
# Initialize a progress bar for feature extraction
extract_widget = ['Extract Features: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
pgr = progressbar.ProgressBar(maxval=len(images_path), widgets=extract_widget).start()

# loop over the image paths to extract features of images in patches
batch_size = 10
for i in np.arange(0, len(images_path), batch_size):
    batchPath = images_path[i:i + batch_size]
    batchLabels = labels[i:i + batch_size]
    batchImages = []

    for (j, path) in enumerate(batchPath):
        # load the input images using keras utility function
        image = load_img(path, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess the image by expanding the dimensions and subtracting
        # the mean RGB pixel intensity from the imagenet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # add images to the batch
        batchImages.append(image)

    # pass the images through the network and use the outputs as extracted features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=batch_size)

    # Flatten the feature to be represented in vector space
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # add features and labels to our HDF5 dataset
    dataset.add(features, batchLabels)

    # update the progressbar
    pgr.update(i)

# close the dataset
dataset.close()

# finish the progress bar
pgr.finish()

print('[INFO] extracting finished.')