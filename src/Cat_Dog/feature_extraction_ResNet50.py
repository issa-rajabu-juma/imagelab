from loader import input_loader
from config import cat_dog_config as config
from sklearn.preprocessing import LabelEncoder
from keras.applications import ResNet50
from keras.layers import Input
from hdf5_datasets.dataset import HDF5DatasetWriter
import progressbar
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils


# get images paths and labels
images_path, labels = input_loader.load_path_and_labels(config.IMAGES_PATH)

# encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)
class_names = le.classes_

# load ResNet50
model = ResNet50(weights='imagenet', include_top=False)

# instantiate HDF5 dataset writer and store the class labels
writer = HDF5DatasetWriter((len(images_path), 7 * 7 * 2048), outputFile=config.CAT_DOG_FEATURES)
writer.storeClassLabels(class_names)

# start a progress bar
widget = ['Features extracting: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
progress = progressbar.ProgressBar(maxval=len(images_path), widgets=widget).start()

# loop over the path in batches
batch_size = 50
for i in np.arange(0, len(images_path), batch_size):
    # create batch labels and paths to load their images
    batchPaths = images_path[i: i + batch_size]
    batchLabels = labels[i: i + batch_size]
    batchImages = []

    # loop over the images batch path
    for j, path in enumerate(batchPaths):
        # load the image and convert it into an array
        image = load_img(path, target_size=(224, 224))
        image = img_to_array(image)

        # expand image dimension and preprocess the image
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        batchImages.append(image)

    # pass the image to the network
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=batch_size)

    # flatten the features and store to the dataset
    features = np.reshape(features, (features.shape[0], 7 * 7 * 2048))
    writer.add(features, batchLabels)

    # update progress bar
    progress.update(i)

# close dataset
progress.finish()
writer.close()

print('[INFO] finished')