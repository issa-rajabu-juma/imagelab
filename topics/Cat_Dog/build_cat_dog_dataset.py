import cv2
import json
import progressbar
import numpy as np
from config import cat_dog_config as config
from sklearn.preprocessing import LabelEncoder
from preprocessing.preprocessor import AspectRatioAwarePreprocessor
from sklearn.model_selection import train_test_split
from hdf5_datasets.dataset import HDF5DatasetWriter
from loader.input_loader import load_path_and_labels

# Load images path and labels
(images_path, labels) = load_path_and_labels(config.IMAGES_PATH)


# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)
label_names = le.classes_

# data splitting
(sample_path, test_path, sample_labels, test_labels) = train_test_split(
    images_path,
    labels,
    test_size=config.NUM_TEST_IMAGES,
    random_state=42,
    stratify=labels)

(train_path, validation_path, train_labels, validation_labels) = train_test_split(
    sample_path,
    sample_labels,
    test_size=config.NUM_VAL_IMAGES,
    random_state=42,
    stratify=sample_labels
)

# construct a dataset list
datasets = [
    ('train', train_path, train_labels, config.TRAIN_HDF5),
    ('validation', validation_path, validation_labels, config.VAL_HDF5),
    ('test', test_path, test_labels, config.TEST_HDF5)
]

# initialize the preprocessor and the lists of RGB channel average
preprocessor = AspectRatioAwarePreprocessor(128, 128)
(R, G, B) = ([], [], [])


widget = ['Building Dataset: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
# build dataset by looping over the tuples
for(dType, paths, labels, outputPath) in datasets:
    # create HDF5 writer
    print('[INFO] building {}...'.format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 128, 128, 3), outputPath)

    # initialize the progress bar
    progress = progressbar.ProgressBar(maxval=len(paths), widgets=widget).start()

    # loop over the images path
    for (i, (path, label)) in enumerate(zip(paths, labels)):

        image = cv2.imread(path)
        image = preprocessor.preprocess(image)

        # if we are building the training dataset, then compute the mean of each
        # channel in the image, then update the respective lists
        if dType == 'train':
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            B.append(b)
            G.append(g)

        # add the image and label to the HDF5 dataset
        writer.add([image], [label])
        progress.update(i)

    # close the hdf5 writer
    progress.finish()
    writer.close()

# construct a dictionary of averages, then serialize the means to a JSON file
print('[INFO] serialize means...')
D = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
f = open(config.DATASET_MEAN, 'w')
f.write(json.dumps(D))
f.close()