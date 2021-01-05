from keras.utils import np_utils
from numpy import unicode
import numpy as np
import h5py as h5
import os


# dataset writer
class HDF5DatasetWriter:
    # constructor
    def __init__(self, dimensions, outputFile, bufSize=1000, dataKey='Features'):
        # check to see if the outputFile already exists, and if so raise an exception
        if os.path.exists(outputFile):
            raise ValueError('The supplied output file already exists and can not be overwritten.'
                             ' Manually delete the file before continuing', outputFile)

        # open hdf5 database and create two datasets
        self.h5_database = h5.File(outputFile, 'w')
        self.data = self.h5_database.create_dataset(dataKey, dimensions, dtype='float')
        self.labels = self.h5_database.create_dataset('Labels', (dimensions[0],), dtype='int')

        # initialize buffer and its size
        self.bufSize = bufSize
        self.buffer = {'data': [], 'labels': []}
        self.index = 0

    # add data to buffer
    def add(self, data, labels):
        # add data and labels to buffer
        self.buffer['data'].extend(data)
        self.buffer['labels'].extend(labels)

        # check to see if buffer needs to be flushed to disk
        if len(self.buffer['data']) >= self.bufSize:
            self.flush()

    # flush buffer to disk
    def flush(self):
        count = self.index + len(self.buffer['data'])
        self.data[self.index:count] = self.buffer['data']
        self.labels[self.index:count] = self.buffer['labels']
        self.index = count
        self.buffer = {'data': [], 'labels': []}

    # store class label names
    def storeClassLabels(self, classLabels):
        # create a dataset to store class label names
        datatype = h5.special_dtype(vlen=unicode)
        label_names = self.h5_database.create_dataset('Label Names', (len(classLabels),), dtype=datatype)
        label_names[:] = classLabels

    # close the database
    def close(self):
        # check to see if there are any other entry in the buffer that need to be flushed to disk
        if len(self.buffer['data']) > 0:
            self.flush()

        # close the dataset
        self.h5_database.close()


# dataset generator
class HDF5DatasetGenerator:
    # constructor
    def __init__(self, dataset_path, batch_size, preprocessors=None, aug=None, binarize=True, classes=2):
        # store all the required data
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        # open hdf5 dataset for reading and determine the number of entries in the dataset
        self.dataset = h5.File(dataset_path, 'r')
        self.num_images = self.dataset['Labels'].shape[0]

    # data generator
    def generator(self, passes=np.inf):
        # initialize the epoch count

        epochs = 0

        # keep looping infinitely -- the model will stop once we have reach the desired number of epochs
        while epochs < passes:
            # loop over the HDF5 dataset
            for i in np.arange(0, self.num_images, self.batch_size):
                # extract the images and labels from the HDF5 dataset
                images = self.dataset['Features'][i: i + self.batch_size]
                labels = self.dataset['Labels'][i: i + self.batch_size]

                # check if the labels should be binarized
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)

                # check to see if we should preprocess the inputs
                if self.preprocessors is not None:
                    # initialize the list of processed images
                    processed_images = []

                    # loop over the images
                    for image in images:
                        # loop over the preprocessors and apply each to the image
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        # update the list of preprocessed images
                        processed_images.append(image)

                    # update images array to be the preprocessed images
                    images = np.array(processed_images)

                # if the data augmenter exists, apply it
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels, batch_size=self.batch_size))

                # yield a tuple of images and labels
                yield images, labels

            # increment the total number of epochs
            epochs += 1

    # close the dataset
    def close(self):
        self.dataset.close()