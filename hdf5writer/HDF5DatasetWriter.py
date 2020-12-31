from numpy import unicode
import h5py as h5
import os


class HDF5DatasetWriter:
    # constructor
    def __init__(self, dimensions, outputFile, bufSize=100, dataKey='Features'):
        # check to see if the outputFile already exists, and if so raise an exception
        if os.path.exists(outputFile):
            raise ValueError('The supplied output file already exists and can not be overwritten.'
                             ' Manually delete the file before continuing', outputFile)

        # open hdf5 database and create two datasets
        self.h5_database = h5.File(outputFile, 'w')
        self.data = self.h5_database.create_dataset(dataKey, dimensions, dtype='float')
        self.labels = self.h5_database.create_dataset('Labels', (dimensions[0],), stype='int')

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