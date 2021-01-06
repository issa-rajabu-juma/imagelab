from sklearn.utils import shuffle
from imutils import paths
import progressbar
import numpy as np
import cv2


# collect image path and extract labels
def load_path_and_labels(images_dir):
    path_list = shuffle(list(paths.list_images(images_dir)))
    labels = []

    # loop over the paths to extract labels
    for path in path_list:
        label = path.split('/')[-1].split()[0]
        labels.append(label)

    # convert labels into a numpy array
    labels = np.array(labels)

    return path_list, labels


# DataPreprocessor class
class DataLoader:
    # constrictor
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

    def load_data(self, path_list):
        data = []
        load_widget = ['Load Data: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]

        progress = progressbar.ProgressBar(maxval=len(path_list), widgets=load_widget).start()
        # loop over the path list to read and process image data
        for (i, path) in enumerate(path_list):
            image = cv2.imread(path)

            # check to see if preprocessors list is provided
            if self.preprocessors is not None:
                # loop over the preprocessor
                for p in self.preprocessors:
                    image = p.preprocess(image)
                    data.append(image)

            # update progress bar
            progress.update(i)

        # convert to array and resize the pixels
        data = np.array(data).astype('float32') / 255.0
        progress.finish()

        return data

