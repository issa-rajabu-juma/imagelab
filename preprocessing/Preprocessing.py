from keras.preprocessing.image import img_to_array
from sklearn.utils import shuffle
from imutils import paths
import numpy as np
import progressbar
import imutils
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


# Preprocessing class
class Preprocessing:
    # constructor
    def __init__(self, width, height, inter=cv2.INTER_AREA, dataFormat=None):
        self.width = width
        self.height = height
        self.inter = inter
        self.dataFormat = dataFormat

    # resize image
    def resize(self, image):
        # get height and width of an image and then initialize deltas
        #
        # h = image.shape[0]
        # w = image.shape[1]
        # dh = 0
        # dw = 0
        #
        # # resize along longest side
        # if w < h:
        #     image = imutils.resize(image, width=self.width, inter=self.inter)
        #     dh = int((image.shape[0] + self.height) / 2.0)
        # else:
        #     image = imutils.resize(image, height=self.height, inter=self.inter)
        #     dw = int((image.shape[1] + self.width) / 2.0)
        #
        # # crop the image
        # h = image.shape[0]
        # w = image.shape[1]
        # image = image[dh:h - dh, dw:w - dw]

        # return an image while maintaining spatial dimension
        try:
            image = cv2.resize(image, (self.width, self.height), interpolation=self.inter)
        except Exception as e:
            print(str(e))

        return image

    #  convert image to array
    def img2array(self, image):
        return img_to_array(image, data_format=self.dataFormat)

    # load data
    def load_data(self, path_list):
        data = []
        load_widget = ['Load Data: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]

        progress = progressbar.ProgressBar(maxval=len(path_list), widgets=load_widget).start()
        # loop over the path list to read and process image data
        for (i, path) in enumerate(path_list):
            image = cv2.imread(path)
            image = self.resize(image)
            image = self.img2array(image)
            data.append(image)
            progress.update(i)

        # convert to array and resize the pixels
        data = np.array(data).astype('float') / 255.0
        progress.finish()

        return data
