from keras.preprocessing.image import img_to_array
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
import imutils
import cv2


# Keras array compatibility preprocessor
class ImageToArrayPreprocessor:
    # constructor
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    # preprocessor
    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)


# resize without aspect ration
class SimplePreprocessor:
    # constructor
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the required parameters
        self.width = width
        self.height = height
        self.inter = inter

    # preprocess
    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)


# Aspect ratio aware preprocessor
class AspectRatioAwarePreprocessor:
    # constructor
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    # preprocess
    def preprocess(self, image):
        # get the height and width of the image, the initialize the deltas
        (h, w) = image.shape[:2]
        dH = 0
        dW = 0

        # resizing
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        # cropping
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]

        # resize the image to provide spatial dimensions to ensure our output image is always a fixed size
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)


# MeanPreprocessor class
class MeanPreprocessor:
    # constructor
    def __init__(self, rMean, gMean, bMean):
        # store the RGB averages
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    # Preprocess
    def preprocess(self, image):
        # split the image into its respective Red, Green, and Blue channels
        (B, G, R) = cv2.split(image.astype('float'))

        # subtract the mean for each channel
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        # merge the channels together and return the image
        return cv2.merge([B, G, R])


# Patch Preprocessor class
class PatchPreprocessor:
    # constructor
    def __init__(self, width, height):
        # store the target width and height of the image
        self.width = width
        self.height = height

    # preprocess
    def preprocess(self, image):
        # extract a random crop from the image with the target width and height
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]


# Crop Preprocessor
class CropPreprocessor:
    # constructor
    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
        # store required arguments
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    # preprocess
    def preprocess(self, image):
        # initialize the list of crops
        crops = []

        # grab the width and height of the image, then use as dimensions to define corners
        (h, w) = image.shape[:2]
        coords = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h]
        ]

        # compute the center crop of the image
        dW = int(0.5 * (w - self.width))
        dH = int(0.5, (h - self.height))
        coords.append([dW, dH, w - dW, h - dH])

        # loop over the coordinates, extract each of the crops and resize each of them to a fixed size
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height), interpolation=self.inter)
            crops.append(crop)

        # check to see if the horizontal flips should be taken
        if self.horiz:
            # compute the horizontal mirror flips for each crop
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        # return the set of crops
        return np.array(crops)
