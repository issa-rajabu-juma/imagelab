from config import cat_dog_config as config
from preprocessing import preprocessor
from keras.models import load_model
from hdf5_datasets.dataset import HDF5DatasetGenerator
from utils import ranked
# import build_cat_dog_dataset
from sklearn.metrics import classification_report
import json
import progressbar
import numpy as np

# load the RGB means
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
simple_preprocessor = preprocessor.SimplePreprocessor(171, 171)
mean_preprocessor = preprocessor.MeanPreprocessor(means['R'], means['G'], means['B'])
crop_preprocessor = preprocessor.CropPreprocessor(171, 171)
i2a_preprocessor = preprocessor.ImageToArrayPreprocessor()

# load the pretrained model
model = load_model(config.MODEL_PATH)

# initialize the testing dataset generator, then make predictions on the testing data
print('[INFO] predicting on test data (no crops)...')
test_generator = HDF5DatasetGenerator(config.TEST_HDF5,
                                      batch_size=50,
                                      classes=2,
                                      preprocessors=[simple_preprocessor, mean_preprocessor, i2a_preprocessor])

predictions = model.predict_generator(test_generator.generator(),
                                      steps=test_generator.num_images // 50,
                                      max_queue_size=50 * 2)

report = classification_report(test_generator.dataset['Labels'],
                               predictions.argmax(axis=1),
                               target_names=test_generator.dataset['Label Names'][:])

print(report)

# compute the rank-1 and rank-5 accuracies
(rank_1, _) = ranked.rank5_accuracy(predictions, test_generator.dataset['Labels'][:])
print('[INFO] rank-1: {:.2f}%'.format(rank_1 * 100))


test_generator.close()


# reinitialize test generator while excluding simple preprocessor
testGen = HDF5DatasetGenerator(config.TEST_HDF5, batch_size=50, preprocessors=[mean_preprocessor], classes=2)
prediction_ = []

# initialize progress bar
widgets = ['Evaluating: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
prog = progressbar.ProgressBar(maxval=testGen.num_images // 50, widgets=widgets).start()

# loop over a single pass of the test data
for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):
    # loop over each of the individual images
    for image in images:
        # apply the crop preprocessor to the image to generate 10 separate crops, then convert them from
        # images to arrays
        crops = crop_preprocessor.preprocess(image)
        crops = np.array([i2a_preprocessor.preprocess(crop) for crop in crops], dtype='float32')

        # make predictions on the crops and then average them together to obtain the final prediction
        pred = model.predict(crops)
        prediction_.append(pred.mean(axis=0))

    # update the progress bar
    prog.update(i)

# finish the progress bar
prog.finish()

# classification report
prediction__ = np.array(prediction_)
report = classification_report(testGen.dataset['Labels'], prediction__.argmax(axis=1), target_names=testGen.dataset['Label Names'][:])
print(report)

# compute the rank-1 accuracy
print('[INFO] predicting on test data(with crops)...')
(rank_1, _) = ranked.rank5_accuracy(prediction_, testGen.dataset['Labels'][:])
print('[INFO] rank-1: {:.2f}%'.format(rank_1 * 100))
testGen.close()

