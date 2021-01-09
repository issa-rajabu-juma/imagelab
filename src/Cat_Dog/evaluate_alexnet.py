from config import cat_dog_config as config
from preprocessing import preprocessor
from keras.models import load_model
from hdf5_datasets.dataset import HDF5DatasetGenerator
from utils import ranked
# import build_cat_dog_dataset
from sklearn.metrics import classification_report
import json
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
