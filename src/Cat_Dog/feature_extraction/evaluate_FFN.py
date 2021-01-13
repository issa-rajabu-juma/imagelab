from keras.models import load_model
from config import cat_dog_config as config
import h5py as h5
from sklearn.metrics import classification_report
from utils.ranked import rank5_accuracy
from keras.utils import to_categorical

# get model
model = load_model(config.FFN_MODEL_PATH)

# get data
dataset = h5.File(config.CAT_DOG_FEATURES, 'r')

# acquire test data split
index = 0
train_index = int(dataset['Features'].shape[0] * 0.5)
val_index = int(dataset['Features'].shape[0] * 0.75)

labels = to_categorical(dataset['Labels'])

# evaluation
print('[INFO] evaluating...')
result = model.evaluate(dataset['Features'][val_index:], labels[val_index:])
print(result)

# make predictions
print('[INFO] predicting...')
predictions = model.predict(dataset['Features'][val_index:], batch_size=50)

# print report
report = classification_report(dataset['Labels'][val_index:], predictions.argmax(axis=1), target_names=dataset['Label Names'][:])
print(report)

# # calculate rank-1 accuracy
ground_truth = dataset['Labels'][val_index:]
score = 0
for pred, gt in zip(predictions.argmax(axis=1), ground_truth):
    if pred == gt:
        score += 1
rank_1 = score/float(len(ground_truth))

print('[INFO] rank-1: {}%'.format(rank_1 * 100))
