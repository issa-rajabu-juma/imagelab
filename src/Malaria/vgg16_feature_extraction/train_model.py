from sklearn.metrics import classification_report
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras import models
from keras import layers
import numpy as np
import h5py as h5
import os

# Specify working directories
base_dir = 'C:/Users/Tajr/Desktop/imagelab/'
dataset_file = os.path.join(base_dir, 'datasets/hdf5/malaria.hdf5')
plot_dir = os.path.join(base_dir, 'src/Malaria/vgg16_feature_extraction/plots/train_process.jpg')

# opening up a dataset
dataset = h5.File(dataset_file, 'r')

# data splitting
index = 0
train_index = int(dataset['Features'].shape[0] * 0.5)
val_index = int(dataset['Features'].shape[0] * 0.75)

# define a network
model = models.Sequential()
model.add(layers.Dense(63, input_shape=(dataset['Features'].shape[1],), activation='relu'))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dropout(0.25))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# compile a model
learning_rate = 0.001
epchs = 100
opt = SGD(lr=learning_rate, decay=learning_rate/epchs, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

# train a model
print('[INFO] training...')
history = model.fit(
    dataset['Features'][index:index + train_index],
    dataset['Labels'][index:index + train_index],
    epochs=100,
    batch_size=32,
    validation_data=(
        dataset['Features'][index + train_index:val_index],
        dataset['Labels'][index + train_index:val_index])
)
print('[INFO] train complete.')

# plots variables
history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = np.arange(1, len(history_dict['acc']) + 1)

# plotting
plt.style.use('ggplot')
plt.plot(epochs, acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.plot(epochs, loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.title('Train Process')
plt.legend()
plt.savefig(plot_dir)

# Evaluation
print('[INFO] evaluating...')
result = model.evaluate(dataset['Features'][val_index:], dataset['Labels'][val_index:])
print(result)

# Prediction
predictions = model.predict(dataset['Features'][val_index:], batch_size=32)


# Format predictions
def format_predictions(prediction_list):
    # initialize a new predictions bin
    new_predictions = []

    # loop over predictions to format them
    for prediction in prediction_list:
        if prediction > 0.5:
            new_prediction = 1
        else:
            new_prediction = 0

        new_predictions.append(new_prediction)

    return new_predictions


# Format returned predictions and print out a classification report
predictions = format_predictions(predictions)
print(classification_report(dataset['Labels'][val_index:], predictions, target_names=dataset['Label Names']))
