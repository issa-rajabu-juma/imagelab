from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainMonitor(BaseLogger):
    # constructor
    def __init__(self, figure_path, json_path=None, startAt=0):
        # store the required parameters
        super(TrainMonitor, self).__init__()
        self.figure_path = figure_path
        self.json_path = json_path
        self.startAt = startAt

    # on train begin
    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}

        # if the JSON path exists, load the training history
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.H = json.load(open(self.json_path).read())

                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and trim any entries that are past the starting epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    # on epoch end
    def on_epoch_end(self, epoch, logs=None):
        # loop over the logs and update the loss, accuracy, etc for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        # check to see if the training history should be serialized to file
        if self.json_path is not None:
            f = open(self.json_path, 'w')
            f.write(json.dumps(str(self.H)))
            f.close()

        # construct the plots but ensure at least two epochs have passed before plotting
        if len(self.H['acc']) > 1:
            # plots the loss and accuracy
            N = np.arange(0, len(self.H['acc']))
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(N, self.H['acc'], label='Train Accuracy')
            plt.plot(N, self.H['val_acc'], label='Validation Accuracy')
            plt.plot(N, self.H['loss'], label='Train Loss')
            plt.plot(N, self.H['val_loss'], label='Validation Loss')
            plt.title('Training loss and accuracy [Epoch {}]'.format(len(self.H['acc'])))
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy/Loss')
            plt.legend()
            plt.savefig(self.figure_path)
            plt.close()
