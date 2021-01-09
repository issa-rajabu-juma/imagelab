import numpy as np


def rank5_accuracy(predictions, labels):
    # initialize the rank-1 and rank-5 accuracies
    rank_1 = 0
    rank_5 = 0
    # new_predictions = []

    # loop over the predictions and the ground-truth labels
    for (prediction_, ground_truth) in zip(predictions, labels):
        # sort the probabilities by their index in descending order
        # so that the more confident guesses are at the front of the list
        prediction_ = np.argsort(prediction_)[::-1]

        # check if the ground-truth label is in the top-5 predictions
        if ground_truth in prediction_[:5]:
            rank_5 += 1

        # check if the ground-truth is in #1 prediction
        if ground_truth == prediction_[0]:
            rank_1 += 1

    # compute the final rank-1 and rank-5 accuracy
    rank_1 /= float(len(labels))
    rank_5 /= float(len(labels))

    # return a tuple of the rank-1 and rank-5 accuracies
    return rank_1, rank_5