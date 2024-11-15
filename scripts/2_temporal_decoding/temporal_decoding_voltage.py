"""
Author: Pablo Rodríguez-San Esteban (prodriguez@ugr.es)

Defines and runs functions for the temporal decoding on the epoched data for the experimental blocks.
"""

# import packages
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import mne
from mne.decoding import (SlidingEstimator, cross_val_multiscore, Vectorizer)

# set up directories
DATA_EPOCH = '../data/epochdata' # directory where our epoched data is stored
SAVE_DATA = '../data/temporal_decoding' # folder to save our data after running the decoding analyses
FILE_PREFIX = 'PRODRIGUEZ_' # prefix of the files, set up during BrainVision recording

# define the temporal decoding functions
def temporal_decoding(subject_id, condition):
    # read epoched data
    epochs = mne.read_epochs("{}/{}{:06d}-epo.fif".format(DATA_EPOCH, FILE_PREFIX, subject_id))

    # select the condition to load (target presence, awareness or tilt orientation), and combine event codes
    if condition == 'presence':
        epochs = mne.epochs.combine_event_ids(epochs['Absent', 'Present/Seen'], ['Present/Seen/Left', 'Present/Seen/Right'], {'Present': 100})
        epochs = mne.epochs.combine_event_ids(epochs, ['Absent/Seen', 'Absent/Unseen'],  {'Absent': 101})

    elif condition == 'awareness':
        epochs = mne.epochs.combine_event_ids(epochs['Present'], ['Present/Seen/Left', 'Present/Seen/Right'], {'Present/Seen': 100})
        epochs = mne.epochs.combine_event_ids(epochs, ['Present/Unseen/Left', 'Present/Unseen/Right'], {'Present/Unseen': 101})

    elif condition == 'tilt':
        epochs = mne.epochs.combine_event_ids(epochs['Present/Seen'], ['Present/Seen/Right'], {'Tilt Right': 100})
        epochs = mne.epochs.combine_event_ids(epochs, ['Present/Seen/Left'], {'Tilt Left': 101})

    # select data (X) and labels (y) from epochs
    X = epochs.get_data()
    y = epochs.events[:,-1]

    # define the classifier pipeline
    clf = make_pipeline(Vectorizer(), StandardScaler(), SVC(kernel='linear', probability=True, class_weight="balanced", max_iter=-1))
    
    # apply the slidingestimator to fit and test classifier across all time points
    time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='roc_auc', verbose=True)

    # define cross-validator and cv splits
    scores_td = cross_val_multiscore(time_decod, X, y, cv=10, n_jobs=-1)

    # compute mean scores across cross-validation splits
    mean_scores_td = np.mean(scores_td, axis=0)
    return mean_scores_td

# run classifier across subjects for the different conditions
avg_td = None
conditions = ['presence', 'awareness', 'tilt']

for c in conditions:
    for s in range(10, 43): # the range of all our subjects, in this case (10, 43)
        if s in [11, 24, 26]: # indicate subjects excluded from the analysis
            pass
        else:
            mean_scores_id = temporal_decoding(s, c)
            np.save("{}/S{}_temporal_decoding_{}_lsvm_experimental.npy".format(SAVE_DATA, s, c)) # save subject data
