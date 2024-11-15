"""
Author: Pablo Rodríguez-San Esteban (prodriguez@ugr.es)

Defines and runs functions for the time-frequency decoding on the epoched data for the experimental blocks.
"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.time_frequency import tfr_multitaper

import mne
from mne.decoding import (SlidingEstimator, cross_val_multiscore, Vectorizer)
import numpy as np

# set up directories
DATA_EPOCH='../data/prodriguez/epochdata/' # directory where our epoched data is stored
SAVE_DATA = '../data/prodriguez/decoding_time-frequency' # directory to save our data after running the decoding analyses
FILE_PREFIX = 'PRODRIGUEZ_' # prefix of the files, set up during BrainVision recording

# define functions
def temporal_decoding_tf(subject_id, condition):
    epochs = mne.read_epochs("{}/{}{:06d}-epo.fif".format(DATA_EPOCH, FILE_PREFIX, subject_id))

    # select the condition to load (target presence, awareness or tilt orientation)
    if condition == 'presence':
        epochs = mne.epochs.combine_event_ids(epochs['Absent', 'Present/Seen'], ['Present/Seen/Left', 'Present/Seen/Right'], {'Present': 100})
        epochs = mne.epochs.combine_event_ids(epochs, ['Absent/Unseen'], {'Absent': 101})

    elif condition == 'awareness':
        epochs = mne.epochs.combine_event_ids(epochs['Present'], ['Present/Seen/Left', 'Present/Seen/Right'], {'Present/Seen': 100})
        epochs = mne.epochs.combine_event_ids(epochs, ['Present/Unseen/Left', 'Present/Unseen/Right'], {'Present/Unseen': 101})

    elif condition == 'tilt':
        epochs = mne.epochs.combine_event_ids(epochs['Present/Seen'], ['Present/Seen/Right'], {'Tilt Right': 100})
        epochs = mne.epochs.combine_event_ids(epochs, ['Present/Seen/Left'], {'Tilt Left': 101})

    # load tfr files
    freqs =  np.logspace(np.log10(4), np.log10(50), 25)
    n_cycles = 5

    power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, n_jobs=-1, picks='eeg', average=False)

    # select data (X) from power and labels (y) from epochs
    X = power.data
    del power
    y = epochs.events[:,-1]

    # define the classifier pipeline
    clf = make_pipeline(Vectorizer(), StandardScaler(), SVC(kernel='linear', probability=True, class_weight="balanced", max_iter=-1))

    # apply the SlidingEstimator to fit and test classifier across all time-frequency points
    time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='roc_auc', verbose=True)

    # define cv folds, default is 5, can be 3 for speed
    scores_td = cross_val_multiscore(time_decod, X, y, cv=10, n_jobs = -1)
    
    # compute mean scores across cross-validation splits
    mean_scores_td = np.mean(scores_td, axis=0)

    return mean_scores_td

# run the function across subjects
conditions = ['presence', 'tilt', 'awareness']
avg_td_tf = None

for c in conditions:
    for s in range(10, 43):
        if s in [11, 24, 26]:
            pass
        else:
            avg_td_tf = temporal_decoding_tf(s, c)
            np.save("{}/S{}_temporal_decoding_tf_{}_lsvm_experimental.npy".format(SAVE_DATA, id, c), avg_td_tf, allow_pickle=True, fix_imports=True)