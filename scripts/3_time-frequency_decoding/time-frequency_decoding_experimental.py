"""
Author: Pablo Rodríguez-San Esteban (prodriguez@ugr.es)

Defines and runs functions for the time-frequency decoding on the epoched data for the experimental blocks.
"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from mne.time_frequency import tfr_multitaper
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import mne
from mne.decoding import (SlidingEstimator, cross_val_multiscore, Vectorizer)
import numpy as np

# set up directories
DATA_EPOCH='../data/epochdata/' # directory where our epoched data is stored
DATA_TFR = '../data/time-frequency' # directory where the decoding scores will be saved
SAVE_DATA = '../data/time-frequency_decoding' # directory to save our data after running the decoding analyses
FILE_PREFIX = 'PRODRIGUEZ_' # prefix of the files, set up during BrainVision recording

# define functions
def time_frequency_decoding(subject_id, condition, model):

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
    freqs = np.logspace(np.log10(4), np.log10(50), 25)
    n_cycles = 5

    power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, n_jobs=-1, picks='eeg', average=False)

    # transforms data from (n_trials, n_channels, n_frequencies, n_times) to (n_trials, n_channels, n_time-frequency); adapted from https://github.com/kingjr/decod_unseen_maintenance/
    tfr = power.data
    n_trial, n_chan, n_freq, n_time = tfr.shape
    tfr = np.reshape(tfr, [n_trial, n_chan, n_freq * n_time])
    tfr.data

    # select data (X) from power and labels (y) from epochs
    X = tfr
    y = epochs.events[:,-1]

    # define the classifier pipeline
    clf = make_pipeline(Vectorizer(), StandardScaler(), SVC(kernel='linear', probability=True, class_weight="balanced", max_iter=-1))
    
    # apply the SlidingEstimator to fit and test classifier across all time-frequency points
    time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='roc_auc', verbose=True)
    
    # define cv folds, default is 5, can be 3 for speed
    scores_td = cross_val_multiscore(time_decod, X, y, cv=10, n_jobs=-1)
    
    # compute mean scores across cross-validation splits
    mean_scores_td = np.mean(scores_td, axis=0)
    return mean_scores_td

# run the function across subjects
conditions = ['presence', 'awareness', 'tilt']

for c in conditions:
    for s in range(10, 43):
        if s in [11, 24, 26]:
            pass
        else:
            avg_tfd = time_frequency_decoding(s, c)
            np.save("{}/S{}_time_frequency_decoding_{}_lsvm_experimental.npy".format(SAVE_DATA, s, c), avg_tfd, allow_pickle=True, fix_imports=True)
