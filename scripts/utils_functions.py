import numpy as np
import os

subject_ids = list(range(10, 43))
wrong_subjects = [11, 24, 26, 38]
valid_subjects = np.array(list(set(subject_ids) - set(wrong_subjects)))

def load_scores(dir_path, file_name, average, valid_subjects = valid_subjects):
    """
    Loads decoding scores for all valid subjects. The valid subjects list is defined by 
    subtracting discarded subjects from the whole sample. If average == True, it returns
    an array with the grand mean (averaged data for all subjects). If average == False,
    it returns an array with data for all subjects, not averaged.
    Loaded data must be in .npy format. 

    Args: 
        dir_path (str): dir_pathectory where the data file_names are stored
        file_name (str): file_name name
        average (bool): whether we want the scores averaged for all subjects or not

    Returns:
        scores (array): array with the loaded scores
    """  
    scores = []
    
    for _, id in enumerate(valid_subjects):
        file_path = f'{dir_path}/S{id}_{file_name}.npy'
        if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:
            subject_scores = np.load(file_path, allow_pickle=True)
            if isinstance(subject_scores, np.ndarray) and subject_scores.size > 0:
                scores.append(subject_scores)

    if average == True:
        scores = np.mean(scores, axis = 0)
    else:
        pass
    
    return scores

def load_scores_tf(dir_path, file_name, average, valid_subjects = valid_subjects):
    """
    Loads decoding scores for all valid subjects. The valid subjects list is defined by 
    subtracting discarded subjects from the whole sample. If average == True, it returns
    an array with the grand mean (averaged data for all subjects). If average == False,
    it returns an array with data for all subjects, not averaged.
    Loaded data must be in .npy format. Used for time-frequency formatted data
    (25 frequencies * 1024 time points).

    Args: 
        dir_path (str): dir_pathectory where the data file_names are stored
        file_name (str): file_name name
        average (bool): whether we want the scores averaged for all subjects or not

    Returns:
        scores (array): array with the loaded scores
    """  
    scores = []
    
    for _, id in enumerate(valid_subjects):
        file_path = f'{dir_path}/S{id}_{file_name}.npy'
        if os.path.exists(file_path):
            subject_scores = np.load(file_path, allow_pickle=True)
            if isinstance(subject_scores, np.ndarray) and subject_scores.size > 0:
                scores.append(subject_scores)

    if average == True:
        scores = np.mean(scores, axis = 0)
    else:
        pass
    
    # scores = np.stack(scores)
    
    return scores

def load_scores_gen(dir_path, file_name, average, valid_subjects = valid_subjects):
    scores = []
    for _, id in enumerate(valid_subjects):
        file_path = f'{dir_path}/S{id}_{file_name}.npy'
        if os.path.exists(file_path) and os.path.getsize(file_path) > 1024 * 1024:
            subject_scores = np.load(file_path, allow_pickle=True)
            if isinstance(subject_scores, np.ndarray) and subject_scores.size > 0:
                scores.append(subject_scores)
        
    if average:
        scores = np.mean(scores, axis=0)
        
    else:
        pass
        
    return scores