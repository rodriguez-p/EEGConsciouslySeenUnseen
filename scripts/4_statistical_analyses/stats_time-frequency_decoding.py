import numpy as np
from stats_functions import wilcoxon_fdr_tfdecod
from utils_functions import load_scores_tf
from plotting_functions import plot_stats_timefrequency_decoding
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
        
#%%
FIGURES_DIR = '../figures'
DATA_PATH = '../data'
DECOD_TFR = '../data/decoding_time-frequency'
STATS = '../data/stats'

#%% 
times= np.load("{}/{}".format(DATA_PATH, 'times.npy'), allow_pickle=True)
freqs = np.logspace(np.log10(4), np.log10(50), 25)
sfreq = 256

#%% Target presence

# posterior channels 

presence_postch = load_scores_tf(DECOD_TFR, 'time_frequency_decoding_presence_post-ch', average = False) 

presence_postch_avg = np.mean(presence_postch, axis = 0)
presence_postch_avg = np.reshape(presence_postch_avg, (25, 1024))

# Wilcoxon signed-rank test with FDR correction
stats_file = 'stats_time-frequency_decoding_presence_postch_signpoints.npy'

# check if the analyses have been already run and load stats files
sign_points_path = f'{STATS}/{stats_file}'

if os.path.exists(sign_points_path):
    print("The file exists!")
    presence_postch_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    x = presence_postch
    reject, p_corrected, presence_postch_sign_points = wilcoxon_fdr_tfdecod(x, times, stats_file=stats_file)

# plot results with contour for significant time-frequency bins
title = "Target presence TF (posterior channels) "
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_timefrequency_decoding(presence_postch_avg, sign_points = presence_postch_sign_points, color = 'Blues', 
                                  times = times, title = title, save_figure = save_figure)

# anterior channels 

presence_antch = load_scores_tf(DECOD_TFR, 'time_frequency_decoding_presence_ant-ch', average = False) 

presence_antch_avg = np.mean(presence_antch, axis = 0)
presence_antch_avg = np.reshape(presence_antch_avg, (25, 1024))

# Wilcoxon signed-rank test with FDR correction
stats_file = 'stats_time-frequency_decoding_presence_antch_signpoints.npy'

# check if the analyses have been already run and load stats files
sign_points_path = f'{STATS}/{stats_file}'

if os.path.exists(sign_points_path):
    print("The file exists!")
    presence_antch_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    x = presence_antch
    reject, p_corrected, presence_antch_sign_points = wilcoxon_fdr_tfdecod(x, times, stats_file=stats_file)

# plot results with contour for significant time-frequency bins
title = "Target presence TF (anterior channels)"
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_timefrequency_decoding(presence_antch_avg, sign_points = presence_antch_sign_points, color = 'Blues', 
                                  times = times, title = title, save_figure = save_figure)

#%% Awareness

# posterior channels 

awareness_postch = load_scores_tf(DECOD_TFR, 'time_frequency_decoding_awareness_post-ch', average = False) 

awareness_postch_avg = np.mean(awareness_postch, axis = 0)
awareness_postch_avg = np.reshape(awareness_postch_avg, (25, 1024))

# Wilcoxon signed-rank test with FDR correction
stats_file = 'stats_time-frequency_decoding_awareness_postch_signpoints.npy'

# check if the analyses have been already run and load stats files
sign_points_path = f'{STATS}/{stats_file}'

if os.path.exists(sign_points_path):
    print("The file exists!")
    awareness_postch_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    x = awareness_postch
    reject, p_corrected, awareness_postch_sign_points = wilcoxon_fdr_tfdecod(x, times, stats_file=stats_file)

# plot results with contour for significant time-frequency bins
title = "Target awareness TF (posterior channels)"
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_timefrequency_decoding(awareness_postch_avg, sign_points = awareness_postch_sign_points, color = 'Reds', 
                                  times = times, title = title, save_figure = save_figure)

# anterior channels 

awareness_antch = load_scores_tf(DECOD_TFR, 'time_frequency_decoding_awareness_ant-ch', average = False) 

awareness_antch_avg = np.mean(awareness_antch, axis = 0)
awareness_antch_avg = np.reshape(awareness_antch_avg, (25, 1024))

# Wilcoxon signed-rank test with FDR correction
stats_file = 'stats_time-frequency_decoding_awareness_antch_signpoints.npy'

# check if the analyses have been already run and load stats files
sign_points_path = f'{STATS}/{stats_file}'

if os.path.exists(sign_points_path):
    print("The file exists!")
    awareness_antch_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    x = awareness_antch
    reject, p_corrected, awareness_antch_sign_points = wilcoxon_fdr_tfdecod(x, times, stats_file=stats_file)

# plot results with contour for significant time-frequency bins
title = "Target awareness TF (anterior channels)"
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_timefrequency_decoding(awareness_antch_avg, sign_points = awareness_antch_sign_points, color = 'Reds', 
                                  times = times, title = title, save_figure = save_figure)

#%% Gabor tilt

# posterior channels 

tilt_postch = load_scores_tf(DECOD_TFR, 'time_frequency_decoding_tilt_post-ch', average = False) 

tilt_postch_avg = np.mean(tilt_postch, axis = 0)
tilt_postch_avg = np.reshape(tilt_postch_avg, (25, 1024))

# Wilcoxon signed-rank test with FDR correction
stats_file = 'stats_time-frequency_decoding_tilt_postch_signpoints.npy'

# check if the analyses have been already run and load stats files
sign_points_path = f'{STATS}/{stats_file}'

if os.path.exists(sign_points_path):
    print("The file exists!")
    tilt_postch_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    x = tilt_postch
    reject, p_corrected, tilt_postch_sign_points = wilcoxon_fdr_tfdecod(x, times, stats_file=stats_file)

# plot results with contour for significant time-frequency bins
title = "Target tilt TF (posterior channels)"
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_timefrequency_decoding(tilt_postch_avg, sign_points = tilt_postch_sign_points, color = 'Blues', 
                                  times = times, title = title, save_figure = save_figure)

# anterior channels 

tilt_antch = load_scores_tf(DECOD_TFR, 'time_frequency_decoding_tilt_ant-ch', average = False) 

tilt_antch_avg = np.mean(tilt_antch, axis = 0)
tilt_antch_avg = np.reshape(tilt_antch_avg, (25, 1024))

# Wilcoxon signed-rank test with FDR correction
stats_file = 'stats_time-frequency_decoding_tilt_antch_signpoints.npy'

# check if the analyses have been already run and load stats files
sign_points_path = f'{STATS}/{stats_file}'

if os.path.exists(sign_points_path):
    print("The file exists!")
    tilt_antch_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    x = tilt_antch
    reject, p_corrected, tilt_antch_sign_points = wilcoxon_fdr_tfdecod(x, times, stats_file=stats_file)

# plot results with contour for significant time-frequency bins
title = "Target tilt TF (anterior channels)"
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_timefrequency_decoding(tilt_antch_avg, sign_points = tilt_antch_sign_points, color = 'Blues', 
                                  times = times, title = title, save_figure = save_figure)

#%% combine plots

img1 = mpimg.imread('Target presence TF.png')
img2 = mpimg.imread('Awareness TF.png')
img3 = mpimg.imread('Gabor tilt TF.png')

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
plt.subplots_adjust(wspace=0)

# Plot the images on subplots
axes[0].imshow(img1)
axes[0].axis('off')
axes[0].set_title('Target presence', fontsize = 16, color = 'navy')

axes[1].imshow(img2)
axes[1].axis('off')
axes[1].set_title('Awareness', fontsize = 16, color = 'red')

axes[2].imshow(img3)
axes[2].axis('off')
axes[2].set_title('Gabor tilt', fontsize = 16, color = 'green')

plt.tight_layout(w_pad = 4)
plt.savefig('frequencies_decoding_combined.png', dpi = 1200)
plt.close()
