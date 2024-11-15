import os
os.chdir('../scripts')

import numpy as np
from stats_functions import wilcoxon_fdr_tfdecod
from utils_functions import load_scores_tf
from plotting_functions import plot_stats_timefrequency_decoding
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
        
#%%
FIGURES_DIR = '../figures'
DATA_PATH = '../data'
DECOD_TFR = '../data/decoding_time-frequency'

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
sign_points_path = f'{DATA_PATH}/stats/{stats_file}'

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
sign_points_path = f'{DATA_PATH}/stats/{stats_file}'

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

#%%
stats_file = 'stats_time-frequency_decoding_presence_antvspost_signpoints.npy'

x = presence_antch
y = presence_postch

reject, p_corrected, presence_sign_points = wilcoxon_fdr_tfdecod(x_data = x, times = times, stats_file=stats_file, y_data = y)

title = "Target presence TF (anterior vs posterior channels)"
save_figure = "{}/{}".format(FIGURES_DIR, title)
plot_stats_timefrequency_decoding(presence_antch_avg, sign_points = presence_sign_points, color = 'Blues', 
                                  times = times, title = title, save_figure = save_figure)

#%% Awareness

# posterior channels 

awareness_postch = load_scores_tf(DECOD_TFR, 'time_frequency_decoding_awareness_post-ch', average = False) 

awareness_postch_avg = np.mean(awareness_postch, axis = 0)
awareness_postch_avg = np.reshape(awareness_postch_avg, (25, 1024))

# Wilcoxon signed-rank test with FDR correction
stats_file = 'stats_time-frequency_decoding_awareness_postch_signpoints.npy'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/{stats_file}'

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
sign_points_path = f'{DATA_PATH}/stats/{stats_file}'

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
sign_points_path = f'{DATA_PATH}/stats/{stats_file}'

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

plot_stats_timefrequency_decoding(tilt_postch_avg, sign_points = tilt_postch_sign_points, color = 'Greens', 
                                  times = times, title = title, save_figure = save_figure)

# anterior channels 

tilt_antch = load_scores_tf(DECOD_TFR, 'time_frequency_decoding_tilt_ant-ch', average = False) 

tilt_antch_avg = np.mean(tilt_antch, axis = 0)
tilt_antch_avg = np.reshape(tilt_antch_avg, (25, 1024))

# Wilcoxon signed-rank test with FDR correction
stats_file = 'stats_time-frequency_decoding_tilt_antch_signpoints.npy'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/{stats_file}'

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

plot_stats_timefrequency_decoding(tilt_antch_avg, sign_points = tilt_antch_sign_points, color = 'Greens', 
                                  times = times, title = title, save_figure = save_figure)

#%% combine plots
    
os.chdir('C:/Users/pablr/Desktop/prodriguez_eeg_decoding/figures')

img1 = mpimg.imread('Target presence TF (anterior channels).png')
img2 = mpimg.imread('Target awareness TF (anterior channels).png')
img3 = mpimg.imread('Target tilt TF (anterior channels).png')
img4 = mpimg.imread('Target presence TF (posterior channels).png')
img5 = mpimg.imread('Target awareness TF (posterior channels).png')
img6 = mpimg.imread('Target tilt TF (posterior channels).png')


fig, axes = plt.subplots(2, 3, figsize=(20, 12))
plt.subplots_adjust(wspace=0, hspace=0)

# Plot the images on subplots
axes[0, 0].imshow(img1)
axes[0, 0].axis('off')
axes[0, 0].set_title('Target presence', x = 0.50, fontsize = 16, color = 'black')
axes[0, 0].text(0, 0.15, 'A', fontsize = 16, fontweight='bold')

axes[0, 1].imshow(img2)
axes[0, 1].axis('off')
axes[0, 1].set_title('Awareness', x = 0.50, fontsize = 16, color = 'black')
axes[0, 1].text(0, 0.15, 'B', fontsize = 16, fontweight='bold')

axes[0, 2].imshow(img3)
axes[0, 2].axis('off')
axes[0, 2].set_title('Gabor tilt', x = 0.50, fontsize = 16, color = 'black')
axes[0, 2].text(0, 0.15, 'C', fontsize = 16, fontweight='bold')

axes[1, 0].imshow(img4)
axes[1, 0].axis('off')
# axes[3].set_title('Target presence', x = 0.55, fontsize = 16, color = 'navy')
axes[1, 0].text(0, 0.15, 'D', fontsize = 16, fontweight='bold')

axes[1, 1].imshow(img5)
axes[1, 1].axis('off')
# axes[4].set_title('Awareness', x = 0.55, fontsize = 16, color = 'red')
axes[1, 1].text(0, 0.15, 'E', fontsize = 16, fontweight='bold')

axes[1, 2].imshow(img6)
axes[1, 2].axis('off')
# axes[5].set_title('Gabor tilt', x = 0.55, fontsize = 16, color = 'green')
axes[1, 2].text(0, 0.15, 'F', fontsize = 16, fontweight='bold')

plt.show()
plt.tight_layout(w_pad = 4)
plt.savefig('Fig. S4.svg')
plt.close()
