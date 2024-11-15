import os
os.chdir('../scripts')

from plotting_functions import plot_stats_generalization
from stats_functions import wilcoxon_fdr_generalization
from utils_functions import load_scores_gen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%%
DATA_PATH = '/data/prodriguez'
DECOD_TFR = '/data/prodriguez/decoding_time-frequency'
FIGURES_DIR = '/data/prodriguez/figures'

#%% 
times = np.load("{}/{}".format(DATA_PATH, 'times.npy'), allow_pickle=True) 

#%% Target presence

# posterior channels

presence_timegen_postch = load_scores_gen(DECOD_TFR, 'temporal-generalization_tf_presence_post-ch', average = False)

presence_timegen_postch_avg = np.mean(presence_timegen_postch, axis=0)

stats_file = 'stats_temporal_generalization_tf_presence'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_generalization_tf_presence_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    presence_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, presence_sign_points = wilcoxon_fdr_generalization(presence_timegen_postch, times, stats_file)

title='Target presence TGM (Posterior electrodes)'
save_figure = "{}/{}".format(FIGURES_DIR, title)
plot_stats_generalization(presence_timegen_postch_avg[512:, 512:], presence_sign_points[512:, 512:], times, title, save_figure)

# anterior channels

presence_timegen_antch = load_scores_gen(DECOD_TFR, 'temporal-generalization_tf_presence_ant-ch', average = False)

presence_timegen_antch_avg = np.mean(presence_timegen_antch, axis=0)

stats_file = 'stats_temporal_generalization_tf_presence'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_generalization_tf_presence_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    presence_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, presence_sign_points = wilcoxon_fdr_generalization(presence_timegen_antch, times, stats_file)

title='Target presence TGM (Anterior electrodes)'
save_figure = "{}/{}".format(FIGURES_DIR, title)
plot_stats_generalization(presence_timegen_antch_avg[512:, 512:], presence_sign_points[512:, 512:], times, title, save_figure)

#%% Subject awareness

# posterior channels

awareness_timegen_postch = load_scores_gen(DECOD_TFR, 'temporal-generalization_tf_awareness_post-ch', average = False)

awareness_timegen_postch_avg = np.mean(awareness_timegen_postch, axis=0)

stats_file = 'stats_temporal_generalization_tf_awareness'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_generalization_tf_awareness_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    awareness_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, awareness_sign_points = wilcoxon_fdr_generalization(awareness_timegen_postch, times, stats_file)

title='Target awareness TGM (Posterior electrodes)'
save_figure = "{}/{}".format(FIGURES_DIR, title)
plot_stats_generalization(awareness_timegen_postch_avg[512:, 512:], awareness_sign_points[512:, 512:], times, title, save_figure)

# anterior channels

awareness_timegen_antch = load_scores_gen(DECOD_TFR, 'temporal-generalization_tf_awareness_ant-ch', average = False)

awareness_timegen_antch_avg = np.mean(awareness_timegen_antch, axis=0)

stats_file = 'stats_temporal_generalization_tf_awareness'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_generalization_tf_awareness_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    awareness_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, awareness_sign_points = wilcoxon_fdr_generalization(awareness_timegen_antch, times, stats_file)

title='Target awareness TGM (Anterior electrodes)'
save_figure = "{}/{}".format(FIGURES_DIR, title)
plot_stats_generalization(awareness_timegen_antch_avg[512:, 512:], awareness_sign_points[512:, 512:], times, title, save_figure)

#%% Gabor tilt

# posterior channels

tilt_timegen_postch = load_scores_gen(DECOD_TFR, 'temporal-generalization_tf_tilt_post-ch', average = False)

tilt_timegen_postch_avg = np.mean(tilt_timegen_postch, axis=0)

stats_file = 'stats_temporal_generalization_tf_tilt'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_generalization_tf_tilt_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    tilt_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, tilt_sign_points = wilcoxon_fdr_generalization(tilt_timegen_postch, times, stats_file)

title='Target tilt TGM (Posterior electrodes)'
save_figure = "{}/{}".format(FIGURES_DIR, title)
plot_stats_generalization(tilt_timegen_postch_avg[512:, 512:], tilt_sign_points[512:, 512:], times, title, save_figure)

# anterior channels

tilt_timegen_antch = load_scores_gen(DECOD_TFR, 'temporal-generalization_tf_tilt_ant-ch', average = False)

tilt_timegen_antch_avg = np.mean(tilt_timegen_antch, axis=0)

stats_file = 'stats_temporal_generalization_tf_tilt'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_generalization_tf_tilt_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    tilt_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, tilt_sign_points = wilcoxon_fdr_generalization(tilt_timegen_antch, times, stats_file)

title='Target tilt TGM (Anterior electrodes)'
save_figure = "{}/{}".format(FIGURES_DIR, title)
plot_stats_generalization(tilt_timegen_antch_avg[512:, 512:], tilt_sign_points[512:, 512:], times, title, save_figure)

#%% combine plots
    
os.chdir('C:/Users/pablr/Desktop/prodriguez_eeg_decoding/figures')

img1 = mpimg.imread('Target presence TGM (Anterior electrodes).png')
img2 = mpimg.imread('Target awareness TGM (Anterior electrodes).png')
img3 = mpimg.imread('Target tilt TGM (Anterior electrodes).png')
img4 = mpimg.imread('Target presence TGM (Posterior electrodes).png')
img5 = mpimg.imread('Target awareness TGM (Posterior electrodes).png')
img6 = mpimg.imread('Target tilt TGM (Posterior electrodes).png')

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
plt.subplots_adjust(wspace=0, hspace=0)

# Plot the images on subplots
axes[0, 0].imshow(img1)
axes[0, 0].axis('off')
axes[0, 0].set_title('Target presence', x = 0.48, fontsize = 16, color = 'black')
axes[0, 0].text(0, 0.15, 'A', fontsize = 16, fontweight='bold')

axes[0, 1].imshow(img2)
axes[0, 1].axis('off')
axes[0, 1].set_title('Awareness', x = 0.48, fontsize = 16, color = 'black')
axes[0, 1].text(0, 0.15, 'B', fontsize = 16, fontweight='bold')

axes[0, 2].imshow(img3)
axes[0, 2].axis('off')
axes[0, 2].set_title('Gabor tilt', x = 0.48, fontsize = 16, color = 'black')
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
plt.savefig('Fig. S3.svg')
plt.close()