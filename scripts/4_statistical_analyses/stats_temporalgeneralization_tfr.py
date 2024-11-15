from plotting_functions import plot_stats_generalization
from stats_functions import wilcoxon_fdr_generalization
from utils_functions import load_scores_gen
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%%
DATA_PATH = '/data/prodriguez'
DECOD_TFR = '/data/prodriguez/decoding_time-frequency'
FIGURES_DIR = '/data/prodriguez/figures'

#%% 
times= np.load("{}/{}".format(DATA_PATH, 'times.npy'), allow_pickle=True) 

#%% Target presence
presence_timegen = load_scores_gen(DECOD_TFR, 'temporal-generalization_presence_tfr', average = False)

presence_timegen_avg = np.mean(presence_timegen, axis=0)

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
    reject, p_corrected, presence_sign_points = wilcoxon_fdr_generalization(presence_timegen, times, stats_file)

title='Target presence TGM'
save_figure = "{}/{}".format(FIGURES_DIR, title)
plot_stats_generalization(presence_timegen_avg[512:, 512:], presence_sign_points[512:, 512:], times, title, save_figure)
    
#%% Subject awareness
awareness_timegen = load_scores_gen(DECOD_TFR, 'temporal-generalization_awareness_tfr', average = False)

awareness_timegen_avg = np.mean(awareness_timegen, axis=0)

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
    reject, p_corrected, awareness_sign_points = wilcoxon_fdr_generalization(awareness_timegen, times, stats_file)

title='Subject awareness TGM'
save_figure = "{}/{}".format(FIGURES_DIR, title)
plot_stats_generalization(awareness_timegen_avg[500:, 500:], awareness_sign_points[500:, 500:], times, title, save_figure)

#%% Gabor tilt
tilt_timegen = load_scores_gen(DECOD_TFR, 'temporal-generalization_tilt_tfr', average = False)

tilt_timegen_avg = np.mean(tilt_timegen, axis=0)

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
    reject, p_corrected, tilt_sign_points = wilcoxon_fdr_generalization(tilt_timegen, times, stats_file)

title='Gabor tilt TGM'
save_figure = "{}/{}".format(FIGURES_DIR, title)
plot_stats_generalization(tilt_timegen_avg[512:, 512:], tilt_sign_points[512:, 512:], times, title, save_figure)

#%%

os.chdir(FIGURES_DIR)

# Load the images
img1 = mpimg.imread('Target presence TGM.png')
img2 = mpimg.imread('Subject awareness TGM.png')
img3 = mpimg.imread('Gabor tilt TGM.png')

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
plt.subplots_adjust(wspace=0)

# Plot the images on subplots
axes[0].imshow(img1)
axes[0].axis('off')
axes[0].set_title('Target presence', x = 0.46, fontsize = 16, color = 'navy', ha = 'center', va = 'center')
axes[0].text(0, 0.15, 'A', fontsize = 16, fontweight='bold')

axes[1].imshow(img2)
axes[1].axis('off')
axes[1].set_title('Awareness', x = 0.47, fontsize = 16, color = 'red', ha = 'center', va = 'center')
axes[1].text(0, 0.15, 'B', fontsize = 16, fontweight='bold')

axes[2].imshow(img3)
axes[2].axis('off')
axes[2].set_title('Gabor tilt', x = 0.45, fontsize = 16, color = 'green', ha = 'center', va = 'center')
axes[2].text(0, 0.15, 'C', fontsize = 16, fontweight='bold')

# plt.show()
plt.tight_layout(w_pad = 4)
plt.savefig('Fig. 6.svg', format = 'svg', dpi = 1200)
plt.close()

