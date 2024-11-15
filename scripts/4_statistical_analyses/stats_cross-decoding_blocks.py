
import os
os.chdir("../scripts")

from plotting_functions import plot_stats_generalization
from stats_functions import wilcoxon_fdr_generalization
import numpy as np
from utils_functions import load_scores_gen
import matplotlib.pyplot as plt
from scipy.stats import sem
import seaborn as sns
import matplotlib.image as mpimg

#%%
FIGURES_DIR = '../figures'
DATA_PATH = '../data'
DECOD_TFR = '../data/decoding_time-frequency'

#%% 
times= np.load("{}/{}".format(DATA_PATH, 'times.npy'), allow_pickle=True) 

#%% SEEN
crossdecod_presence_seen = load_scores_gen(DECOD_TFR, 'cross-decoding_blocks_seen_generalizer_tf', average = False)

crossdecod_presence_seen_avg = np.mean(crossdecod_presence_seen, axis=0)
crossdecod_presence_seen_se = np.mean(np.diag(sem(crossdecod_presence_seen)), axis=0)

stats_file = 'stats_cross-decoding_blocks_presence_seen'

# check if the analyses have been already run and load stats files
sign_points_path = "D:\EEG_decoding_offline\stats\stats_cross-decoding_blocks_presence_seen_signpoints.npy"

if os.path.exists(sign_points_path):
    print("The file exists!")
    crossdecod_presence_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, crossdecod_presence_sign_points = wilcoxon_fdr_generalization(crossdecod_presence_seen, times, stats_file)

title='Cross-decoding Localizer-Experimental Matrix (Seen)'
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_generalization(crossdecod_presence_seen_avg[500:, 500:], crossdecod_presence_sign_points[500:, 500:], 
                          times, title, save_figure)

# plot_stats_generalization(crossdecod_presence_seen_avg, crossdecod_presence_sign_points, 
                          # times, title, save_figure)

#%% UNSEEN
crossdecod_presence_unseen = load_scores_gen(DECOD_TFR, 'cross-decoding_blocks_unseen_tf', average = False)

crossdecod_presence_unseen_avg = np.mean(crossdecod_presence_unseen, axis=0)
crossdecod_presence_unseen_se = np.mean(np.diag(sem(crossdecod_presence_unseen)), axis=0)

sign_points_path = f'{DATA_PATH}/stats/stats_crossdecod_unseen_generalizer_signpoints.npy'
crossdecod_presence_unseen_sign_points = np.load(sign_points_path)

title='Cross-decoding Localizer-Experimental Matrix (Unseen)'
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_generalization(crossdecod_presence_unseen_avg[500:, 500:], crossdecod_presence_unseen_sign_points[500:, 500:],
                          times, title, save_figure)

# plot_stats_generalization(crossdecod_presence_unseen_avg, crossdecod_presence_sign_points, 
#                           times, title, save_figure)

#%% plot data with dots for significant comparisons
title='Cross-decoding Localizer-Experimental'
save_figure = "{}/{}".format(FIGURES_DIR, title)

# Calcular la media y desviación estándar a lo largo de la diagonal
crossdecod_presence_unseen_avg_diag = np.diagonal(crossdecod_presence_unseen_avg)
crossdecod_presence_sign_points_diag = np.diagonal(crossdecod_presence_sign_points)
crossdecod_presence_seen_avg_diag = np.diagonal(crossdecod_presence_seen_avg)

data_to_plot = {
    'Seen': (crossdecod_presence_seen_avg_diag[384:], crossdecod_presence_seen_se, 'indigo', 'mediumpurple'),
    'Unseen': (crossdecod_presence_unseen_avg_diag[384:], crossdecod_presence_unseen_se, 'darkorchid', 'orchid')
}

fig, ax = plt.subplots()

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times[384:], avg_data, color=color, label=label)
    ax.fill_between(times[384:], (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.1)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both',labelsize=16)
ax.set_xlabel('Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
ax.set_ylabel('AUC', fontsize = 16, fontweight = 'regular', labelpad = 20.0)  # Area Under the Curve
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5, alpha = 0.2)
ax.set_ylim(0.45, 0.75)
sns.despine()

for i, point_present in enumerate(crossdecod_presence_sign_points_diag):
    if point_present:
        plot_point = 0.46
        ax.scatter(times[i], plot_point, color='indigo', alpha=0.4, s=20)

plt.tight_layout()
plt.legend(loc = 'upper left', frameon=False, fontsize=12)
plt.savefig(save_figure)    
plt.close()

#%% combine plots 

os.chdir(FIGURES_DIR)

img1 = mpimg.imread('Cross-decoding Localizer-Experimental.png')
img2 = mpimg.imread('Cross-decoding Localizer-Experimental Matrix (Seen).png')
img3 = mpimg.imread('Cross-decoding Localizer-Experimental Matrix (Unseen).png')

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
plt.subplots_adjust(wspace=0)

# Plot the images on subplots
axes[0].imshow(img1)
axes[0].axis('off')
axes[0].set_title('Decoding Target presence', x = 0.6, fontsize = 16, ha = 'center', va = 'center')
axes[0].text(0, 0.15, 'A', fontsize = 16, fontweight='bold')

axes[1].imshow(img2)
axes[1].axis('off')
axes[1].set_title('Decoding Target presence \n for seen targets', x = 0.46, y = 1.05, fontsize = 16, ha = 'center', va = 'center')
axes[1].text(0, 0.15, 'B', fontsize = 16, fontweight='bold')

axes[2].imshow(img3)
axes[2].axis('off')
axes[2].set_title('Decoding Target presence \n for unseen targets', x = 0.46, y = 1.05, fontsize = 16, ha = 'center', va = 'center')
axes[2].text(0, 0.15, 'C', fontsize = 16, fontweight='bold')

# plt.show()
plt.tight_layout(w_pad = 4)
plt.savefig('Fig. 7.svg', format = 'svg', dpi = 1200)
plt.close()

