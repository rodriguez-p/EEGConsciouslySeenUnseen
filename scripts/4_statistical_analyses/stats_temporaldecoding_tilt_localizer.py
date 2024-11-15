import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.stats import sem
from stats_functions import wilcoxon_fdr_tempdecod
from utils_functions import load_scores
import seaborn as sns
import os

#%%
DATA_PATH = '/data/prodriguez'
DECOD_TFR = '/data/prodriguez/decoding_time-frequency'
FIGURES_DIR = '/data/prodriguez/figures'

#%%
times= np.load("{}/{}".format(DATA_PATH, 'times.npy'), allow_pickle=True) 

#%% TILT LOCALIZER

tilt_tf = load_scores(DECOD_TFR, 'temporal_decoding_tf_tilt_localizer', average = False)

tilt_tf_se = np.mean(sem(tilt_tf), axis = 0)
tilt_tf_avg = np.mean(tilt_tf, axis = 0)

stats_file = 'stats_temporal_decoding_tilt_localizer'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_decoding_tilt_localizer_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    tilt_sign_points_chance = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = tilt_tf
    y = np.full((29, 1024), 0.5)
    reject, p_corrected, tilt_sign_points_chance = wilcoxon_fdr_tempdecod(x, y, times, stats_file, stats_path = f'{DATA_PATH}/stats')

# plot data with dots for significant comparisons

title='Temporal decoding tilt localizer.png'
save_figure = "{}/{}".format(FIGURES_DIR, title)
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)

data_to_plot = {
    'Gabor tilt': (tilt_tf_avg[384:], tilt_tf_se, 'green', 'darkseagreen')
}

fig, ax = plt.subplots(figsize = (8, 6))

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times[384:], avg_data, color=color, label=label)
    ax.fill_between(times[384:], (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.5)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both',labelsize=16)
ax.set_xlabel('Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
ax.set_ylabel('AUC', fontsize = 16, fontweight = 'regular', labelpad = 20.0)  # Area Under the Curve
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5, alpha = 0.2)
sns.despine()
ax.set_ylim(0.45, 0.75)

for k in tilt_sign_points_chance:
    plot_points = [0.47]*1024
    ax.scatter(times[k], plot_points[k], color='coral', alpha=0.4, s = 0.1)

plt.tight_layout()
plt.legend(loc = 'upper left', frameon=False, fontsize=12)
plt.savefig(save_figure)
plt.close() 

#%% TILT LOCALIZER posterior electrodes

tilt_tf_postch = load_scores(DECOD_TFR, 'temporal_decoding_tf_tilt_localizer_postch', average = False)

tilt_tf_postch_se = np.mean(sem(tilt_tf_postch), axis = 0)
tilt_tf_postch_avg = np.mean(tilt_tf_postch, axis = 0)

stats_file = 'stats_temporal_decoding_tilt_localizer_postch'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_decoding_tilt_localizer_postch_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    tilt_sign_points_chance = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = tilt_tf_postch
    y = np.full((29, 1024), 0.5)
    reject, p_corrected, tilt_sign_points_chance = wilcoxon_fdr_tempdecod(x, y, times, stats_file, stats_path = f'{DATA_PATH}/stats')

# plot data with dots for significant comparisons

title='Temporal decoding tilt localizer (posterior channels).png'
save_figure = "{}/{}".format(FIGURES_DIR, title)
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)

data_to_plot = {
    'Gabor tilt': (tilt_tf_postch_avg[384:], tilt_tf_postch_se, 'green', 'darkseagreen')
}

fig, ax = plt.subplots(figsize = (8, 6))

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times[384:], avg_data, color=color, label=label)
    ax.fill_between(times[384:], (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.5)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both',labelsize=16)
ax.set_xlabel('Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
ax.set_ylabel('AUC', fontsize = 16, fontweight = 'regular', labelpad = 20.0)  # Area Under the Curve
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5, alpha = 0.2)
sns.despine()
ax.set_ylim(0.45, 0.75)

for k in tilt_sign_points_chance:
    plot_points = [0.47]*1024
    ax.scatter(times[k], plot_points[k], color='coral', alpha=0.4, s = 0.1)

plt.tight_layout()
plt.legend(loc = 'upper left', frameon=False, fontsize=12)
plt.savefig(save_figure)
plt.close() 

#%% TILT LOCALIZER anterior electrodes

tilt_tf_antch = load_scores(DECOD_TFR, 'temporal_decoding_tf_tilt_localizer_antch', average = False)

tilt_tf_antch_se = np.mean(sem(tilt_tf_antch), axis = 0)
tilt_tf_antch_avg = np.mean(tilt_tf_antch, axis = 0)

stats_file = 'stats_temporal_decoding_tilt_localizer_antch'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_decoding_tilt_localizer_antch_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    tilt_sign_points_chance = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = tilt_tf_antch
    y = np.full((29, 1024), 0.5)
    reject, p_corrected, tilt_sign_points_chance = wilcoxon_fdr_tempdecod(x, y, times, stats_file, stats_path = f'{DATA_PATH}/stats')

# plot data with dots for significant comparisons

title='Temporal decoding tilt localizer (anterior channels).png'
save_figure = "{}/{}".format(FIGURES_DIR, title)
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)

data_to_plot = {
    'Gabor tilt': (tilt_tf_antch_avg[384:], tilt_tf_antch_se, 'green', 'darkseagreen')
}

fig, ax = plt.subplots(figsize = (8, 6))

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times[384:], avg_data, color=color, label=label)
    ax.fill_between(times[384:], (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.5)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both',labelsize=16)
ax.set_xlabel('Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
ax.set_ylabel('AUC', fontsize = 16, fontweight = 'regular', labelpad = 20.0)  # Area Under the Curve
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5, alpha = 0.2)
sns.despine()
ax.set_ylim(0.45, 0.75)

for k in tilt_sign_points_chance:
    plot_points = [0.47]*1024
    ax.scatter(times[k], plot_points[k], color='coral', alpha=0.4, s = 0.1)

plt.tight_layout()
plt.legend(loc = 'upper left', frameon=False, fontsize=12)
plt.savefig(save_figure)
plt.close() 

#%% combine plots
    
os.chdir('C:/Users/pablr/Desktop/prodriguez_eeg_decoding/figures')

img1 = mpimg.imread('Temporal decoding tilt localizer.png')
img2 = mpimg.imread('Temporal decoding tilt localizer (anterior channels).png')
img3 = mpimg.imread('Temporal decoding tilt localizer (posterior channels).png')

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
plt.subplots_adjust(wspace=0, hspace=0)

# Plot the images on subplots
axes[0].imshow(img1)
axes[0].axis('off')
# axes[0].set_title('Target presence', x = 0.55, fontsize = 16, color = 'navy')
axes[0].text(0, 0.15, 'A', fontsize = 16, fontweight='bold')

axes[1].imshow(img2)
axes[1].axis('off')
# axes[1].set_title('Awareness', x = 0.55, fontsize = 16, color = 'red')
axes[1].text(0, 0.15, 'B', fontsize = 16, fontweight='bold')

axes[2].imshow(img3)
axes[2].axis('off')
# axes[2].set_title('Gabor tilt', x = 0.55, fontsize = 16, color = 'green')
axes[2].text(0, 0.15, 'C', fontsize = 16, fontweight='bold')

plt.show()
plt.tight_layout(w_pad = 4)
plt.savefig('Fig. S1.svg')
plt.close()

