import os
os.chdir('../scripts')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.stats import sem
from stats_functions import wilcoxon_fdr_tempdecod
from utils_functions import load_scores
import seaborn as sns

#%%
FIGURES_DIR = '../figures'
DATA_PATH = '../data'
DECOD_TFR = '../data/decoding_time-frequency'

#%%
times= np.load("{}/{}".format(DATA_PATH, 'times.npy'), allow_pickle=True) 

#%% target presence 

# posterior channels

presence_postch = load_scores(DECOD_TFR, 'temporal_decoding_tf_presence_post-ch', average = False)

presence_postch_se = np.mean(sem(presence_postch), axis = 0)
presence_postch_avg = np.mean(presence_postch, axis = 0)

stats_file = 'stats_tempdecod_tfvschance_presence_postch'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_tempdecod_tfvschance_presence_postch_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    presence_postch_sign_points_chance = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = presence_postch
    y = np.full((29, 1024), 0.5)
    reject, p_corrected, presence_postch_sign_points_chance = wilcoxon_fdr_tempdecod(x, y, times, stats_file, stats_path = f'{DATA_PATH}/stats')

# target presence anterior channels

presence_antch = load_scores(DECOD_TFR, 'temporal_decoding_tf_presence_ant-ch', average = False)

presence_antch_se = np.mean(sem(presence_antch), axis = 0)
presence_antch_avg = np.mean(presence_antch, axis = 0)

stats_file = 'stats_tempdecod_tfvschance_presence_antch'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_tempdecod_tfvschance_presence_antch_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    presence_antch_sign_points_chance = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = presence_antch
    y = np.full((29, 1024), 0.5)
    reject, p_corrected, presence_antch_sign_points_chance = wilcoxon_fdr_tempdecod(x, y, times, stats_file, stats_path = f'{DATA_PATH}/stats')


# plot data with dots for significant comparisons

title='Target presence (anterior electrodes).svg'
save_figure = "{}/{}".format(FIGURES_DIR, title)
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)

data_to_plot = {
    'Target presence': (presence_antch_avg[384:], presence_antch_se, 'navy', 'lavender'),
}

fig, ax = plt.subplots(figsize = (8, 6))

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times[384:], avg_data, color=color, label=label)
    ax.fill_between(times[384:], (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.2)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both',labelsize=16)
ax.set_xlabel('Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
ax.set_ylabel('AUC', fontsize = 16, fontweight = 'regular', labelpad = 20.0)  # Area Under the Curve
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5, alpha = 0.2)
sns.despine()
ax.set_ylim(0.45, 0.75)

for j in presence_antch_sign_points_chance:
    plot_points = [0.455]*1024
    ax.scatter(times[j], plot_points[j], color='navy', alpha=0.4, s = 20)

plt.tight_layout()
plt.legend(loc = 'upper left', frameon=False, fontsize=12)
plt.savefig(save_figure)
plt.close() 

# plot data with dots for significant comparisons

title='Target presence (posterior electrodes).svg'
save_figure = "{}/{}".format(FIGURES_DIR, title)
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)

data_to_plot = {
    'Target presence': (presence_postch_avg[384:], presence_postch_se, 'navy', 'lavender'),
}

fig, ax = plt.subplots(figsize = (8, 6))

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times[384:], avg_data, color=color, label=label)
    ax.fill_between(times[384:], (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.2)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both',labelsize=16)
ax.set_xlabel('Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
ax.set_ylabel('AUC', fontsize = 16, fontweight = 'regular', labelpad = 20.0)  # Area Under the Curve
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5, alpha = 0.2)
sns.despine()
ax.set_ylim(0.45, 0.75)

for i in presence_postch_sign_points_chance: 
    plot_points = [0.46]*1024
    ax.scatter(times[i], plot_points[i], color='navy', alpha=0.4, s = 20)

plt.tight_layout()
plt.legend(loc = 'upper left', frameon=False, fontsize=12)
plt.savefig(save_figure)
plt.close() 

#%% awareness

# posterior channels

awareness_postch = load_scores(DECOD_TFR, 'temporal_decoding_tf_awareness_post-ch', average = False)

awareness_postch_se = np.mean(sem(awareness_postch), axis = 0)
awareness_postch_avg = np.mean(awareness_postch, axis = 0)

stats_file = 'stats_tempdecod_tfvschance_awareness_postch'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_tempdecod_tfvschance_awareness_postch_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    awareness_postch_sign_points_chance = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = awareness_postch
    y = np.full((29, 1024), 0.5)
    reject, p_corrected, awareness_postch_sign_points_chance = wilcoxon_fdr_tempdecod(x, y, times, stats_file, stats_path = f'{DATA_PATH}/stats')

# target awareness anterior channels

awareness_antch = load_scores(DECOD_TFR, 'temporal_decoding_tf_awareness_ant-ch', average = False)

awareness_antch_se = np.mean(sem(awareness_antch), axis = 0)
awareness_antch_avg = np.mean(awareness_antch, axis = 0)

stats_file = 'stats_tempdecod_tfvschance_awareness_antch'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_tempdecod_tfvschance_awareness_antch_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    awareness_antch_sign_points_chance = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = awareness_antch
    y = np.full((29, 1024), 0.5)
    reject, p_corrected, awareness_antch_sign_points_chance = wilcoxon_fdr_tempdecod(x, y, times, stats_file, stats_path = f'{DATA_PATH}/stats')


# plot data with dots for significant comparisons

title='Awareness (anterior electrodes).svg'
save_figure = "{}/{}".format(FIGURES_DIR, title)
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)

data_to_plot = {
    'Awareness': (awareness_antch_avg[384:], awareness_antch_se, 'red', 'lightcoral'),
}

fig, ax = plt.subplots(figsize = (8, 6))

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times[384:], avg_data, color=color, label=label)
    ax.fill_between(times[384:], (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.2)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both',labelsize=16)
ax.set_xlabel('Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
ax.set_ylabel('AUC', fontsize = 16, fontweight = 'regular', labelpad = 20.0)  # Area Under the Curve
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5, alpha = 0.2)
sns.despine()
ax.set_ylim(0.45, 0.75)

for j in awareness_antch_sign_points_chance:
    plot_points = [0.455]*1024
    ax.scatter(times[j], plot_points[j], color='red', alpha=0.4, s = 20)

plt.tight_layout()
plt.legend(loc = 'upper left', frameon=False, fontsize=12)
plt.savefig(save_figure)
plt.close() 

# plot data with dots for significant comparisons

title='Awareness (posterior electrodes).svg'
save_figure = "{}/{}".format(FIGURES_DIR, title)
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)

data_to_plot = {
    'Awareness': (awareness_postch_avg[384:], awareness_postch_se, 'red', 'lightcoral'),
}

fig, ax = plt.subplots(figsize = (8, 6))

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times[384:], avg_data, color=color, label=label)
    ax.fill_between(times[384:], (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.2)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both',labelsize=16)
ax.set_xlabel('Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
ax.set_ylabel('AUC', fontsize = 16, fontweight = 'regular', labelpad = 20.0)  # Area Under the Curve
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5, alpha = 0.2)
sns.despine()
ax.set_ylim(0.45, 0.75)

for i in awareness_postch_sign_points_chance: 
    plot_points = [0.46]*1024
    ax.scatter(times[i], plot_points[i], color='red', alpha=0.4, s = 20)

plt.tight_layout()
plt.legend(loc = 'upper left', frameon=False, fontsize=12)
plt.savefig(save_figure)
plt.close() 

#%% gabor tilt

# posterior channels

tilt_postch = load_scores(DECOD_TFR, 'temporal_decoding_tf_tilt_post-ch', average = False)

tilt_postch_se = np.mean(sem(tilt_postch), axis = 0)
tilt_postch_avg = np.mean(tilt_postch, axis = 0)

stats_file = 'stats_tempdecod_tfvschance_tilt_postch'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_tempdecod_tfvschance_tilt_postch_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    tilt_postch_sign_points_chance = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = tilt_postch
    y = np.full((29, 1024), 0.5)
    reject, p_corrected, tilt_postch_sign_points_chance = wilcoxon_fdr_tempdecod(x, y, times, stats_file, stats_path = f'{DATA_PATH}/stats')

# target tilt anterior channels

tilt_antch = load_scores(DECOD_TFR, 'temporal_decoding_tf_tilt_ant-ch', average = False)

tilt_antch_se = np.mean(sem(tilt_antch), axis = 0)
tilt_antch_avg = np.mean(tilt_antch, axis = 0)

stats_file = 'stats_tempdecod_tfvschance_tilt_antch'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_tempdecod_tfvschance_tilt_antch_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    tilt_antch_sign_points_chance = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = tilt_antch
    y = np.full((29, 1024), 0.5)
    reject, p_corrected, tilt_antch_sign_points_chance = wilcoxon_fdr_tempdecod(x, y, times, stats_file, stats_path = f'{DATA_PATH}/stats')


# plot data with dots for significant comparisons

title='Tilt (anterior electrodes).svg'
save_figure = "{}/{}".format(FIGURES_DIR, title)
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)

data_to_plot = {
    'Gabor tilt': (tilt_antch_avg[384:], tilt_antch_se, 'green', 'darkseagreen'),
}

fig, ax = plt.subplots(figsize = (8, 6))

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times[384:], avg_data, color=color, label=label)
    ax.fill_between(times[384:], (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.2)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both',labelsize=16)
ax.set_xlabel('Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
ax.set_ylabel('AUC', fontsize = 16, fontweight = 'regular', labelpad = 20.0)  # Area Under the Curve
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5, alpha = 0.2)
sns.despine()
ax.set_ylim(0.45, 0.75)

for j in tilt_antch_sign_points_chance:
    plot_points = [0.455]*1024
    ax.scatter(times[j], plot_points[j], color='green', alpha=0.4, s = 20)

plt.tight_layout()
plt.legend(loc = 'upper left', frameon=False, fontsize=12)
plt.savefig(save_figure)
plt.close() 

# plot data with dots for significant comparisons

title='Tilt (posterior electrodes).svg'
save_figure = "{}/{}".format(FIGURES_DIR, title)
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)

data_to_plot = {
    'Gabor tilt': (tilt_postch_avg[384:], tilt_postch_se, 'green', 'darkseagreen'),
}

fig, ax = plt.subplots(figsize = (8, 6))

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times[384:], avg_data, color=color, label=label)
    ax.fill_between(times[384:], (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.2)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both',labelsize=16)
ax.set_xlabel('Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
ax.set_ylabel('AUC', fontsize = 16, fontweight = 'regular', labelpad = 20.0)  # Area Under the Curve
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5, alpha = 0.2)
sns.despine()
ax.set_ylim(0.45, 0.75)

for i in tilt_postch_sign_points_chance: 
    plot_points = [0.46]*1024
    ax.scatter(times[i], plot_points[i], color='green', alpha=0.4, s = 20)

plt.tight_layout()
plt.legend(loc = 'upper left', frameon=False, fontsize=12)
plt.savefig(save_figure)
plt.close() 

#%% combine plots
    
os.chdir('C:/Users/pablr/Desktop/prodriguez_eeg_decoding/figures')

img1 = mpimg.imread('Target presence (anterior electrodes).png')
img2 = mpimg.imread('Awareness (anterior electrodes).png')
img3 = mpimg.imread('Tilt (anterior electrodes).png')
img4 = mpimg.imread('Target presence (posterior electrodes).png')
img5 = mpimg.imread('Awareness (posterior electrodes).png')
img6 = mpimg.imread('Tilt (posterior electrodes).png')

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
plt.subplots_adjust(wspace=0, hspace=0)

# Plot the images on subplots
axes[0, 0].imshow(img1)
axes[0, 0].axis('off')
axes[0, 0].set_title('Target presence', x = 0.55, fontsize = 16, color = 'navy')
axes[0, 0].text(0, 0.15, 'A', fontsize = 16, fontweight='bold')

axes[0, 1].imshow(img2)
axes[0, 1].axis('off')
axes[0, 1].set_title('Awareness', x = 0.55, fontsize = 16, color = 'red')
axes[0, 1].text(0, 0.15, 'B', fontsize = 16, fontweight='bold')

axes[0, 2].imshow(img3)
axes[0, 2].axis('off')
axes[0, 2].set_title('Gabor tilt', x = 0.55, fontsize = 16, color = 'green')
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

# plt.show()
plt.tight_layout(w_pad = 4)
plt.savefig('Fig. S2.svg')
plt.close()

