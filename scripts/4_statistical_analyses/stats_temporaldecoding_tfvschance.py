import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import sem
from stats_functions import wilcoxon_fdr_tempdecod
from utils_functions import load_scores
import seaborn as sns
import os

#%%
FIGURES_DIR = '../figures'
DATA_PATH = '../data'
DECOD_TFR = '../data/decoding_time-frequency'

#%%
times= np.load("{}/{}".format(DATA_PATH, 'times.npy'), allow_pickle=True) 

#%% target presence 

presence_tf = load_scores(DECOD_TFR, 'temporal_decoding_time-frequency_presence_lsvm', average = False)

presence_tf_se = np.mean(sem(presence_tf), axis = 0)
presence_tf_avg = np.mean(presence_tf, axis = 0)

stats_file = 'stats_temporal_decoding_tfvschance_presence'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_decoding_tfvschance_presence_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    presence_sign_points_chance = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = presence_tf
    y = np.full((29, 1024), 0.5)
    reject, p_corrected, presence_sign_points_chance = wilcoxon_fdr_tempdecod(x, y, times, stats_file, stats_path = f'{DATA_PATH}/stats')

#%% subject awareness

awareness_tf = load_scores(DECOD_TFR, 'temporal_decoding_time-frequency_awareness_lsvm', average = False)

awareness_tf_se = np.mean(sem(awareness_tf), axis = 0)
awareness_tf_avg = np.mean(awareness_tf, axis = 0)

stats_file = 'stats_temporal_decoding_tfvschance_awareness'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_decoding_tfvschance_awareness_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    awareness_sign_points_chance = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = awareness_tf
    y = np.full((29, 1024), 0.5)
    reject, p_corrected, awareness_sign_points_chance = wilcoxon_fdr_tempdecod(x, y, times, stats_file, stats_path = f'{DATA_PATH}/stats')

#%% gabor tilt

tilt_tf = load_scores(DECOD_TFR, 'temporal_decoding_time-frequency_tilt_lsvm', average = False)

tilt_tf_se = np.mean(sem(tilt_tf), axis = 0)
tilt_tf_avg = np.mean(tilt_tf, axis = 0)

stats_file = 'stats_temporal_decoding_tfvschance_tilt'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_decoding_tfvschance_tilt_signpoints.npy'

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

#%% plot data with dots for significant comparisons

title='Fig. 2.png'
# save_figure = f"{}/{}".format(FIGURES_DIR, title)
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)

data_to_plot = {
    'Target presence': (presence_tf_avg[384:], presence_tf_se, 'navy', 'lavender'),
    'Awareness': (awareness_tf_avg[384:], awareness_tf_se, 'red', 'lightcoral'),
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

for i in presence_sign_points_chance: 
    plot_points = [0.46]*1024
    ax.scatter(times[i], plot_points[i], color='navy', alpha=0.4, s = 20)
    
for j in awareness_sign_points_chance:
    plot_points = [0.455]*1024
    ax.scatter(times[j], plot_points[j], color='red', alpha=0.4, s = 20)
    
for k in tilt_sign_points_chance:
    plot_points = [0.47]*1024
    ax.scatter(times[k], plot_points[k], color='coral', alpha=0.4, s = 0.1)

plt.tight_layout()
plt.legend(loc = 'upper left', frameon=False, fontsize=12)
plt.title('Time-frequency data')
plt.savefig(f'{FIGURES_DIR}/Fig. 3B.svg', format = 'svg', dpi = 1200)
plt.savefig(f'{FIGURES_DIR}/Fig. 3B.png', format = 'png', dpi = 1200)
plt.close() 