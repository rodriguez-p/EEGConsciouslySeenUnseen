import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from stats_functions import wilcoxon_fdr_tempdecod
from utils_functions import load_scores
import seaborn as sns
from scipy import signal
import os
import scipy

#%%
FIGURES_DIR = '../figures'
DATA_PATH = '../data'
DECOD_VOLT = '../data/temporal_decoding'

#%%
times = np.load("{}/{}".format(DATA_PATH, 'times.npy'), allow_pickle=True) 

#%% target presence 

presence_volt = load_scores(DECOD_VOLT, 'temporal_decoding_presence_lsvm_experimental', average = False)

presence_volt_se = np.mean(sem(presence_volt), axis = 0)
presence_volt_avg = np.mean(presence_volt, axis = 0)

stats_file = 'stats_temporal_decoding_voltvschance_presence'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_decoding_voltvschance_presence_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    presence_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = presence_volt
    y = np.full((29, 1024), 0.5)
    reject, p_corrected, presence_sign_points = wilcoxon_fdr_tempdecod(x, y, times, stats_file, stats_path = f'{DATA_PATH}/stats')
    
#%% subject awareness

awareness_volt = load_scores(DECOD_VOLT, 'temporal_decoding_awareness_lsvm_experimental', average = False)

awareness_volt_se = np.mean(sem(awareness_volt), axis = 0)
awareness_volt_avg = np.mean(awareness_volt, axis = 0)

stats_file = 'stats_temporal_decoding_voltvschance_awareness'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_decoding_voltvschance_awareness_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    awareness_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = awareness_volt
    y = np.full((29, 1024), 0.5)
    reject, p_corrected, awareness_sign_points = wilcoxon_fdr_tempdecod(x, y, times, stats_file, stats_path = f'{DATA_PATH}/stats')

#%% gabor tilt

tilt_volt = load_scores(DECOD_VOLT, 'temporal_decoding_tilt_lsvm_experimental', average = False)

tilt_volt_se = np.mean(sem(tilt_volt), axis = 0)
tilt_volt_avg = np.mean(tilt_volt, axis = 0)

stats_file = 'stats_temporal_decoding_voltvschance_tilt'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_decoding_voltvschance_tilt_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    tilt_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = tilt_volt
    y = np.full((29, 1024), 0.5)
    reject, p_corrected, tilt_sign_points = wilcoxon_fdr_tempdecod(x, y, times, stats_file, stats_path = f'{DATA_PATH}/stats')

#%% plot data with dots for significant comparisons

presence_sign_points = [i for i in presence_sign_points if times[i] > 0]
awareness_sign_points = [j for j in awareness_sign_points if times[j] > 0]
tilt_sign_points = [k for k in tilt_sign_points if times[k] > 0]

title='Temporal decoding voltage vs chance'
save_figure = "{}/{}".format(FIGURES_DIR, title)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

data_to_plot = {
    'Target presence': (presence_volt_avg[384:], presence_volt_se, 'navy', 'lavender'),
    'Awareness': (awareness_volt_avg[384:], awareness_volt_se, 'red', 'lightcoral'),
    'Gabor tilt': (tilt_volt_avg[384:], tilt_volt_se, 'green', 'darkseagreen')
}

fig, ax = plt.subplots(figsize = (8, 6))

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times[384:], signal.savgol_filter(avg_data,9,3), color=color, label=label)
    ax.fill_between(times[384:], (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.5)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both',labelsize=16)
ax.set_xlabel('Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
ax.set_ylabel('AUC', fontsize = 16, fontweight = 'regular', labelpad = 20.0)  # Area Under the Curve
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5, alpha = 0.2)
sns.despine()
ax.set_ylim(0.45, 0.75)
# ax.set_xlim(-0.5, 2)

for i in presence_sign_points: 
    plot_points = [0.46]*1024
    ax.scatter(times[i], plot_points[i], color='navy', alpha=0.4, s = 20)
    
for j in awareness_sign_points:
    plot_points = [0.455]*1024
    ax.scatter(times[j], plot_points[j], color='red', alpha=0.4, s = 20)
    
for k in tilt_sign_points:
    plot_points = [0.47]*1024
    ax.scatter(times[k], plot_points[k], color='coral', alpha=0.4, s = 0.1)

plt.tight_layout()
plt.legend(loc = 'upper left', frameon=False, fontsize=12)
plt.title('Voltage data')
plt.savefig(f'{FIGURES_DIR}/Fig. 3A.svg', format = 'svg', dpi = 1200)
plt.savefig(f'{FIGURES_DIR}/Fig. 3A.png', format = 'png', dpi = 1200)
plt.close()