from plotting_functions import plot_stats_generalization
from stats_functions import wilcoxon_fdr_generalization
import numpy as np
from utils_functions import load_scores_gen
import os
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import sem
import scipy

#%%
FIGURES_DIR = '../figures'
DATA_PATH = '../data'
DECOD_TFR = '../data/decoding_time-frequency'

#%% 
times= np.load("{}/{}".format(DATA_PATH, 'times.npy'), allow_pickle=True) 

#%% Target presence
presence_blocksgen = load_scores_gen(DECOD_TFR, 'blocks_generalization_presence_tf', average = False)

presence_blocksgen_avg = np.mean(presence_blocksgen, axis=0)
presence_blocksgen_se = np.mean(np.diag(sem(presence_blocksgen)), axis=0)

stats_file = 'stats_blocks_generalization_tf_presence'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_blocks_generalization_tf_presence_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    presence_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, presence_sign_points = wilcoxon_fdr_generalization(presence_blocksgen, times, stats_file)

# plot_stats_generalization(presence_blocksgen_avg[500:, 500:], presence_sign_points[500:, 500:], 
#                           times, title, save_figure)

# plot data with dots for significant comparisons

matplotlib.style.use(matplotlib.get_data_path()+'/stylelib/apa.mplstyle')

title='Cross-decoding Localizer-Experimental'
save_figure = "{}/{}".format(FIGURES_DIR, title)
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)

# Calcular la media y desviación estándar a lo largo de la diagonal
presence_blocksgen_avg_diag = np.diagonal(presence_blocksgen_avg)
presence_sign_points_diag = np.diagonal(presence_sign_points)

data_to_plot = {
    'Target presence': (presence_blocksgen_avg_diag, presence_blocksgen_se, 'navy', 'lavender')
}

fig, ax = plt.subplots()

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times, avg_data, color=color, label=label)
    ax.fill_between(times, (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.5)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both',length=0)
ax.set_xlabel('Time (s)')
ax.set_ylabel('AUC')  # Area Under the Curve
ax.legend(loc='upper left', fontsize='small', fancybox=True)
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5)
ax.set_ylim(0.45, 0.75)

for i, point_present in enumerate(presence_sign_points_diag):
    if point_present:
        plot_point = 0.46
        ax.scatter(times[i], plot_point, color='navy', alpha=0.4, s=20)

plt.savefig(save_figure)
plt.close()

#%% Target presence
presence_blocksgen = load_scores_gen(DECOD_TFR, 'blocks_generalization_2_presence_tf', average = False)

presence_blocksgen_avg = np.mean(presence_blocksgen, axis=0)
presence_blocksgen_se = np.mean(np.diag(sem(presence_blocksgen)), axis=0)

stats_file = 'stats_blocks_generalization_tf_presence_2'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_blocks_generalization_tf_presence_2_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    presence_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, presence_sign_points = wilcoxon_fdr_generalization(presence_blocksgen, times, stats_file)

# plot_stats_generalization(presence_blocksgen_avg[500:, 500:], presence_sign_points[500:, 500:], 
#                           times, title, save_figure)

# plot data with dots for significant comparisons

matplotlib.style.use(matplotlib.get_data_path()+'/stylelib/apa.mplstyle')

title='Cross-decoding Experimental-Localizer'
save_figure = "{}/{}".format(FIGURES_DIR, title)
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)

# Calcular la media y desviación estándar a lo largo de la diagonal
presence_blocksgen_avg_diag = np.diagonal(presence_blocksgen_avg)
presence_sign_points_diag = np.diagonal(presence_sign_points)

data_to_plot = {
    'Target presence': (presence_blocksgen_avg_diag, presence_blocksgen_se, 'navy', 'lavender')
}

fig, ax = plt.subplots()

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times, avg_data, color=color, label=label)
    ax.fill_between(times, (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.5)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both',length=0)
ax.set_xlabel('Time (s)')
ax.set_ylabel('AUC')  # Area Under the Curve
ax.legend(loc='upper left', fontsize='small', fancybox=True)
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5)
ax.set_ylim(0.45, 0.75)

for i, point_present in enumerate(presence_sign_points_diag):
    if point_present:
        plot_point = 0.46
        ax.scatter(times[i], plot_point, color='navy', alpha=0.4, s=20)

plt.savefig(save_figure)
plt.close()

#%% Gabor tilt 
tilt_blocksgen = load_scores_gen(DECOD_TFR, 'blocks_generalization_tilt_tf', average = False)

tilt_blocksgen_avg = np.mean(tilt_blocksgen, axis=0)

stats_file = 'stats_blocks_generalization_tf_tilt'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_blocks_generalization_tf_tilt_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    tilt_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, tilt_sign_points = wilcoxon_fdr_generalization(tilt_blocksgen, times, stats_file)

title='Generalization Localizer-Experimental Gabor tilt'
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_generalization(tilt_blocksgen_avg[512:, 512:], tilt_sign_points[512:, 512:], 
                          times, title, save_figure)