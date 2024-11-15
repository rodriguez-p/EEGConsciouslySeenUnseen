import numpy as np
from scipy.stats import sem
from plotting_functions import plot_stats_tfvsvoltage
from stats_functions import wilcoxon_fdr_tempdecod
from utils_functions import load_scores
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

#%%
FIGURES_DIR = '../figures'
DATA_PATH = '../data'
DECOD_TFR = '../data/decoding_time-frequency'
DECOD_TIME = '../data/decoding_time'

#%%
times= np.load("{}/{}".format(DATA_PATH, 'times.npy'), allow_pickle=True) 

#%% target presence 
presence_volt = load_scores(DECOD_TIME, 'temporal_decoding_presence_lsvm_experimental', average = False)
presence_tf = load_scores(DECOD_TFR, 'temporal_decoding_time-frequency_presence_lsvm', average = False)

presence_tf_se = np.mean(sem(presence_tf), axis = 0)
presence_tf_avg = np.mean(presence_tf, axis = 0)

presence_volt_se = np.mean(sem(presence_volt), axis = 0)
presence_volt_avg = np.mean(presence_volt, axis = 0)

stats_file = 'stats_temporal_decoding_tfvsvolt_presence'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_decoding_tfvsvolt_presence_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    presence_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = presence_tf
    y = presence_volt
    reject, p_corrected, presence_sign_points = wilcoxon_fdr_tempdecod(x, y, times, stats_file)

#%% plot data with dots for significant comparisons

title='Target presence - TF power vs voltage.png'
# save_figure = "{}/{}".format(FIGURES_DIR, title)

# plot the whole epoch (-2, 2)
# plot_stats_tfvsvoltage(presence_tf_avg, presence_volt_avg, presence_tf_se, presence_volt_se, 
#                         'navy', 'royalblue', 'lavender', 'lightsteelblue', 
#                         times, presence_sign_points, 0.46, title, save_figure)

# this plots from (-0.5, 2)
data_to_plot = {
    'TF Power': (presence_tf_avg[384:], presence_tf_se, 'navy', 'lavender'),
    'Voltage': (presence_volt_avg[384:], presence_volt_se, 'royalblue', 'lightsteelblue'),
}

fig, ax = plt.subplots()

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times[384:], avg_data, color=color, label=label)
    ax.fill_between(times[384:], (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.5)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both', labelsize=16)
ax.set_xlabel('Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
ax.set_ylabel('AUC', fontsize = 16, fontweight = 'regular', labelpad = 20.0)  # Area Under the Curve
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5, alpha = 0.2)
sns.despine()
ax.set_ylim(0.45, 0.75)

for i in presence_sign_points:
    plot_points = [0.46]*1024
    ax.scatter(times[i], plot_points[i], color='navy', alpha=0.4, s = 20)

# fig.set_size_inches(12, 10)
plt.tight_layout()
plt.legend(loc = 'upper left', frameon=False, fontsize=12)
plt.savefig(f'{FIGURES_DIR}/Target presence - TF power vs voltage.png', dpi = 1200)
plt.close()   
    
#%% subject awareness 
awareness_volt = load_scores(DECOD_TIME, 'temporal_decoding_awareness_lsvm_experimental', average = False)
awareness_tf = load_scores(DECOD_TFR, 'temporal_decoding_time-frequency_awareness_lsvm', average = False)

awareness_tf_se = np.mean(sem(awareness_tf), axis = 0)
awareness_tf_avg = np.mean(awareness_tf, axis = 0)

awareness_volt_se = np.mean(sem(awareness_volt), axis = 0)
awareness_volt_avg = np.mean(awareness_volt, axis = 0)

stats_file = 'stats_temporal_decoding_tfvsvolt_awareness'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_decoding_tfvsvolt_awareness_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    awareness_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = awareness_tf
    y = awareness_volt
    reject, p_corrected, awareness_sign_points = wilcoxon_fdr_tempdecod(x, y, times, stats_file)


#%% plot data with dots for significant comparisons
title='Awareness - TF power vs voltage.png'
save_figure = "{}/{}".format(FIGURES_DIR, title)

# plot the whole epoch (-2, 2)
# plot_stats_tfvsvoltage(awareness_tf_avg, awareness_volt_avg, awareness_tf_se, awareness_volt_se, 
#                         'coral', 'red', 'bisque', 'lightcoral', 
#                         times, awareness_sign_points, 0.46, title, save_figure)

# this plots from (-0.5, 2)
data_to_plot = {
    'TF Power': (awareness_tf_avg[384:], awareness_tf_se, 'red', 'lightcoral'),
    'Voltage': (awareness_volt_avg[384:], awareness_volt_se, 'coral', 'bisque'),
}

fig, ax = plt.subplots()

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

for i in awareness_sign_points:
    plot_points = [0.46]*1024
    ax.scatter(times[i], plot_points[i], color='red', alpha=0.4, s = 20)

# fig.set_size_inches(12, 10)
plt.tight_layout()
plt.legend(loc = 'upper left', frameon=False, fontsize=12)
plt.savefig(f'{FIGURES_DIR}/Awareness - TF power vs voltage.png', dpi = 1200)
plt.close() 

#%% gabor tilt 
tilt_volt = load_scores(DECOD_TIME, 'temporal_decoding_tilt_lsvm_experimental', average = False)
tilt_tf = load_scores(DECOD_TFR, 'temporal_decoding_time-frequency_tilt_lsvm', average = False)

tilt_tf_se = np.mean(sem(tilt_tf), axis = 0)
tilt_tf_avg = np.mean(tilt_tf, axis = 0)

tilt_volt_se = np.mean(sem(tilt_volt), axis = 0)
tilt_volt_avg = np.mean(tilt_volt, axis = 0)

stats_file = 'stats_temporal_decoding_tfvsvolt_tilt'

# check if the analyses have been already run and load stats files
sign_points_path = f'{DATA_PATH}/stats/stats_temporal_decoding_tfvsvolt_tilt_signpoints.npy'

if os.path.exists(sign_points_path):
    print("The file exists!")
    tilt_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    x = tilt_tf
    y = tilt_volt
    reject, p_corrected, tilt_sign_points = wilcoxon_fdr_tempdecod(x, y, times, stats_file)

#%% plot data with dots for significant comparisons
title='Gabor tilt - TF power vs voltage.png'
save_figure = "{}/{}".format(FIGURES_DIR, title)

# plot the whole epoch (-2, 2)
# plot_stats_tfvsvoltage(tilt_tf_avg, tilt_volt_avg, tilt_tf_se, tilt_volt_se, 
#                         'green', 'olivedrab', 'darkseagreen', 'lightgreen', 
#                         times, tilt_sign_points, 0.46, title, save_figure)

# this plots from (-0.5, 2)
data_to_plot = {
    'TF Power': (tilt_tf_avg[384:], tilt_tf_se, 'green', 'darkseagreen'),
    'Voltage': (tilt_volt_avg[384:], tilt_volt_se, 'olivedrab', 'lightgreen'),
}

fig, ax = plt.subplots()

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times[384:], avg_data, color=color, label=label)
    ax.fill_between(times[384:], (avg_data + se_data), (avg_data - se_data), color=color_fill, alpha=0.5)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both', labelsize=16)
ax.set_xlabel('Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
ax.set_ylabel('AUC', fontsize = 16, fontweight = 'regular', labelpad = 20.0)  # Area Under the Curve
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5, alpha = 0.2)
sns.despine()
ax.set_ylim(0.45, 0.75)

for i in tilt_sign_points:
    plot_points = [0.46]*1024
    ax.scatter(times[i], plot_points[i], color='green', alpha=0.4, s = 20)
    
# fig.set_size_inches(12, 10)
plt.tight_layout()
plt.legend(loc = 'upper left', frameon=False, fontsize=12)
plt.savefig(f'{FIGURES_DIR}/Gabor tilt - TF power vs voltage.png', dpi = 1200)
plt.close() 

#%%
os.chdir(FIGURES_DIR)

# Load the images
img1 = mpimg.imread('Target presence - TF power vs voltage.png')
img2 = mpimg.imread('Awareness - TF power vs voltage.png')
img3 = mpimg.imread('Gabor tilt - TF power vs voltage.png')

# Create subplots

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
plt.subplots_adjust(wspace=0)

# Plot the images on subplots
axes[0].imshow(img1)
axes[0].axis('off')
axes[0].set_title('Target presence', x = 0.55, fontsize = 16, color = 'navy')
axes[0].text(0, 0.15, 'A', fontsize = 16, fontweight='bold')

axes[1].imshow(img2)
axes[1].axis('off')
axes[1].set_title('Awareness', x = 0.55, fontsize = 16, color = 'red')
axes[1].text(0, 0.15, 'B', fontsize = 16, fontweight='bold')

axes[2].imshow(img3)
axes[2].axis('off')
axes[2].set_title('Gabor tilt', x = 0.55, fontsize = 16, color = 'green')
axes[2].text(0, 0.15, 'C', fontsize = 16, fontweight='bold')

plt.show()
plt.tight_layout(w_pad = 4)
plt.savefig(f'{FIGURES_DIR}/Fig. 4.svg', format = 'svg', dpi = 1200)
plt.close()
