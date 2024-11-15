import os
os.chdir("../scripts")

from utils_functions import load_scores, load_scores_gen
import matplotlib.image as mpimg
import seaborn as sns
from scipy.stats import sem
import matplotlib.pyplot as plt
import numpy as np
from stats_functions import wilcoxon_fdr_tempdecod, wilcoxon_fdr_generalization
from plotting_functions import plot_stats_generalization

# %%
DATA_PATH = '/data/prodriguez'
DECOD_TFR = '/data/prodriguez/decoding_time-frequency'
FIGURES_DIR = '/data/prodriguez/figures'

# %%
times = np.load("{}/{}".format(DATA_PATH, 'times.npy'), allow_pickle=True)

# %% frontal electrodes

# SEEN
crossdecod_seen_antch = load_scores(
    DECOD_TFR, 'cross-decoding_blocks_seen_sliding_ant-ch_tf', average=False)

crossdecod_seen_antch_avg = np.mean(crossdecod_seen_antch, axis=0)
crossdecod_seen_antch_se = np.mean(sem(crossdecod_seen_antch), axis=0)

stats_file = 'stats_crossdecod_blocks_seen_antch'

# check if the analyses have been already run and load stats files
sign_points_path = f"{DECOD_TFR}/stats_crossdecod_blocks_seen_antch_signpoints.npy"

if os.path.exists(sign_points_path):
    print("The file exists!")
    crossdecod_seen_antch_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    x = crossdecod_seen_antch
    y = np.full((29, 1024), 0.5)
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, crossdecod_seen_antch_sign_points = wilcoxon_fdr_tempdecod(
        x, y, times=times, stats_file=stats_file, stats_path=f"{DECOD_TFR}")

# UNSEEN
crossdecod_unseen_antch = load_scores(
    DECOD_TFR, 'cross-decoding_blocks_unseen_sliding_ant-ch_tf', average=False)

crossdecod_unseen_antch_avg = np.mean(crossdecod_unseen_antch, axis=0)
crossdecod_unseen_antch_se = np.mean(sem(crossdecod_unseen_antch), axis=0)

stats_file = 'stats_crossdecod_blocks_unseen_antch'

# check if the analyses have been already run and load stats files
sign_points_path = "f{DECOD_TFR}/stats_crossdecod_blocks_unseen_antch_signpoints.npy"

if os.path.exists(sign_points_path):
    print("The file exists!")
    crossdecod_unseen_antch_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    x = crossdecod_unseen_antch
    y = np.full((29, 1024), 0.5)
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, crossdecod_unseen_antch_sign_points = wilcoxon_fdr_tempdecod(
        x, y, times, stats_file, stats_path=f"{DECOD_TFR}")

# %% plot against chance

data_to_plot = {
    'Seen': (crossdecod_seen_antch_avg[384:], crossdecod_seen_antch_se, 'indigo', 'mediumpurple'),
    'Unseen': (crossdecod_unseen_antch_avg[384:], crossdecod_unseen_antch_se, 'darkorchid', 'orchid')
}

fig, ax = plt.subplots()

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times[384:], avg_data, color=color, label=label)
    ax.fill_between(times[384:], (avg_data + se_data),
                    (avg_data - se_data), color=color_fill, alpha=0.1)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both', labelsize=16)
ax.set_xlabel('Time (s)', fontsize=16, fontweight='regular', labelpad=20.0)
ax.set_ylabel('AUC', fontsize=16, fontweight='regular',
              labelpad=20.0)  # Area Under the Curve
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5, alpha=0.2)
ax.set_ylim(0.45, 0.75)
sns.despine()

for i in crossdecod_seen_antch_sign_points:
    plot_points = [0.46]*1024
    ax.scatter(times[i], plot_points[i], color='indigo', alpha=0.6, s=20)

for i in crossdecod_unseen_antch_sign_points:
    plot_points = [0.47]*1024
    ax.scatter(times[i], plot_points[i], color='darkorchid', alpha=0.6, s=20)

plt.tight_layout()
plt.legend(loc='upper left', frameon=False, fontsize=12)
plt.savefig(f'{FIGURES_DIR}/crossdecod_antch')
plt.close()

# %% occipital electrodes

# SEEN
crossdecod_seen_postch = load_scores(
    DECOD_TFR, 'cross-decoding_blocks_seen_sliding_post-ch_tf', average=False)

crossdecod_seen_postch_avg = np.mean(crossdecod_seen_postch, axis=0)
crossdecod_seen_postch_se = np.mean(sem(crossdecod_seen_postch), axis=0)

stats_file = 'stats_crossdecod_blocks_seen_postch'

# check if the analyses have been already run and load stats files
sign_points_path ="f{DECOD_TFR}/stats_crossdecod_blocks_seen_postch_signpoints.npy"

if os.path.exists(sign_points_path):
    print("The file exists!")
    crossdecod_seen_postch_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    x = crossdecod_seen_postch
    y = np.full((29, 1024), 0.5)
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, crossdecod_seen_postch_sign_points = wilcoxon_fdr_tempdecod(
        x, y, times=times, stats_file=stats_file, stats_path=f"{DECOD_TFR}")

# UNSEEN
crossdecod_unseen_postch = load_scores(
    DECOD_TFR, 'cross-decoding_blocks_unseen_sliding_post-ch_tf', average=False)

crossdecod_unseen_postch_avg = np.mean(crossdecod_unseen_postch, axis=0)
crossdecod_unseen_postch_se = np.mean(sem(crossdecod_unseen_postch), axis=0)

stats_file = 'stats_crossdecod_blocks_unseen_postch'

# check if the analyses have been already run and load stats files
sign_points_path = "f{DECOD_TFR}/stats_crossdecod_blocks_unseen_postch_signpoints.npy"

if os.path.exists(sign_points_path):
    print("The file exists!")
    crossdecod_unseen_postch_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    x = crossdecod_unseen_postch
    y = np.full((29, 1024), 0.5)
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, crossdecod_unseen_postch_sign_points = wilcoxon_fdr_tempdecod(
        x, y, times, stats_file, stats_path=f"{DECOD_TFR}")

# %% plot against chance

data_to_plot = {
    'Seen': (crossdecod_seen_postch_avg[384:], crossdecod_seen_postch_se, 'indigo', 'mediumpurple'),
    'Unseen': (crossdecod_unseen_postch_avg[384:], crossdecod_unseen_postch_se, 'darkorchid', 'orchid')
}

fig, ax = plt.subplots()

for label, (avg_data, se_data, color, color_fill) in data_to_plot.items():
    ax.plot(times[384:], avg_data, color=color, label=label)
    ax.fill_between(times[384:], (avg_data + se_data),
                    (avg_data - se_data), color=color_fill, alpha=0.1)

ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both', labelsize=16)
ax.set_xlabel('Time (s)', fontsize=16, fontweight='regular', labelpad=20.0)
ax.set_ylabel('AUC', fontsize=16, fontweight='regular',
              labelpad=20.0)  # Area Under the Curve
ax.axvline(0.0, color='k', linestyle='-', linewidth=0.5, alpha=0.2)
ax.set_ylim(0.45, 0.75)
sns.despine()

for i in crossdecod_seen_postch_sign_points:
    plot_points = [0.46]*1024
    ax.scatter(times[i], plot_points[i], color='indigo', alpha=0.6, s=20)

for i in crossdecod_unseen_postch_sign_points:
    plot_points = [0.46]*1024
    ax.scatter(times[i], plot_points[i], color='darkorchid', alpha=0.6, s=20)

plt.tight_layout()
plt.legend(loc='upper left', frameon=False, fontsize=12)
plt.savefig(f'{FIGURES_DIR}/crossdecod_postch')
plt.close()

# %% frontal electrodes TGM

# SEEN
crossdecod_seen_antch_tempgen = load_scores_gen(     
    DECOD_TFR, 'cross-decoding_blocks_seen_generalizer_ant-ch_tf', average=False)

crossdecod_seen_antch_tempgen_avg = np.mean(
    crossdecod_seen_antch_tempgen, axis=0)

stats_file = 'stats_crossdecod_blocks_seen_antch_tempgen'

# check if the analyses have been already run and load stats files
sign_points_path = "f{DECOD_TFR}/stats_crossdecod_blocks_seen_antch_tempgen_signpoints.npy"

if os.path.exists(sign_points_path):
    print("The file exists!")
    crossdecod_seen_antch_tempgen_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, crossdecod_seen_antch_tempgen_sign_points = wilcoxon_fdr_generalization(
        crossdecod_seen_antch_tempgen, times, stats_file)

title = 'crossdecod_seen_antch_tgm'
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_generalization(crossdecod_seen_antch_tempgen_avg[500:, 500:], crossdecod_seen_antch_tempgen_sign_points[500:, 500:],
                          times, title, save_figure)

# UNSEEN
crossdecod_unseen_antch_tempgen = load_scores_gen(
    DECOD_TFR, 'cross-decoding_blocks_unseen_generalizer_ant-ch_tf', average=False)

crossdecod_unseen_antch_tempgen_avg = np.mean(
    crossdecod_unseen_antch_tempgen, axis=0)

stats_file = 'stats_crossdecod_blocks_unseen_antch_tempgen'

# check if the analyses have been already run and load stats files
sign_points_path = "f{DECOD_TFR}/stats_crossdecod_blocks_unseen_antch_tempgen_signpoints.npy"

if os.path.exists(sign_points_path):
    print("The file exists!")
    crossdecod_unseen_antch_tempgen_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, crossdecod_unseen_antch_tempgen_sign_points = wilcoxon_fdr_generalization(
        crossdecod_unseen_antch_tempgen, times, stats_file)

title = 'crossdecod_unseen_antch_tgm'
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_generalization(crossdecod_unseen_antch_tempgen_avg[500:, 500:], crossdecod_unseen_antch_tempgen_sign_points[500:, 500:],
                          times, title, save_figure)

# %% occipital electrodes TGM

# SEEN
crossdecod_seen_postch_tempgen = load_scores_gen(
     DECOD_TFR, 'cross-decoding_blocks_seen_generalizer_post-ch_tf', average=False)

crossdecod_seen_postch_tempgen_avg = np.mean(
     crossdecod_seen_postch_tempgen, axis=0)

stats_file = 'stats_crossdecod_blocks_seen_postch_tempgen'

# check if the analyses have been already run and load stats files
sign_points_path = "f{DECOD_TFR}/stats_crossdecod_blocks_seen_postch_tempgen_signpoints.npy"

if os.path.exists(sign_points_path):
    print("The file exists!")
    crossdecod_seen_postch_tempgen_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, crossdecod_seen_postch_tempgen_sign_points = wilcoxon_fdr_generalization(
        crossdecod_seen_postch_tempgen, times, stats_file)

title = 'crossdecod_seen_postch_tgm'
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_generalization(crossdecod_seen_postch_tempgen_avg[500:, 500:], crossdecod_seen_postch_tempgen_sign_points[500:, 500:],
                          times, title, save_figure)

# UNSEEN
crossdecod_unseen_postch_tempgen = load_scores_gen(
    DECOD_TFR, 'cross-decoding_blocks_unseen_generalizer_post-ch_tf', average=False)

crossdecod_unseen_postch_tempgen_avg = np.mean(
    crossdecod_unseen_postch_tempgen, axis=0)

stats_file = 'stats_crossdecod_blocks_unseen_postch_tempgen'

# check if the analyses have been already run and load stats files
sign_points_path = "f{DECOD_TFR}/stats_crossdecod_blocks_unseen_postch_tempgen_signpoints.npy"

if os.path.exists(sign_points_path):
    print("The file exists!")
    crossdecod_unseen_postch_tempgen_sign_points = np.load(sign_points_path)
    print("File loaded")
else:
    print("File does not exist!")
    # Wilcoxon signed-rank test with FDR correction
    reject, p_corrected, crossdecod_unseen_postch_tempgen_sign_points = wilcoxon_fdr_generalization(
        crossdecod_unseen_postch_tempgen, times, stats_file)

title = 'crossdecod_unseen_postch_tgm'
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_generalization(crossdecod_unseen_postch_tempgen_avg[500:, 500:], crossdecod_unseen_postch_tempgen_sign_points[500:, 500:],
                          times, title, save_figure)

# frontal vs occipital TGM (Seen)

y = crossdecod_seen_antch_tempgen
x = crossdecod_seen_postch_tempgen

stats_file = 'stats_crossdecod_blocks_seen_postvsant_tempgen'

reject, p_corrected, crossdecod_seen_postvsant_tempgen_sign_points = wilcoxon_fdr_generalization(x_data = x, y_data = y, times = times, stats_file = stats_file)

crossdecod_unseen_postvsant_tempgen_sign_points = np.load('stats_crossdecod_blocks_seen_postvsant_tempgen_signpoints.npy')

title = 'crossdecod_unseen_postvsant_tgm'
save_figure = "{}/{}".format(FIGURES_DIR, title)

plot_stats_generalization(crossdecod_seen_postch_tempgen_avg[500:, 500:], crossdecod_unseen_postvsant_tempgen_sign_points[500:, 500:],
                          times, title, save_figure)


#%% combine plots
    
os.chdir('C:/Users/pablr/Desktop/prodriguez_eeg_decoding/figures')

img1 = mpimg.imread('crossdecod_antch.png')
img2 = mpimg.imread('crossdecod_seen_antch_tgm.png')
img3 = mpimg.imread('crossdecod_unseen_antch_tgm.png')

img4 = mpimg.imread('crossdecod_postch.png')
img5 = mpimg.imread('crossdecod_seen_postch_tgm.png')
img6 = mpimg.imread('crossdecod_unseen_postch_tgm.png')

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
plt.subplots_adjust(wspace=0, hspace=0)

# Plot the images on subplots
axes[0, 0].imshow(img1)
axes[0, 0].axis('off')
axes[0, 0].set_title('Decoding Target presence', x = 0.55, fontsize = 16, color = 'black')
axes[0, 0].text(0, 0.15, 'A', fontsize = 16, fontweight='bold')

axes[0, 1].imshow(img2)
axes[0, 1].axis('off')
axes[0, 1].set_title('Decoding Target presence \n for seen targets', x = 0.46, fontsize = 16, color = 'black')
axes[0, 1].text(0, 0.15, 'B', fontsize = 16, fontweight='bold')

axes[0, 2].imshow(img3)
axes[0, 2].axis('off')
axes[0, 2].set_title('Decoding Target presence \n for unseen targets', x = 0.46, fontsize = 16, color = 'black')
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
plt.savefig('Fig. S5.svg')
plt.close()