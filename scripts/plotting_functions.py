import mne
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import seaborn as sns
from scipy.ndimage import label
from mne import create_info
from mne.time_frequency import AverageTFR
import matplotlib.ticker as tkr

"""
Plotting functions. It stores functions to plot temporal decoding (for both TF power and voltage data),
temporal generalization matrices, compare evoked activities and represent statistical comparisons
between TF power and voltage. 
"""

def plotTemporalDecoding(data, se, times, color, color_se, data_type, title, save_figure):
    """ 
    Function to plot temporal decoding scores for TF power and voltage data. It adds dots to represent
    statistical comparisons between both datasets at each time point. Does not show the plot,
    saves it to a .png file.

    Args:
        data (array): decoding scores for the TF power/voltage data. Must be an array of averaged 
        data for all subjects
        se (int): mean SE for all subjects in the sample (used to plot a shaded area)
        times (array): temporal data
        color (str): Matplotlib named color for the plotted lines
        color_se (str): Matplotlib named color for the shaded areas (SE)
        data_type (str): must be either 'volt' or 'tf'
        title (str): plot title
        save_figure (str): directory where we want to save the plot (.png file)
    """
    fig, ax = plt.subplots()
    if data_type == 'volt':
        ax.plot(times, signal.savgol_filter(data,9,3), color=color, label='Decoding scores')
    else:
        ax.plot(times, data, color=color, label='Decoding scores')
    ax.fill_between(times, (data + se), (data - se), color = color_se, alpha=0.5)
    ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth = 0.5)
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC')  # Area Under the Curve
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-', linewidth = 0.5)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(title)
    fig.set_size_inches(15, 10)
    plt.tight_layout()
    plt.savefig(save_figure)
    plt.close()   

def plotCompareEvokeds(data, picks, color_dict, axes, combine, title, figures_path):
    mne.viz.plot_compare_evokeds(data, # data file to plot
                                  combine=combine, # can be 'mean' for multiple channels, or None
                                  legend='upper right',
                                  picks=picks,  # channel number or 'eeg' to plot all
                                  show_sensors='lower right',
                                  cmap=color_dict, # colormap, can be conditions dictionary
                                  title=title,
                                  axes=axes) # 'topo' for topographical map, if None one figure per channel
    plt.savefig(figures_path)
    plt.close()
    
def plot_stats_tfvsvoltage(data_tf, data_volt, se_tf, se_volt, color_tf, color_volt, color_tf_se, 
                            color_volt_se, times, sign_points, x_distance, title, save_figure, cluster_threshold=100):
    """ 
    Function to plot temporal decoding scores for TF power and voltage data. It adds dots to represent
    statistical comparisons between both datasets at each time point. Does not show the plot,
    saves it to a .png file.

    Args:
        data_tf (array), data_volt (array): decoding scores for the TF power/voltage data. 
        Must be an array of averaged data for all subjects
        se_tf (int), se_volt (int): mean SE for all subjects in the sample. Used to plot
        a shaded area
        color_tf (str), color_volt (str): Matplotlib named color for the plotted lines 
        color_tf_se (str), color_volt_se (str): Matplotlib named color for the shaded areas (SE)
        times (array): temporal data
        sign_points (array): time points where the comparison is significant 
        x_distance (int): at which distance from the X-axis we want the dots to be plotted
        title (str): plot title
        save_figure (str): directory where we want to save the plot (.png file)
    """
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    fig, ax = plt.subplots()
    ax.plot(times, signal.savgol_filter(data_volt,9,3), color_volt, label='Voltage')
    ax.fill_between(times, (data_volt + se_volt), (data_volt - se_volt), color = color_volt_se, alpha=0.5)
    ax.plot(times, data_tf, color_tf, label='TF power')
    ax.fill_between(times, (data_tf + se_tf), (data_tf - se_tf), color = color_tf_se, alpha=0.5)
    ax.axhline(0.5, color='k', linestyle='dashed', label='Chance level', linewidth = 0.5)
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC')  # Area Under the Curve
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.legend(loc = 'upper left')
    ax.axvline(.0, color='k', linestyle='-', linewidth = 0.5)
    
    # Use binary morphology to label connected components (clusters) in sign_points
    labeled_clusters, num_clusters = label(sign_points > 0.001)

    # Loop through each cluster and plot contour only if the cluster size is above the threshold
    for cluster_label in range(1, num_clusters + 1):
        cluster_mask = labeled_clusters == cluster_label
        cluster_size = np.sum(cluster_mask)

        if cluster_size >= cluster_threshold:
            for i in sign_points:
                plot_points = [x_distance]*times.size
                ax.scatter(times[i], plot_points[i], color = color_tf, alpha=0.5, s = 10)
                
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_figure)
    plt.close()
    
def plot_stats_timefrequency_decoding(data_avg, sign_points, color, times, title,
                                      save_figure, freqs = np.logspace(np.log10(4), np.log10(50), 25),
                                      sfreq = 256, cluster_threshold = 100):
    """
   Function to plot decoding scores for time-frequency bins. It adds a contour (a solid black line) to
   those bins that are statistically significant from chance. Does not show the plot, saves it to a .png file.
   
   Args:
    data_avg (array): average decoding scores for the time-frequency data
    sign_points (array): time-frequency points where the comparison is significant  
    color (str): Matplotlib colormap 
    times (array): temporal data
    title (str): plot title
    save_figure (str): directory where we want to save the plot (.png file)
    freqs (array): time-frequecy bins, default is 25 logarithmically spaced (4-50 Hz)
    sfreq (int): spatial frequency of the data, default is 256.
    """
    av_tfr = AverageTFR(create_info(['freq'], sfreq), data_avg[np.newaxis, :], 
    freqs = freqs[0:], nave=0, times=times)
    
    # Use helper function to label connected components (clusters) in sign_points
    labeled_clusters, num_clusters = label(sign_points > 0.5)
    
    # Create a mask for significant clusters
    cluster_mask = np.zeros_like(sign_points)
    for cluster_label in range(1, num_clusters + 1):
        cluster_size = np.sum(labeled_clusters == cluster_label)
        if cluster_size >= cluster_threshold:
            cluster_mask |= (labeled_clusters == cluster_label)
    
    # fig, ax = plt.subplots()
    
    av_tfr.plot('all', cmap=color, vmin = 0.5, vmax = np.abs(data_avg).max(), colorbar = True,
    mask = cluster_mask, mask_style = 'contour', yscale='linear')

    plt.xlabel('Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
    plt.ylabel('Frequency (Hz)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)  # Area Under the Curve
    
    sns.despine()
    plt.xticks(np.arange(-2, 2.3, 0.5))
    plt.tick_params(axis=u'both', which=u'both', labelsize=16)
    
    # plt.tight_layout()
    plt.savefig(save_figure)
    plt.close()
    
def plot_stats_generalization(data_avg, sign_points, times, title, save_figure, color='coolwarm', 
                              cluster_threshold=1000):
    """
    Function to plot decoding scores for the generalization analyses.
    It adds a contour (a solid black line) to clusters of time points that are statistically significant from chance.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel('Testing Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
    ax.set_ylabel('Training Time (s)', fontsize = 16, fontweight = 'regular', labelpad = 20.0)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.8)

    # Set the extent parameter to show a little bit of the top-left and bottom quadrants
    extent = [-0.1, 2, -0.1, 2]

    # Calculate contour levels based on the full data range
    data_range = np.max(data_avg) - np.min(data_avg)
    contour_levels = [np.min(data_avg) + i * (data_range / 10) for i in range(1, 10)]

    im = ax.imshow(data_avg, interpolation='lanczos', origin='lower', cmap=color, extent=extent, vmin = 0.45, vmax = 0.62)
    
    # Use binary morphology to label connected components (clusters) in sign_points
    # labeled_clusters, num_clusters = label(sign_points > 0.001)

    # # Loop through each cluster and plot contour only if the cluster size is above the threshold
    # for cluster_label in range(1, num_clusters + 1):
    #     cluster_mask = labeled_clusters == cluster_label
    #     cluster_size = np.sum(cluster_mask)

    #     if cluster_size >= cluster_threshold:
    #         # Plot contour for significant clusters using fixed levels based on full data range
    #         ax.contour(cluster_mask, levels=contour_levels, colors='black', linewidths=0.8, extent=extent, linestyles='dashed')

    ax.contour(sign_points, levels=contour_levels, colors='black', linewidths=0.8, extent=extent, linestyles='dashed')
    
    # Add a solid black diagonal line in the top-right quadrant
    ax.plot([0, 2], [0, 2], color='black', linewidth=0.6, linestyle = 'dashed')

    sns.despine()
    cbar = plt.colorbar(im, ax=ax, format=tkr.FormatStrFormatter('%.2g'))
    cbar.set_label('AUC', fontsize = 16, labelpad = 20)
    cbar.ax.tick_params(labelsize=16)
    # ax.set_title(title, fontweight = 'regular')

    # Save the figure to a file
    # fig.set_size_inches(15, 10)
    plt.tick_params(axis=u'both', which=u'both', labelsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_figure}.png', dpi = 1200)
    plt.close() 