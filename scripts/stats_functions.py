import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection
import xlsxwriter

subject_ids = list(range(10, 43))
wrong_subjects = [11, 24, 26, 38]
valid_subjects = np.array(list(set(subject_ids) - set(wrong_subjects)))

def wilcoxon_fdr_tempdecod(x_data, y_data, times, stats_file, stats_path, p_threshold = .001):
    """
    Loads decoding scores on time-series data (either with voltage or TF power) and performs
    a Wilcoxon signed-rank test in each time point for both datasets. The resulting p values 
    are then FDR-adjusted for multiple comparisons. Returns the corrected p values, 
    a boolean array indicating whether or not the comparison is significant and a
    temporal array with the significant time points.

    Args:
        x_data (array): the first dataset
        y_data (array): the second dataset
        times (array): temporal data
        stats_file (str): xlsx file to save adjusted p-values and significant time points
        p_threshold (int): significance threshold, default is 0.001

    Returns:
        reject: boolean array the same size as the input data indicating whether
        or not the comparison is significant
        p_corrected: arrray with the FDR-adjusted p values (also called q values)
        sign_points: array with the time points where the comparison is
        significant (used for plotting)
    """
    w_array = []  # initialize array to store w values
    p_vals_array = []  # initialize array to store p values
    x_data = np.stack(x_data)
    y_data = np.stack(y_data)

    for i, _ in enumerate(times):  # loop over the whole time-series and
        # compare x vs y in each time point
        x = x_data[:, i]
        y = y_data[:, i]
        w, p_val = wilcoxon(x, y)

        if np.any(w_array):  # store w values
            w_array = np.append(w_array, w)
        else:
            w_array = w

        if np.any(p_vals_array):  # store p values
            p_vals_array = np.append(p_vals_array, p_val)
        else:
            p_vals_array = p_val

    reject, p_corrected = fdrcorrection(
        p_vals_array, alpha=0.05, method='indep')  # FDR-adjust p values

    # save the significant time points
    sign_points = np.array([i for i, p in enumerate(p_corrected) if p < p_threshold])
    
    # save stats to xlsx file
    workbook = xlsxwriter.Workbook(f'{stats_file}.xlsx')
    worksheet = workbook.add_worksheet()
    row = 1
    worksheet.write(0, 0, 'Significant time point')
    worksheet.write(0, 1, 'Adjusted p-value')

    for t in range(0, 1024):
        worksheet.write(row, 0, reject[t])
        worksheet.write(row, 1, p_corrected[t])
        row += 1

    workbook.close()

    # save stats to npy files
    np.save(f'{stats_path}/{stats_file}_reject.npy', reject)
    np.save(f'{stats_path}/{stats_file}_p-corrected.npy', p_corrected)
    np.save(f'{stats_path}/{stats_file}_signpoints.npy', sign_points)
    
    return reject, p_corrected, sign_points

def wilcoxon_fdr_tfdecod(x_data, times, stats_file, p_threshold = .05, freqs = np.logspace(*np.log10([4, 50]), num=25), y_data = 'chance'):
    """
    Loads decoding scores on time-frequency data and performs a Wilcoxon signed-rank test in each 
    time-frequency point against chance (50%). The resulting p values are then FDR-adjusted
    for multiple comparisons. Returns the corrected p values, a boolean array indicating
    whether or not the comparison is significant and a temporal array with the
    significant time points.

    Args:
        x_data (array): the first dataset
        times (array): temporal data
        stats_file (str): xlsx file to save adjusted p-values and significant time points
        p_threshold (int): significance threshold, default is 0.001

    Returns:
        reject: boolean array the same size as the input data indicating whether
        or not the comparison is significant
        p_corrected: arrray with the FDR-adjusted p values (also called q values)
        sign_points: array with the time points where the comparison is
        significant (used for plotting)
    """
    x_data = np.reshape(x_data, (len(valid_subjects), freqs.size*times.size)) # reshape time-frequency dimensions
    
    if y_data == 'chance':
        y_data = np.full((len(valid_subjects), freqs.size*times.size), 0.5) # create chance array with the same shape
        
    else:
        y_data = np.reshape(y_data, (len(valid_subjects), freqs.size*times.size))
        
    w_array = []  # initialize array to store w values
    p_vals_array = []  # initialize array to store p values

    for i in range(0, (freqs.size*times.size)):  # loop over the whole time-frequency series and
        # compare x vs y in each time point
        x = x_data[:, i]
        y = y_data[:, i]
        w, p_val = wilcoxon(x, y)

        if np.any(w_array):  # store w values
            w_array = np.append(w_array, w)
        else:
            w_array = w

        if np.any(p_vals_array):  # store p values
            p_vals_array = np.append(p_vals_array, p_val)
        else:
            p_vals_array = p_val

    reject, p_corrected = fdrcorrection(
        p_vals_array, alpha=0.05, method='indep')  # FDR-adjust p values

    reject = np.reshape(reject, (freqs.size, times.size))
    p_corrected = np.reshape(p_corrected, (freqs.size, times.size))
    sign_points = p_corrected < p_threshold

    # sign_points = [i for i, p in enumerate(p_corrected) if p < p_threshold]
    
    # workbook = xlsxwriter.Workbook(f'{stats_file}.xlsx')
    # worksheet = workbook.add_worksheet()
    # row = 1
    # worksheet.write(0, 0, 'Significant time point')
    # worksheet.write(0, 1, 'Adjusted p-value')

    # for t in range(0, 1024):
    #     worksheet.write(row, 0, reject[t])
    #     worksheet.write(row, 1, p_corrected[t])
    #     row += 1

    # workbook.close()
    
    # save stats to npy files
    np.save(f'{stats_file}_reject.npy', reject)
    np.save(f'{stats_file}_p-corrected.npy', p_corrected)
    np.save(f'{stats_file}_signpoints.npy', sign_points)
    
    return reject, p_corrected, sign_points

def wilcoxon_fdr_generalization(x_data, times, stats_file, y_data = 'chance', p_threshold = .001, correction = True):
    x = np.stack(x_data)
    x_data = np.reshape(x, (len(x_data), times.size*times.size))

    if y_data == 'chance':
      y_data = np.full((len(x_data), times.size*times.size), 0.5)
    else:
        y = np.stack(y_data)
        y_data = np.reshape(y, (len(y_data), times.size*times.size))

    w_array = []  # initialize array to store w values
    p_vals_array = []  # initialize array to store p values
    
    for i in range(0, (times.size*times.size)):  # loop over the whole time-frequency series and
        # compare x vs y in each time point
        x = x_data[:, i]
        y = y_data[:, i]
        w, p_val = wilcoxon(x, y)
    
        if np.any(w_array):  # store w values
            w_array = np.append(w_array, w)
        else:
            w_array = w
    
        if np.any(p_vals_array):  # store p values
            p_vals_array = np.append(p_vals_array, p_val)
        else:
            p_vals_array = p_val
            
    
    reject, p_corrected = fdrcorrection(
        p_vals_array, alpha=0.05, method='indep')  # FDR-adjust p values
    
    reject = np.reshape(reject, (times.size, times.size))
    p_corrected = np.reshape(p_corrected, (times.size, times.size))
    sign_points = p_corrected < p_threshold
    # sign_points = [i for i, p in enumerate(p_corrected) if p < p_threshold]
    
    # save stats to npy files
    np.save(f'{stats_file}_reject.npy', reject)
    np.save(f'{stats_file}_p-corrected.npy', p_corrected)
    np.save(f'{stats_file}_signpoints.npy', sign_points)
    np.save(f'{stats_file}_p-unc.npy', p_vals_array)
    
    return reject, p_corrected, sign_points, p_vals_array
