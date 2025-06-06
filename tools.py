""" This module provides a set of useful functions used for Pcross and P3D computations"""
 

import numpy as np
from astropy.table import Table, vstack
import astropy.constants
from astropy.cosmology import FlatLambdaCDM

SPEED_LIGHT = astropy.constants.c.to('km/s').value  # km/s
LAMBDA_LYA = 1215.67  # Angstrom
# Wavelength grid to estimate skyline masks:
# default values from eg. desispec/scripts/proc.py
DEFAULT_MINWAVE_SKYMASK = 3600.0
DEFAULT_MAXWAVE_SKYMASK = 9824.0
DEFAULT_DWAVE_SKYMASK = 0.8


def convert_units(data_to_convert, input_units, output_units, z, inverse_units=False):
    
    # Computing cosmo used for conversions
    Omega_m = 0.3153
    h = 0.7
    cosmo = FlatLambdaCDM(H0=100*h, Om0=Omega_m)
    
    # Defining all conversion factors
    conversion_A_kmps = SPEED_LIGHT / ((1 + z) * LAMBDA_LYA)
    conversion_kmps_A = 1 / conversion_A_kmps
    conversion_A_Mpcph = SPEED_LIGHT * h / (cosmo.H(z).value * LAMBDA_LYA)
    conversion_Mpcph_A = 1 / conversion_A_Mpcph
    conversion_kmps_Mpcph = conversion_kmps_A * conversion_A_Mpcph
    conversion_Mpcph_kmps = conversion_Mpcph_A * conversion_A_kmps

    # Conversions
    if input_units == 'Angstrom' and output_units == 'km/s':
        if inverse_units:
            converted_data = data_to_convert * conversion_kmps_A
        else:
            converted_data = data_to_convert * conversion_A_kmps
    elif input_units == 'km/s' and output_units == 'Angstrom':
        if inverse_units:
            converted_data = data_to_convert * conversion_A_kmps
        else:
            converted_data = data_to_convert * conversion_kmps_A
    elif input_units == 'Angstrom' and output_units == 'Mpc/h':
        if inverse_units:
            converted_data = data_to_convert * conversion_Mpcph_A
        else:
            converted_data = data_to_convert * conversion_A_Mpcph
    elif input_units == 'Mpc/h' and output_units == 'Angstrom':
        if inverse_units:
            converted_data = data_to_convert * conversion_A_Mpcph
        else:
            converted_data = data_to_convert * conversion_Mpcph_A
    elif input_units == 'km/s' and output_units == 'Mpc/h':
        if inverse_units:
            converted_data = data_to_convert * conversion_Mpcph_kmps
        else:
            converted_data = data_to_convert * conversion_kmps_Mpcph
    elif input_units == 'Mpc/h' and output_units == 'km/s':
        if inverse_units:
            converted_data = data_to_convert * conversion_kmps_Mpcph
        else:
            converted_data = data_to_convert * conversion_Mpcph_kmps
    else:
        converted_data = data_to_convert
    
    return converted_data


def rebin_vector(arr, pack=2, rebin_opt='mean', verbose=True):
    # Rebin 1D array. Not at all cpu-optimized
    arr = np.asarray(arr)
    if len(arr.shape)!=1 :
        print("ERROR: only 1-D array in rebin_vector.")
    if rebin_opt not in ['mean','sum'] :
        print("ERROR: wrong option in rebin_vector.")
    if verbose and (len(arr) % pack != 0):
        print("WARNING: rebin_vector: pack not adapted to size, last bin will be wrong")
    v, i = [], 0
    while i+pack<=len(arr) :
        if rebin_opt=='mean':
            v.append(np.mean([arr[i:i+pack]]))
        else:
            v.append(np.sum([arr[i:i+pack]]))
        i+=pack
    return np.asarray(v)


def cov_to_corrmat(covmatrix):
    sigmas = np.sqrt(np.diag(np.diag(covmatrix)))
    gaid = np.linalg.inv(sigmas)
    return gaid @ covmatrix @ gaid


def eliminate_outlyers(data_set, m_sigma):
    mean = np.mean(data_set)
    sigma = np.std(data_set)
    final_data_set = [x for x in data_set if (x > mean - m_sigma * sigma)]
    final_data_set = [x for x in final_data_set if (x < mean + m_sigma * sigma)]
    final_data_set = np.array(final_data_set)
    if len(final_data_set) != len(data_set):
        print(str(m_sigma)+'sigma outlyers eliminated')
    return final_data_set


def find_bin_edges(arr, mean_values_target, debug=False, for_p_cross=True):
    """ Find bin edges so that when histogramming an array, the 
    mean values in each bin are given.
    option for_px: for the purpose of LyP3D
    Output: edges, size = len(mean_values_target)+1
    """

    arr = np.array(arr)
    mean_values_target = np.sort(mean_values_target)
    if for_p_cross:
        last_value = 2*mean_values_target[-1] - mean_values_target[-2]
        mean_values_target = np.append(mean_values_target, [last_value])

    if np.min(arr) > mean_values_target[0] or np.max(arr) < mean_values_target[-1]:
        raise ValueError("find_bin_edges: mean_values are not adapted to array")

    edges = np.zeros(len(mean_values_target)+1)
    edges[0], edges[-1] = np.min(arr), np.max(arr)
    for i in range(1,len(edges)-1):
        edges[i] = edges[i-1]
        meanval = mean_values_target[i-1]-1
        while(meanval<mean_values_target[i-1]):
            sel = (arr>edges[i])
            edges[i] = np.min(arr[sel])
            sel = (arr>=edges[i-1]) & (arr<edges[i])
            meanval = np.mean(arr[sel])

    if for_p_cross:
        edges[0] = 0
        edges = edges[:-1]

    if debug:
        print("** Debug:")
        print("Bin edges:", edges)
        for i in range(len(mean_values_target)-1):
            sel = (arr>=edges[i]) & (arr<edges[i+1])
            print("Mean values (target/effective):", mean_values_target[i], np.mean(arr[sel]))

    return edges


def fitfunc_std_fftproduct(snr, amp, zero_point):
    """Standard deviation fit equation

    Arguments
    ---------
    snr (float): 
    The signal-to-noise ratio of the signal. Must be greater than 1.

    amp (float): 
    The amplitude of the signal.

    zero_point (float): 
    The zero point offset of the signal.

    Return
    ------
    float: The std of the signal.
    """
    return (amp / (snr - 1)) + zero_point


def fitfunc_variance_pk1d(snr, amp, zero_point):
    """Variance fit equation

    Arguments
    ---------
    snr (float): 
    The signal-to-noise ratio of the signal. Must be greater than 1.

    amp (float): 
    The amplitude of the signal.

    zero_point (float): 
    The zero point offset of the signal.

    Return
    ------
    float: The std of the signal.
    """
    return (amp / (snr - 1)**2) + zero_point


def read_fits_table(tablefilename):
    return Table.read(tablefilename)
