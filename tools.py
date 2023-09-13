""" This module provides a set of useful functions used for Pcross and P3D computations"""
 

import numpy as np
from astropy.table import Table, vstack
import astropy.constants

SPEED_LIGHT = astropy.constants.c.to('km/s').value  # km/s
LAMBDA_LYA = 1215.67  # Angstrom


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


def _meanvalues_from_binedges(arr, edges):
    meanvals = np.zeros(len(edges)-1)
    for i in range(len(meanvals)):
        sel = (arr>=edges[i]) & (arr<edges[i+1])
        meanvals[i] = np.mean(arr[sel])
    return meanvals


def find_bin_edges(arr, mean_values_target, debug=False):
    """ Find bin edges so that when histogramming an array, the 
    mean values in each bin are given.
    Output: edges, size = len(mean_values_target)+1
    """
    arr = np.array(arr)
    mean_values_target = np.sort(mean_values_target)
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

    if debug:
        print("** Debug:")
        print(edges)
        for i in range(len(mean_values_target)):
            sel = (arr>=edges[i]) & (arr<edges[i+1])
            print(mean_values_target[i], np.mean(arr[sel]))

    return edges
