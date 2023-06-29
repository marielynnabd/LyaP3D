""" This module provides a set of useful functions used for Pcross and P3D computations."""
 

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

