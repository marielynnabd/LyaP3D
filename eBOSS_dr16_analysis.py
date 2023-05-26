""" This module provides a set of functions to get P3D from Pcross computed on data """

import numpy as np
import os, sys
from astropy.io import fits
from astropy.table import Table, vstack
import scipy
import matplotlib.pyplot as plt

sys.path.insert(0, os.environ['HOME']+'/Software/picca/py')
from picca import constants
from picca.constants import SPEED_LIGHT # in km/s


def get_qso_deltas(delta_file_name, qso_cat, lambda_min, lambda_max):
    """ This function returns a table of ra, dec, wavelength and delta for each of the QSOs in the qso_cat

    Arguments:
    ----------
    delta_file_name: String
    delta fits file
    
    qso_cat: String
    qso_cat for which we want to find the corresponding deltas in delta_directory
    
    lambda_min: Float
    Value of the minimum forest wavelength required
    
    lambda_max: Float
    Value of the maximum forest wavelength required
    
    Return:
    -------
    all_los_table: Table
    Table where each row corresponds to a QSO, containing [ra, dec, wavelength, delta_los]
    """
    
    lambda_lya = 1215.67 # Angstrom
    
    # Reading the THING_ID of each quasar in the catalog
    qso_thing_id = np.array(qso_cat['THING_ID'])
    
    # Defining wavelength_ref for eBOSS analysis
    wavelength_ref_min = 3600 # Angstrom
    wavelength_ref_max = 7235 # Angstrom
    delta_loglam = 0.0003
    wavelength_ref = np.arange(np.log10(wavelength_ref_min), np.log10(wavelength_ref_max), delta_loglam)
    mask_wavelength_ref = (wavelength_ref > np.log10(lambda_min)) & (wavelength_ref < np.log10(lambda_max))
    wavelength_ref = wavelength_ref[mask_wavelength_ref]
    print('wavelength_ref', len(wavelength_ref), wavelength_ref)
    
    # Initializing table all_los_table
    all_los_table = Table()
    
    delta_file = fits.open(delta_file_name)
    n_hdu = len(delta_file) # Each delta file contains many hdu
 
    for i in range(1, n_hdu):
        delta_i_header = delta_file[i].header
        delta_i_data = delta_file[i].data
        delta_ID = delta_i_header['THING_ID']
        
        if delta_ID in qso_thing_id:
            # Reading data
            delta_los = np.array(delta_i_data['DELTA'])
            wavelength = np.array(delta_i_data['LOGLAM'])
            ra_coord = delta_i_header['RA'] * 180 / np.pi  # in rad and must be converted to degree
            dec_coord = delta_i_header['DEC'] * 180 / np.pi
            
            # Checking if LOGLAM.min < log(lambda_min) & LOGLAM.max > log(lambda_max)
            if (wavelength.min() < np.log10(lambda_min)) and (wavelength.max() > np.log10(lambda_max)):
                # Define wavelength mask
                mask_wavelength = (wavelength > np.log10(lambda_min)) & (wavelength < np.log10(lambda_max))
                #print('wavelength',len(wavelength[mask_wavelength]), wavelength[mask_wavelength])
                
                # Checking that the masked wavelength and wavelength_ref have the same shape otherwise it means that there are masked pixels and we don't want to consider this delta in the cslculation
                if len(wavelength[mask_wavelength]) == len(wavelength_ref):
                    
                    if np.allclose(wavelength[mask_wavelength], wavelength_ref):
                        # Initializing table
                        delta_table = Table()
                        delta_table['ra'] = np.zeros(1)
                        delta_table['dec'] = np.zeros(1)
                        delta_table['delta_los'] = np.zeros((1, np.sum(mask_wavelength)))
                        delta_table['wavelength'] = np.zeros((1, np.sum(mask_wavelength)))

                        # Filling table
                        # print(len(delta_los[mask_wavelength]), delta_los[mask_wavelength])
                        # print(len(wavelength[mask_wavelength]), wavelength[mask_wavelength])
                        delta_table['delta_los'] = delta_los[mask_wavelength]
                        delta_table['wavelength'] = wavelength[mask_wavelength]
                        delta_table['ra'] = ra_coord
                        delta_table['dec'] = dec_coord

                        # Stacking
                        all_los_table = vstack([all_los_table, delta_table])
                        
                    else:
                        print('Warning')
                else:
                    print('Masked pixels in this delta, therefore it will be discarded')
            
    return all_los_table


def get_qso_deltas_with_interp(delta_file_name, qso_cat):
    """ This function returns a table of ra, dec, wavelength and delta for each of the QSOs in the qso_cat

    Arguments:
    ----------
    delta_file_name: String
    delta fits file
    
    qso_cat: String
    qso_cat for which we want to find the corresponding deltas in delta_directory
    
    Return:
    -------
    all_los_table: Table
    Table where each row corresponds to a QSO, containing [ra, dec, wavelength, delta_los]
    """
    
    # Reading the THING_ID of each quasar in the catalog
    qso_thing_id = np.array(qso_cat['THING_ID'])

    # Initializing table all_los_table
    all_los_table = Table()
    
    delta_file = fits.open(delta_file_name)
    n_hdu = len(delta_file) # Each delta file contains many hdu
 
    for i in range(1, n_hdu):
        delta_i_header = delta_file[i].header
        delta_i_data = delta_file[i].data
        delta_ID = delta_i_header['THING_ID']
        
        if delta_ID in qso_thing_id:
            # Reading data
            delta_los = np.array(delta_i_data['DELTA'])
            wavelength = np.array(delta_i_data['LOGLAM'])
            ra_coord = delta_i_header['RA'] * 180 / np.pi  # in rad and must be converted to degree
            dec_coord = delta_i_header['DEC'] * 180 / np.pi
            
            # Interpolating deltas
            wavelength_min = np.min(wavelength)
            wavelength_max = np.max(wavelength)
            n_wavelength = 300
            wavelength_for_interpolation = np.linspace(wavelength_min, wavelength_max, n_wavelength)
            interpolation_function = scipy.interpolate.UnivariateSpline(wavelength, delta_los, s=0)
            delta_los_interpolated = interpolation_function(wavelength_for_interpolation)
            
            # Initializing table
            delta_table = Table()
            delta_table['ra'] = np.zeros(1)
            delta_table['dec'] = np.zeros(1)
            delta_table['delta_los'] = np.zeros((1, n_wavelength))
            delta_table['wavelength'] = np.zeros((1, n_wavelength))
            
            # Filling table
            delta_table['delta_los'] = delta_los_interpolated
            delta_table['wavelength'] = wavelength_for_interpolation
            delta_table['ra'] = ra_coord
            delta_table['dec'] = dec_coord
            
            # Stacking
            all_los_table = vstack([all_los_table, delta_table])
            
    return all_los_table