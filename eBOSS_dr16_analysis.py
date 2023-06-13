""" This module provides a set of functions to get P3D from Pcross computed on data """

import numpy as np
import os, sys, glob
import multiprocessing
from multiprocessing import Pool

from astropy.io import fits
from astropy.table import Table, vstack
import scipy


def get_qso_deltas_singlefile(delta_file_name, qso_cat, lambda_min, lambda_max, with_interpolation=False):
    """ This function returns a table of ra, dec, wavelength and delta for each of the QSOs in qso_cat.
    Wavelenghts are selected in [lambda_min, lambda_max]

    Arguments:
    ----------
    delta_file_name: String
    delta fits file, DR16 format

    qso_cat: Table
    QSO catalog for which we want to find the corresponding deltas, using THING_ID

    lambda_min: Float
    Value of the minimum forest wavelength required

    lambda_max: Float
    Value of the maximum forest wavelength required

    with_interpolation: bool, default False
    If True, interpolate the deltas on the reference BOSS wavelength grid.
    In principle we dont have to use that.

    Return:
    -------
    los_table: Table
    Table where each row corresponds to a QSO, containing [ra, dec, wavelengths, deltas]
    """
    
    lambda_lya = 1215.67 # Angstrom
    
    # This are fixed BOSS parameters: wavelengths should be regularly gridded,
    # between 3600 and 7235A, in log scale
    wavelength_ref_min = 3600 # Angstrom
    wavelength_ref_max = 7235 # Angstrom
    delta_loglam = 0.0003

    # Reading the THING_ID of each quasar in the catalog
    qso_thing_id = np.array(qso_cat['THING_ID'])
    
    # Defining wavelength_ref for eBOSS analysis
    wavelength_ref = np.arange(np.log10(wavelength_ref_min), np.log10(wavelength_ref_max), delta_loglam)
    mask_wavelength_ref = (wavelength_ref > np.log10(lambda_min)) & (wavelength_ref < np.log10(lambda_max))
    wavelength_ref = wavelength_ref[mask_wavelength_ref]
    #print('wavelength_ref', len(wavelength_ref), wavelength_ref)
    
    delta_file = fits.open(delta_file_name)
    n_hdu = len(delta_file)-1 # Each delta file contains many hdu (don't take into account HDU0)
    print("DR16 delta file ", delta_file_name, ":", n_hdu, "HDUs")
    n_masked = 0

    # Initializing table los_table
    los_table = Table()
    los_table['ra'] = np.ones(n_hdu) * np.nan
    los_table['dec'] = np.ones(n_hdu) * np.nan
    los_table['delta_los'] = np.zeros((n_hdu, len(wavelength_ref)))
    los_table['wavelength'] = np.zeros((n_hdu, len(wavelength_ref)))

    for i in range(n_hdu):
        delta_i_header = delta_file[i+1].header
        delta_i_data = delta_file[i+1].data
        delta_ID = delta_i_header['THING_ID']
        
        if delta_ID in qso_thing_id:
            # Reading data
            delta_los = np.array(delta_i_data['DELTA'])
            wavelength = np.array(delta_i_data['LOGLAM'])
            
            # Checking if LOGLAM.min < log(lambda_min) & LOGLAM.max > log(lambda_max)
            if (wavelength.min() < np.log10(lambda_min)) and (wavelength.max() > np.log10(lambda_max)):
                # Define wavelength mask
                mask_wavelength = (wavelength > np.log10(lambda_min)) & (wavelength < np.log10(lambda_max))
                #print('wavelength',len(wavelength[mask_wavelength]), wavelength[mask_wavelength])
                
                if with_interpolation:
                    interp_fct = scipy.interpolate.UnivariateSpline(
                        wavelength[mask_wavelength],
                        delta_los[mask_wavelength],
                        s=0)
                    delta_los_interpolated = interp_fct(wavelength_ref)
                    los_table[i]['ra'] = delta_i_header['RA'] * 180 / np.pi
                    los_table[i]['dec'] = delta_i_header['DEC'] * 180 / np.pi
                    los_table[i]['delta_los'] = delta_los_interpolated
                    los_table[i]['wavelength'] = wavelength_ref

                # Checking that the masked wavelength and wavelength_ref have the same shape
                # otherwise it means that there are masked pixels
                # and we don't want to consider this delta in the calculation
                elif len(wavelength[mask_wavelength]) == len(wavelength_ref):
                    
                    if np.allclose(wavelength[mask_wavelength], wavelength_ref):
                        los_table[i]['ra'] = delta_i_header['RA'] * 180 / np.pi  # must convert rad --> dec.
                        los_table[i]['dec'] = delta_i_header['DEC'] * 180 / np.pi
                        los_table[i]['delta_los'] = delta_los[mask_wavelength]
                        los_table[i]['wavelength'] = wavelength[mask_wavelength]
                    else:
                        print('Warning')  # should not happen in principle
                else:
                    n_masked += 1

    mask_los_used = ~np.isnan(los_table['ra'])
    los_table = los_table[mask_los_used]
    print("DR16 delta file", delta_file_name,":",len(los_table),"LOS used")
    print("                 (",n_masked,"LOS not used presumably due to masked pixels)")
    return los_table



def get_los_table_dr16(qso_cat, deltas_dir, lambda_min, lambda_max, ncpu='all', outputfile=None):
    """ This function returns a table of ra, dec, wavelength and delta for each of the QSOs in qso_cat.
    Wavelenghts are selected in [lambda_min, lambda_max]
    Wrapper around get_qso_deltas_singlefile

    Arguments:
    ----------
    qso_cat: Table
    QSO catalog for which we want to find the corresponding deltas, using THING_ID

    deltas_dir: string
    Directory where DR16 delta files should be

    lambda_min: Float
    Value of the minimum forest wavelength required

    lambda_max: Float
    Value of the maximum forest wavelength required

    ncpu: int or 'all'
    For multiprocessing.Pool

    outputfile: string, default None
    Write LOS table to file

    Return:
    -------
    los_table: Table
    Table where each row corresponds to a QSO, containing [ra, dec, wavelengths, deltas]
    """

    searchstr = '*'
    deltafiles = glob.glob(os.path.join(deltas_dir, f"delta{searchstr}.fits.gz"))

    if ncpu=='all':
        ncpu = multiprocessing.cpu_count()

    print("Nb of delta files:", len(deltafiles))
    print("Number of cpus:", multiprocessing.cpu_count())

    with Pool(ncpu) as pool:
        output_get_qso_deltas = pool.starmap(
            get_qso_deltas_singlefile,
            [[f, qso_cat, lambda_min, lambda_max] for f in deltafiles]
        )

    for x in output_get_qso_deltas:
        if x is None: print("output of get_qso_deltas is None")  # should not happen in principle

    output_get_qso_deltas = [x for x in output_get_qso_deltas if x is not None]
    los_table = vstack([output_get_qso_deltas[i] for i in range(len(output_get_qso_deltas))])

    if outputfile is not None:
        los_table.write(outputfile)

    return los_table

