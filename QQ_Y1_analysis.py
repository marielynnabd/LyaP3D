""" Functions to read DESI delta files (Y1), creating a los_table to be used later by the pipeline """

import numpy as np
import os, sys, glob
import multiprocessing
from multiprocessing import Pool
import fitsio
from astropy.io import fits
from astropy.table import Table, vstack
import scipy

from .tools import SPEED_LIGHT


def get_desi_deltas_singlefile(delta_file_name, qso_cat, lambda_min, lambda_max,
                              include_snr_reso=False):
    """ This function returns a table of ra, dec, wavelength and delta for each of the QSOs in qso_cat.
    Wavelenghts are selected in [lambda_min, lambda_max]

    Arguments:
    ----------
    delta_file_name: String
    delta fits file, DR16 format.

    qso_cat: Table
    QSO catalog for which we want to find the corresponding deltas, using TARGETID

    lambda_min: Float
    Value of the minimum forest wavelength required

    lambda_max: Float
    Value of the maximum forest wavelength required

    include_snr_reso: bool, default False
    If set, includes MEANRESO and MEANSNR from the delta's headers, to the los_table

    Return:
    -------
    los_table: Table
    Table where each row corresponds to a QSO, containing [ra, dec, wavelengths, deltas]
    """
    
    # Reference DESI wavelength grid
    wavelength_ref_min = 3600.  # AA
    wavelength_ref_max = 9824.  # AA
    delta_lambda = 0.8  # AA
    wavelength_ref = np.arange(wavelength_ref_min, wavelength_ref_max+0.01, delta_lambda)
    mask_wavelength_ref = (wavelength_ref > lambda_min) & (wavelength_ref < lambda_max)
    wavelength_ref = wavelength_ref[mask_wavelength_ref]

    # Reading the TARGETID of each quasar in the catalog
    qso_tid = np.array(qso_cat['TARGETID'])
    
    # Reading delta file and accessing useful data
    delta_file = fitsio.FITS(delta_file_name)
    ## Wavelength
    _lambda  = delta_file[1].read()
    ## Meta data
    _meta_data = delta_file[2].read()
    n_los = len(_meta_data['TARGETID'])
    print("DESI delta file ", delta_file_name, ":", n_los, "lines-of-sight")
    ## Delta
    _delta = delta_file[3].read() # Has shape (n_los, len(_lambda))
    
    # Number of masked deltas, i.e. for LOS that include pixel masking and which we don't want to consider in our study for the moment
    n_masked = 0

    # Initializing table los_table
    los_table = Table()
    los_table['ra'] = np.ones(n_los) * np.nan
    los_table['dec'] = np.ones(n_los) * np.nan
    los_table['delta_los'] = np.zeros((n_los, len(wavelength_ref)))
    los_table['wavelength'] = np.zeros((n_los, len(wavelength_ref)))
    los_table['TARGETID'] = np.zeros(n_los, dtype='>i8')

    if include_snr_reso:
        los_table['MEANRESO'] = np.zeros(n_los)
        los_table['MEANSNR'] = np.zeros(n_los)

    for i in range(n_los):
        if i%100==0 :
            print(delta_file_name,": LOS",i,"/",n_los)

        delta_ID = _meta_data['TARGETID'][i]
        delta_ra = _meta_data['RA'][i]
        delta_dec = _meta_data['DEC'][i]

        if delta_ID in qso_tid:
            # Reading data
            delta = _delta[i,:][~np.isnan(_delta[i,:])] # Thse cuts are just to remove the nan that exist in QQ mocks
            wavelength = _lambda[~np.isnan(_delta[i,:])]
            
            # Checking if LAMBDA.min < lambda_min & LAMBDA.max > lambda_max
            if (wavelength.min() < lambda_min) and (wavelength.max() > lambda_max):
                # Define wavelength mask to be applied later to wavelength and delta so that they have the same grid as wavelength_ref
                mask_wavelength = (wavelength > lambda_min) & (wavelength < lambda_max)

                # Checking that the masked wavelength and wavelength_ref have the same shape
                # otherwise it means that there are masked pixels
                # and we don't want to consider this delta in the calculation
                if len(wavelength[mask_wavelength]) == len(wavelength_ref):
                    if np.allclose(wavelength[mask_wavelength], wavelength_ref):
                        los_table[i]['ra'] = delta_ra * 180 / np.pi  # must convert rad --> dec.
                        los_table[i]['dec'] = delta_dec * 180 / np.pi
                        los_table[i]['delta_los'] = delta[mask_wavelength]
                        los_table[i]['wavelength'] = wavelength[mask_wavelength]
                        los_table[i]['TARGETID'] = delta_ID
                    else:
                        print('Warning')  # should not happen in principle
                else:
                    n_masked += 1

        # if (not np.isnan(los_table[i]['ra'])) and include_snr_reso:
        #     if ('MEANSNR' in delta_i_header) and ('MEANRESO' in delta_i_header):
        #         los_table[i]['MEANSNR'] = delta_i_header['MEANSNR']
        #         los_table[i]['MEANRESO'] = delta_i_header['MEANRESO']
        if (not np.isnan(los_table[i]['ra'])) and include_snr_reso:
            if 'MEANSNR' in delta_i_header:
                los_table[i]['MEANSNR'] = delta_i_header['MEANSNR']
            else:
                print('Warning, no MEANSNR/MEANRESO in delta header.')

    # Closing delta_file
    delta_file.close()

    mask_los_used = ~np.isnan(los_table['ra'])
    los_table = los_table[mask_los_used]
    print("DESI delta file", delta_file_name,":",len(los_table),"LOS used")
    if n_masked>0:
        print("    (",n_masked,"LOS not used presumably due to masked pixels)")

    return los_table


def get_desi_deltas_singlehdu_test(i_n_hdu, delta_file, qso_tid, lambda_min, lambda_max, include_snr_reso=False):
    """ This function returns a table of ra, dec, wavelength and delta for each of the QSOs in qso_cat.
    Wavelenghts are selected in [lambda_min, lambda_max]

    Arguments:
    ----------
    delta_file_name: String
    delta fits file, DR16 format.

    qso_cat: Table
    QSO catalog for which we want to find the corresponding deltas, using TARGETID

    lambda_min: Float
    Value of the minimum forest wavelength required

    lambda_max: Float
    Value of the maximum forest wavelength required

    include_snr_reso: bool, default False
    If set, includes MEANRESO and MEANSNR from the delta's headers, to the los_table

    Return:
    -------
    los_table: Table
    Table where each row corresponds to a QSO, containing [ra, dec, wavelengths, deltas]
    """
    
    # Reference DESI wavelength grid
    wavelength_ref_min = 3600.  # AA
    wavelength_ref_max = 9824.  # AA
    delta_lambda = 0.8  # AA
    wavelength_ref = np.arange(wavelength_ref_min, wavelength_ref_max+0.01, delta_lambda)
    mask_wavelength_ref = (wavelength_ref > lambda_min) & (wavelength_ref < lambda_max)
    wavelength_ref = wavelength_ref[mask_wavelength_ref]

    # Initializing table los_table
    los_table_singlehdu = Table()
    los_table_singlehdu['ra'] = np.ones(1) * np.nan
    los_table_singlehdu['dec'] = np.ones(1) * np.nan
    los_table_singlehdu['delta_los'] = np.zeros((1, len(wavelength_ref)))
    los_table_singlehdu['wavelength'] = np.zeros((1, len(wavelength_ref)))
    los_table_singlehdu['TARGETID'] = np.zeros(1, dtype='>i8')

    if include_snr_reso:
        los_table_singlehdu['MEANRESO'] = np.zeros(1)
        los_table_singlehdu['MEANSNR'] = np.zeros(1)

    delta_i_header = delta_file[i_n_hdu+1].read_header()
    delta_ID = delta_i_header['TARGETID']

    if delta_ID in qso_tid:
        # Reading data
        try:
            delta_los = delta_file[i_n_hdu+1]['DELTA'][:]
        except:
            delta_los = delta_file[i_n_hdu+1]['DELTA_BLIND'][:]
        wavelength = delta_file[i_n_hdu+1]['LAMBDA'][:]

        # Checking if LAMBDA.min < lambda_min & LAMBDA.max > lambda_max
        if (wavelength.min() < lambda_min) and (wavelength.max() > lambda_max):
            # Define wavelength mask
            mask_wavelength = (wavelength > lambda_min) & (wavelength < lambda_max)
            #print('wavelength',len(wavelength[mask_wavelength]), wavelength[mask_wavelength])

            # Checking that the masked wavelength and wavelength_ref have the same shape
            # otherwise it means that there are masked pixels
            # and we don't want to consider this delta in the calculation
            if len(wavelength[mask_wavelength]) == len(wavelength_ref):

                if np.allclose(wavelength[mask_wavelength], wavelength_ref):
                    los_table_singlehdu['ra'] = delta_i_header['RA'] * 180 / np.pi  # must convert rad --> deg.
                    los_table_singlehdu['dec'] = delta_i_header['DEC'] * 180 / np.pi
                    los_table_singlehdu['delta_los'] = delta_los[mask_wavelength]
                    los_table_singlehdu['wavelength'] = wavelength[mask_wavelength]
                    los_table_singlehdu['TARGETID'] = delta_ID

                else:
                    print('Warning')  # should not happen in principle

    if (not np.isnan(los_table_singlehdu['ra'])) and include_snr_reso:
        if ('MEANSNR' in delta_i_header) and ('MEANRESO' in delta_i_header):
            los_table_singlehdu['MEANSNR'] = delta_i_header['MEANSNR']
            los_table_singlehdu['MEANRESOLUTION'] = delta_i_header['MEANRESO']
        else:
            print('Warning, no MEANSNR/MEANRESO in delta header.')

    return los_table_singlehdu


def get_desi_deltas_singlefile_test(delta_file_name, qso_cat, lambda_min, lambda_max, ncpu='all', include_snr_reso=False):
    """ This function returns a table of ra, dec, wavelength and delta for each of the QSOs in qso_cat.
    Wavelenghts are selected in [lambda_min, lambda_max]

    Arguments:
    ----------
    delta_file_name: String
    delta fits file, DR16 format.

    qso_cat: Table
    QSO catalog for which we want to find the corresponding deltas, using TARGETID

    lambda_min: Float
    Value of the minimum forest wavelength required

    lambda_max: Float
    Value of the maximum forest wavelength required

    include_snr_reso: bool, default False
    If set, includes MEANRESO and MEANSNR from the delta's headers, to the los_table

    Return:
    -------
    los_table: Table
    Table where each row corresponds to a QSO, containing [ra, dec, wavelengths, deltas]
    """

    # Reading the TARGETID of each quasar in the catalog
    qso_tid = np.array(qso_cat['TARGETID'])
    
    delta_file = fitsio.FITS(delta_file_name)
    n_hdu = len(delta_file)-1 # Each delta file contains many hdu (don't take into account HDU0)
    print("DESI delta file ", delta_file_name, ":", n_hdu, "HDUs")
    
    if ncpu=='all':
        ncpu = multiprocessing.cpu_count()
        
    with Pool(ncpu) as pool:
        output_get_desi_deltas_singlehdu = pool.starmap(
            get_desi_deltas_singlehdu_test,
            [[i_n_hdu, delta_file, qso_tid, lambda_min, lambda_max, ncpu, include_snr_reso] for i_n_hdu in range(n_hdu)]
        )

    for x in output_get_desi_deltas_singlehdu:
        if x is None: print("output of get_desi_deltas_singlehdu is None")  # should not happen in principle

    output_get_desi_deltas_singlehdu = [x for x in output_get_desi_deltas_singlehdu if x is not None]
    los_table_singlefile = vstack([output_get_desi_deltas_singlehdu[i] for i in range(len(output_get_desi_deltas_singlehdu))])

    mask_los_used = ~np.isnan(los_table_singlefile['ra'])
    los_table_singlefile = los_table_singlefile[mask_los_used]
    print("DESI delta file", delta_file_name,":",len(los_table_singlefile),"LOS used")
    n_masked = n_hdu - np.sum(mask_los_used)
    if n_masked>0:
        print("    (",n_masked,"LOS not used presumably due to masked pixels)")

    return los_table_singlefile


def get_los_table_desi_test(qso_cat, deltas_dir, lambda_min, lambda_max, ncpu='all',
                       outputfile=None, include_snr_reso=False):
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
            get_desi_deltas_singlefile_test,
            [[f, qso_cat, lambda_min, lambda_max, include_snr_reso] for f in deltafiles]
        )

    for x in output_get_qso_deltas:
        if x is None: print("output of get_qso_deltas is None")  # should not happen in principle

    output_get_qso_deltas = [x for x in output_get_qso_deltas if x is not None]
    los_table = vstack([output_get_qso_deltas[i] for i in range(len(output_get_qso_deltas))])

    if outputfile is not None:
        los_table.write(outputfile)

    return los_table


def get_los_table_desi(qso_cat, deltas_dir, lambda_min, lambda_max, ncpu='all',
                       outputfile=None, include_snr_reso=False):
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
            get_desi_deltas_singlefile,
            [[f, qso_cat, lambda_min, lambda_max, include_snr_reso] for f in deltafiles]
        )

    for x in output_get_qso_deltas:
        if x is None: print("output of get_qso_deltas is None")  # should not happen in principle

    output_get_qso_deltas = [x for x in output_get_qso_deltas if x is not None]
    los_table = vstack([output_get_qso_deltas[i] for i in range(len(output_get_qso_deltas))])

    if outputfile is not None:
        los_table.write(outputfile)

    return los_table



def _spectral_resolution(wdisp, fiberid, log_lambda):
    # adapted from picca/py/picca/delta_extraction/utils_pk1d.py
    reso = wdisp * SPEED_LIGHT * 1.0e-4 * np.log(10.)

    lambda_ = np.power(10., log_lambda)
    # compute the wavelength correction
    correction = 1.267 - 0.000142716 * lambda_ + 1.9068e-08 * lambda_ * lambda_
    correction[lambda_ > 6000.0] = 1.097

    # add the fiberid correction
    # fiberids greater than 500 corresponds to the second spectrograph
    fiberid = fiberid % 500
    if fiberid < 100:
        correction = (1. + (correction - 1) * .25 + (correction - 1) * .75 *
                      (fiberid) / 100.)
    elif fiberid > 400:
        correction = (1. + (correction - 1) * .25 + (correction - 1) * .75 *
                      (500 - fiberid) / 100.)

    # apply the correction
    reso *= correction

    return reso

