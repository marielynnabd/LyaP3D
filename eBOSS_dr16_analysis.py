""" This module provides a set of functions to get P3D from Pcross computed on data """

import numpy as np
import os, sys, glob
import multiprocessing
from multiprocessing import Pool
import fitsio
from astropy.io import fits
from astropy.table import Table, vstack
import scipy
from scipy.constants import speed_of_light as speed_light
SPEED_LIGHT = speed_light / 1000.  # [km/s]


def get_qso_deltas_singlefile(delta_file_name, qso_cat, lambda_min, lambda_max,
                              include_snr_reso=False, spec_dir=None, with_interpolation=False,
                              wavegrid_type='P1D'):
    """ This function returns a table of ra, dec, wavelength and delta for each of the QSOs in qso_cat.
    Wavelenghts are selected in [lambda_min, lambda_max]

    Arguments:
    ----------
    delta_file_name: String
    delta fits file, DR16 format.

    qso_cat: Table
    QSO catalog for which we want to find the corresponding deltas, using THING_ID

    lambda_min: Float
    Value of the minimum forest wavelength required

    lambda_max: Float
    Value of the maximum forest wavelength required

    include_snr_reso: bool, default False
    If set, load SDSS spectra from `spec_dir`, and includes
    MEANRESOLUTION and MEANSNR to los_table

    spec_dir: String, default None
    Required if include_reso_and_snr is True

    with_interpolation: bool, default False
    If True, interpolate the deltas on the reference BOSS wavelength grid.
    In principle we dont have to use that.

    Return:
    -------
    los_table: Table
    Table where each row corresponds to a QSO, containing [ra, dec, wavelengths, deltas]
    """
    
    lambda_lya = 1215.67 # Angstrom
    
    if wavegrid_type=='DR16-public':
        #- Wavelength grid for the DR16 BAO analysis
        wavelength_ref_min = 3600 # Angstrom
        wavelength_ref_max = 7235 # Angstrom
        delta_loglam = 0.0003
    elif wavegrid_type=='P1D':
        #- P1D-adapted wavelength grid
        # match parameters from P3D/config_picca_delta_extraction_bossdr16.ini
        wavelength_ref_min = 3750 # Angstrom
        wavelength_ref_max = 7200 # Angstrom
        delta_loglam = 1.e-4

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
    los_table['THING_ID'] = np.ones(n_hdu, dtype='>i8') * np.nan
    if include_snr_reso:
        los_table['MEANRESOLUTION'] = np.zeros(n_hdu)
        los_table['MEANSNR'] = np.zeros(n_hdu)

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
                        los_table[i]['THING_ID'] = delta_ID

                    else:
                        print('Warning')  # should not happen in principle
                else:
                    n_masked += 1

        if (not np.isnan(los_table[i]['ra'])) and include_snr_reso:
            # To get snr&reso: take them from the delta files if available
            if ('MEANSNR' in delta_i_header) and ('MEANRESO' in delta_i_header):
                los_table[i]['MEANSNR'] = delta_i_header['MEANSNR']
                los_table[i]['MEANRESOLUTION'] = delta_i_header['MEANRESO']
            else:
                meansnr, meanreso = get_snr_reso_sdss(qso_cat, delta_ID, spec_dir, wavelength_ref)
                los_table[i]['MEANSNR'] = meansnr
                los_table[i]['MEANRESOLUTION'] = meanreso

    mask_los_used = ~np.isnan(los_table['ra'])
    los_table = los_table[mask_los_used]
    print("DR16 delta file", delta_file_name,":",len(los_table),"LOS used")
    if n_masked>0:
        print("    (",n_masked,"LOS not used presumably due to masked pixels)")

    return los_table



def get_los_table_dr16(qso_cat, deltas_dir, lambda_min, lambda_max, ncpu='all',
                       outputfile=None, include_snr_reso=False, spec_dir=None):
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
            [[f, qso_cat, lambda_min, lambda_max, include_snr_reso, spec_dir] for f in deltafiles]
        )

    for x in output_get_qso_deltas:
        if x is None: print("output of get_qso_deltas is None")  # should not happen in principle

    output_get_qso_deltas = [x for x in output_get_qso_deltas if x is not None]
    los_table = vstack([output_get_qso_deltas[i] for i in range(len(output_get_qso_deltas))])

    if outputfile is not None:
        los_table.write(outputfile)

    return los_table


def get_snr_reso_sdss(qso_cat, thing_id, spec_dir, wavelength_ref):
    # Function used in get_qso_deltas_singlefile
    #  inspired from picca/py/picca/delta_extraction/data_catalogues/sdss_data.py
    # !This is needed only if deltas were taken from a "BAO-like" catalog
    # (picca's deltas with P1D option already have mean snr and resolution)

    w, = np.where(np.isin(qso_cat['THING_ID'], thing_id))
    if len(w)!=1: print("Warning THING_ID")  # should not happen in principle
    plate = qso_cat[w[0]]["PLATE"]
    mjd = qso_cat[w[0]]["MJD"]
    fiberid = qso_cat[w[0]]["FIBERID"]
    filename = (f"{spec_dir}/{plate}/spec-{plate}-{mjd}-"
                    f"{fiberid:04d}.fits")
    if not os.path.isfile(filename):  # This happens...
        print("File does not exist:", filename)
        return (0, 0)
    hdul = fitsio.FITS(filename)

    log_lambda = np.array(hdul[1]["loglam"][:], dtype=np.float64)
    flux = np.array(hdul[1]["flux"][:], dtype=np.float64)
    ivar = (np.array(hdul[1]["ivar"][:], dtype=np.float64) * hdul[1]["and_mask"][:] == 0)
    wdisp = hdul[1]["wdisp"][:]

    mask_wavelength = (log_lambda >= np.min(wavelength_ref)) & (log_lambda <= np.max(wavelength_ref))
    if np.sum(mask_wavelength)==0:
        print("No loglam matching RoI")  # should not happen in principle
        return (0, 0)

    log_lambda = log_lambda[mask_wavelength]
    flux = flux[mask_wavelength]
    ivar = ivar[mask_wavelength]
    wdisp = wdisp[mask_wavelength]

    reso = _spectral_resolution(wdisp, fiberid, log_lambda)
    snr = flux * np.sqrt(ivar)

    return (np.mean(snr), np.mean(reso))


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
