""" Functions to read QQ delta files computed by picca, or raw transmission files creating a los_table to be used later by the pipeline """

import numpy as np
import os, sys, glob
import multiprocessing
from multiprocessing import Pool
import fitsio
from astropy.io import fits
from astropy.table import Table, vstack
import scipy

from .tools import SPEED_LIGHT, LAMBDA_LYA


def get_QQ_Y1_deltas_singlefile(delta_file_name, qso_cat, lambda_min, lambda_max,
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
    print("QQ delta file ", delta_file_name, ":", n_los, "lines-of-sight")
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
        # los_table['MEANRESO'] = np.zeros(n_los)
        los_table['MEANSNR'] = np.zeros(n_los)

    for i in range(n_los):
        if i%100==0 :
            print(delta_file_name,": LOS",i,"/",n_los)

        delta_ID = _meta_data['TARGETID'][i]
        delta_ra = _meta_data['RA'][i]
        delta_dec = _meta_data['DEC'][i]

        if delta_ID in qso_tid:
            # Reading data
            delta = _delta[i,:][~np.isnan(_delta[i,:])] # These cuts are just to remove the nan that exist in QQ mocks
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
            try:
                los_table[i]['MEANSNR'] =  _meta_data['MEANSNR'][i]
                # los_info_table_list[j][i]['MEANRESO'] = delta_i_header['MEANRESO'] # Because no MEANRESO in the QQ delta files for now
            except:
                print('Warning, no MEANSNR/MEANRESO in delta header.')

    # Closing delta_file
    delta_file.close()

    mask_los_used = ~np.isnan(los_table['ra'])
    los_table = los_table[mask_los_used]
    print("QQ delta file", delta_file_name,":",len(los_table),"LOS used")
    if n_masked>0:
        print("    (",n_masked,"LOS not used presumably due to masked pixels)")

    return los_table


def get_los_table_QQ_Y1(qso_cat, deltas_dir, lambda_min, lambda_max, ncpu='all',
                       outputfile=None, include_snr_reso=False):
    """ This function returns a table of ra, dec, wavelength and delta for each of the QSOs in qso_cat.
    Wavelenghts are selected in [lambda_min, lambda_max]
    Wrapper around get_qso_deltas_singlefile

    Arguments:
    ----------
    qso_cat: Table
    QSO catalog for which we want to find the corresponding deltas, using THING_ID

    deltas_dir: string
    Directory where DESI delta files are stored

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
        output_get_QQ_Y1_deltas_singlefile = pool.starmap(
            get_QQ_Y1_deltas_singlefile,
            [[f, qso_cat, lambda_min, lambda_max, include_snr_reso] for f in deltafiles]
        )

    for x in output_get_QQ_Y1_deltas_singlefile:
        if x is None: print("output of get_QQ_Y1_deltas_singlefile is None")  # should not happen in principle

    output_get_QQ_Y1_deltas_singlefile = [x for x in output_get_QQ_Y1_deltas_singlefile if x is not None]
    los_table = vstack([output_get_QQ_Y1_deltas_singlefile[i] for i in range(len(output_get_QQ_Y1_deltas_singlefile))])

    if outputfile is not None:
        los_table.write(outputfile)

    return los_table


def get_Nyx_raw_deltas_singlefile(transmission_file_name, qso_cat, lambda_min, lambda_max):
    """ This function returns a table of ra, dec, wavelength and raw_deltas for each of the QSOs in qso_cat.
    Wavelenghts are selected in [lambda_min, lambda_max]

    Arguments:
    ----------
    transmission_file_name: String
    transmission fits file per healpix pixel with QQ input format

    qso_cat: Table
    QSO catalog for which we want to find the corresponding deltas, using TARGETID

    lambda_min: Float
    Value of the minimum forest wavelength required

    lambda_max: Float
    Value of the maximum forest wavelength required

    Return:
    -------
    los_table: Table
    Table where each row corresponds to a QSO, containing [ra, dec, wavelengths, raw_deltas]
    """

    # Reading the TARGETID of each quasar in the catalog
    qso_tid = np.array(qso_cat['TARGETID'])

    # Reading transmission file and accessing useful data
    transmission_file = fitsio.FITS(transmission_file_name)

    ## Meta data
    _meta_data = transmission_file[1].read()

    ## Here CUT all with zqso too small wrt lambda max
    all_z_qso = _meta_data['Z']
    select_qso = ((1 + all_z_qso) * LAMBDA_LYA) > lambda_max # To remove all qso having their Lyman alpha emission > lambda_max since they have ones in their vectors that affect the raw analysis

    ## Z of selected qso
    z_selected_qso = all_z_qso[select_qso] # These are only the qso satisfying the above condition
    n_los = len(z_selected_qso)

    ## MOCKID of selected qso
    mockid_selected_qso = _meta_data['MOCKID'][select_qso]

    ## RA and DEC of selected qso
    ra_selected_qso = _meta_data['RA'][select_qso]
    dec_selected_qso = _meta_data['DEC'][select_qso]
    print("Transmission file ", transmission_file_name, ":", n_los, "selected lines-of-sight satisfying the condition Lyman_alpha emission > lambda_max")

    ## Wavelength
    _lambda  = transmission_file[2].read()

    ### Defining wavelength mask to select chunk
    mask_wavelength = (_lambda > lambda_min) & (_lambda < lambda_max) # This selects the part of LOS between lambda_min and lambda_max

    ### Wavelength array in (lambda_min, lambda_max)
    wavelength = _lambda[mask_wavelength]

    ## raw transmission of selected qso within the (lambda_min, lambda_max) range
    raw_transmission_selected_qso = transmission_file[3].read()[select_qso][mask_wavelength] # Has shape (n_los, len(wavelength)) # To correct because this is the F_LYA and not F/Fmean - 1

    # Computing mean_flux_goal later used to compute raw_delta
    redshift = (wavelength / LAMBDA_LYA) - 1
    mean_flux_goal = np.exp(-25e-4 * (1 + redshift)**3.7)

    # Initializing table los_table
    los_table = Table()
    los_table['ra'] = np.ones(n_los) * np.nan
    los_table['dec'] = np.ones(n_los) * np.nan
    los_table['delta_los'] = np.zeros((n_los, len(wavelength)))
    los_table['wavelength'] = np.zeros((n_los, len(wavelength)))
    los_table['TARGETID'] = np.zeros(n_los, dtype='>i8')

    for i in range(n_los): # Here i corresponds to n_los where qso are already filtered out
        if i%100==0 :
            print(transmission_file_name, ": LOS", i, "/", n_los)

        los_ID = mockid_selected_qso[i]
        los_ra = ra_selected_qso[i]
        los_dec = dec_selected_qso[i]

        if los_ID in qso_tid: # In case we filtered the qso cat, the qso will also be removed from the raw analysis
            # Reading data
            los_raw_transmission = raw_transmission_selected_qso[i,:]

            # Filling table
            los_table[i]['ra'] = los_ra * 180 / np.pi  # must convert rad --> dec.
            los_table[i]['dec'] = los_dec * 180 / np.pi
            los_table[i]['delta_los'] = (los_raw_transmission / mean_flux_goal) - 1
            los_table[i]['wavelength'] = wavelength
            los_table[i]['TARGETID'] = los_ID
        else:
            print('QSO with LOS_ID: '+str(los_ID)+' not in QSO cat') # Should not occur in general except if we want to discard some qso by filtering the QSO cat in input

    # Closing transmission_file
    transmission_file.close()

    print("Transmission file", transmission_file_name, ":", len(los_table), "LOS used")

    return los_table


def get_los_table_Nyx(qso_cat, transmissions_dir, lambda_min, lambda_max, ncpu='all', outputfile=None):
    """ This function returns a table of ra, dec, wavelength and raw_delta (called delta_los in output) for each of the QSOs in qso_cat.
    Wavelenghts are selected in [lambda_min, lambda_max]
    Wrapper around get_Nyx_raw_deltas_singlefile

    Arguments:
    ----------
    qso_cat: Table
    QSO catalog for which we want to find the corresponding deltas, using THING_ID

    transmissions_dir: string
    Directory where transmission files are stored

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
    Table where each row corresponds to a QSO, containing [ra, dec, wavelengths, raw_deltas]
    """

    searchstr = '*'
    transmissionfiles = glob.glob(os.path.join(transmissions_dir, f"transmission-{searchstr}.fits.gz"))

    if ncpu=='all':
        ncpu = multiprocessing.cpu_count()

    print("Nb of transmission files:", len(transmissionfiles))
    print("Number of cpus:", multiprocessing.cpu_count())

    with Pool(ncpu) as pool:
        output_get_Nyx_raw_deltas_singlefile = pool.starmap(
            get_Nyx_raw_deltas_singlefile,
            [[f, qso_cat, lambda_min, lambda_max] for f in transmissionfiles]
        )

    for x in output_get_Nyx_raw_deltas_singlefile:
        if x is None: print("output of get_Nyx_raw_deltas_singlefile is None")  # should not happen in principle

    output_get_Nyx_raw_deltas_singlefile = [x for x in output_get_Nyx_raw_deltas_singlefile if x is not None]
    los_table = vstack([output_get_Nyx_raw_deltas_singlefile[i] for i in range(len(output_get_Nyx_raw_deltas_singlefile))])
    print('LOS tables stacked')
    print('length of final los_table: ', len(los_table))

    if outputfile is not None:
        los_table.write(outputfile)
        print('Output filesaved')

    return los_table