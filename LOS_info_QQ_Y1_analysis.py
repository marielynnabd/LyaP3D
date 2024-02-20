""" Functions to read Quick Quasar delta files (Y1), creating a los_info_table to be used later for pairs counting in redshift bins """

import numpy as np
import os, sys, glob
import multiprocessing
from multiprocessing import Pool
import fitsio
from astropy.io import fits
from astropy.table import Table, vstack
import scipy

from .tools import SPEED_LIGHT


def get_los_info_singlefile(delta_file_name, qso_cat, lambda_min, lambda_max, z_center, include_snr_reso=False):
    """ This function returns a table of ra, dec, TARGETID, MEANRESO and MEANSNR for each of the QSOs in qso_cat, for specific redshift bins.3
    Wavelenghts are selected in [lambda_min, lambda_max]

    Arguments:
    ----------
    delta_file_name: String
    delta fits file, DESI format.

    qso_cat: Table
    QSO catalog for which we want to find the corresponding deltas, using TARGETID

    lambda_min: Float or array of floats
    Value of the minimum forest wavelength required per redshift bin

    lambda_max: Float or array of floats
    Value of the maximum forest wavelength required per redshift bin
    
    z_center: Float or array of floats
    Value of z center per redshift bin
    PS: lambda_min, lambda_max and z_center must have the same length

    include_snr_reso: bool, default False
    If set, includes MEANRESO and MEANSNR from the delta's headers, to the los_info_table

    Return:
    -------
    los_info_table_list: Table or list of tables where each corresponds to one redshift bin
    Table where each row corresponds to a QSO, containing [ra, dec, wavelengths, deltas]
    """
    
    # Checking if z_center is array or float
    if hasattr(z_center,'__len__') is False:
        z_center = np.array([z_center])
    
    # Reference DESI wavelength grid
    wavelength_ref_min = 3600.  # AA
    wavelength_ref_max = 9824.  # AA
    delta_lambda = 0.8  # AA

    # Reading the TARGETID of each quasar in the catalog
    qso_tid = np.array(qso_cat['TARGETID'])
    
    # Reading delta file and accessing useful data
    delta_file = fitsio.FITS(delta_file_name)
    ## Wavelength
    _lambda  = delta_file[1].read()
    ## Meta data
    _meta_data = delta_file[2].read()
    n_los =len(_meta_data['TARGETID'])
    print("DESI delta file ", delta_file_name, ":", n_los, "lines-of-sight")
    ## Delta
    _delta = delta_file[3].read() # Has shape (n_los, len(_lambda))
    
    n_masked = 0
    
    # Creating a list of tables where each table corresponds to a redshift bin
    los_info_table_list = []
    for j in range(len(z_center)):
        # Initializing table los_info_table
        los_info_table = Table()
        los_info_table['z_center'] = np.ones(n_los) * z_center[j]
        los_info_table['ra'] = np.ones(n_los) * np.nan
        los_info_table['dec'] = np.ones(n_los) * np.nan
        los_info_table['TARGETID'] = np.zeros(n_los, dtype='>i8')
        if include_snr_reso:
            # los_info_table['MEANRESO'] = np.zeros(n_los)
            los_info_table['MEANSNR'] = np.zeros(n_los)
        los_info_table_list.append(los_info_table)
    
    # 1st loop over LOS number
    for i in range(n_los):
        if i%100==0 :
            print(delta_file_name,": LOS",i,"/",n_los)
        delta_ID = _meta_data['TARGETID'][i]
        delta_ra = _meta_data['RA'][i]
        delta_dec = _meta_data['DEC'][i]

        if delta_ID in qso_tid:
            # Reading data
            delta = _delta[i,:]
            delta = delta[~np.isnan(delta)]
            wavelength = _lambda[~np.isnan(delta)]
            
            # 2nd loop over Redshift bins
            for j in range(len(z_center)): # los_info_table_list must have n_zbins lists
                wavelength_ref = np.arange(wavelength_ref_min, wavelength_ref_max+0.01, delta_lambda) # same all the time but need to be redefined
                mask_wavelength_ref = (wavelength_ref > lambda_min[j]) & (wavelength_ref < lambda_max[j])
                wavelength_ref = wavelength_ref[mask_wavelength_ref]
                
                # This part is to check if the delta must be included in the redshift bin or not: 
                # Checking if LAMBDA.min < lambda_min & LAMBDA.max > lambda_max
                if (wavelength.min() < lambda_min[j]) and (wavelength.max() > lambda_max[j]):
                    # Define wavelength mask
                    mask_wavelength = (wavelength > lambda_min[j]) & (wavelength < lambda_max[j])

                    # Checking that the masked wavelength and wavelength_ref have the same shape 
                    # otherwise it means that there are masked pixels and we don't want to consider this delta in the calculation
                    if len(wavelength[mask_wavelength]) == len(wavelength_ref):
                        if np.allclose(wavelength[mask_wavelength], wavelength_ref):
                            los_info_table_list[j][i]['ra'] = delta_ra * 180 / np.pi  # must convert rad --> deg
                            los_info_table_list[j][i]['dec'] = delta_dec * 180 / np.pi
                            los_info_table_list[j][i]['TARGETID'] = delta_ID
                            # if include_snr_reso:
                            #     if ('MEANSNR' in delta_i_header) and ('MEANRESO' in delta_i_header):
                            #         los_info_table_list[j][i]['MEANSNR'] = delta_i_header['MEANSNR']
                            #         los_info_table_list[j][i]['MEANRESO'] = delta_i_header['MEANRESO']
                            #     else:
                            #         print('Warning, no MEANSNR/MEANRESO in delta header.')
                            if include_snr_reso:
                                if ('MEANSNR' in delta_i_header)
                                    los_info_table_list[j][i]['MEANSNR'] =  _meta_data['MEANSNR'][i]
                                    # los_info_table_list[j][i]['MEANRESO'] = delta_i_header['MEANRESO'] # Because no MEANRESO in the QQ delta files for now
                                else:
                                    print('Warning, no MEANSNR/MEANRESO in delta header.')
                        else:
                            print('Warning')  # should not happen in principle
                    else:
                        n_masked += 1

    if fits_flag == 'FITSIO':
        delta_file.close()
        
    for j in range(len(z_center)):
        mask_los_used = ~np.isnan(los_info_table_list[j]['ra'])
        los_info_table_list[j] = los_info_table_list[j][mask_los_used]

    return los_info_table_list


def get_los_info_table_desi(qso_cat, deltas_dir, lambda_min, lambda_max, z_center, outputdir, outputfilename, ncpu='all', include_snr_reso=False):
    """ This function returns a table of ra, dec, TARGETID, MEANRESO, MEANSNR, for each of the QSOs in qso_cat.
    Wavelenghts are selected in [lambda_min, lambda_max]
    Wrapper around get_los_info_singlefile

    Arguments:
    ----------
    qso_cat: Table
    QSO catalog for which we want to find the corresponding deltas, using THING_ID

    deltas_dir: string
    Directory where DESI delta files should be

    lambda_min: Float or array of floats
    Value of the minimum forest wavelength required

    lambda_max: Float or array of floats
    Value of the maximum forest wavelength required
    
    z_center: Float or array of floats
    Value of z center per redshift bin
    PS: lambda_min, lambda_max and z_center must have the same length

    ncpu: int or 'all'
    For multiprocessing.Pool

    outputdir: string, default None
    Write LOS table to file
    
    outputfilename: string, default None
    Name of the file

    Return:
    -------
    los_info_allfiles_allz: Table or list of tables where each corresponds to one redshift bin
    Table where each row corresponds to a QSO, containing [ra, dec, TARGETID, MEANSNR]
    """

    searchstr = '*'
    deltafiles = glob.glob(os.path.join(deltas_dir, f"delta{searchstr}.fits.gz"))

    if ncpu=='all':
        ncpu = multiprocessing.cpu_count()

    print("Nb of delta files:", len(deltafiles))
    print("Number of cpus:", multiprocessing.cpu_count())

    # This does the parallelization over all the delta files in deltas_dir, 
    # and the output_get_los_info_singlefile is a list of los_info_table(s), each table corresponding to one z bin
    with Pool(ncpu) as pool:
        output_get_los_info_singlefile = pool.starmap(
            get_los_info_singlefile,
            [[f, qso_cat, lambda_min, lambda_max, z_center, include_snr_reso] for f in deltafiles]
        )

    for x in output_get_los_info_singlefile:
        if x is None: print("output of get_los_info_singlefile is None")  # should not happen in principle

    los_info_allfiles_allz = [] # A list of tables each corresponding to one redshift bin
    for j in range(len(z_center)):
        output_get_los_info_singlefile_onez = [x[j] for x in output_get_los_info_singlefile if x[j] is not None] # Here it is a list of tables
        los_info_table_onez = vstack([output_get_los_info_singlefile_onez[i] for i in range(len(output_get_los_info_singlefile_onez))])
        # Writing table for one z bin
        # outputfile = os.path.join(output_dir, 'los_info_table_desi_'+str(z_center[j])+'.fits.gz')
        outputfile = os.path.join(outputdir, 'output_'+str(z_center[j]), outputfilename)
        los_info_table_onez.write(outputfile)

        los_info_allfiles_allz.append(los_info_table_onez)

    return los_info_allfiles_allz

