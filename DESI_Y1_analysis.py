""" Functions to read DESI delta files (Y1), creating a los_table to be used later by the pipeline """

import numpy as np
import os, sys, glob
import multiprocessing
from multiprocessing import Pool
import fitsio
from astropy.io import fits
from astropy.table import Table, vstack
import scipy

# from .tools import SPEED_LIGHT
sys.path.insert(0, os.environ['HOME']+'/Software/LyaP3D')
from tools import SPEED_LIGHT


# def get_desi_deltas_singlefile(delta_file_name, qso_cat, lambda_min, lambda_max, z_center, 
#                                lambda_pixelmask_min=None, lambda_pixelmask_max=None, 
#                                include_snr_reso=False):
def get_desi_los_singlefile(delta_file_name, qso_cat, lambda_min, lambda_max, z_center, 
                               lambda_pixelmask_min=None, lambda_pixelmask_max=None, 
                               include_snr_reso=False):
    """ This function returns a table of ra, dec, wavelength and delta for each of the QSOs in qso_cat.
    Wavelenghts are selected in [lambda_min, lambda_max]

    Arguments:
    ----------
    delta_file_name: String
    delta fits file, DR16 format.

    qso_cat: Table
    QSO catalog for which we want to find the corresponding deltas, using TARGETID

    lambda_min: Float or array of floats
    Value of the minimum forest wavelength required per redshift bin

    lambda_max: Float or array of floats
    Value of the maximum forest wavelength required per redshift bin
    
    z_center: Float or array of floats
    Value of z center per redshift bin
    PS: lambda_min, lambda_max and z_center must have the same length
    
    lambda_pixelmask_min: Float or array of floats
    Minimum of wavelength intervals to be masked. This will be applied not on data, but on wavelength_ref
    
    lambda_pixelmask_max: Float or array of floats
    Maximum of wavelength intervals to be masked. This will be applied not on data, but on wavelength_ref

    include_snr_reso: bool, default False
    If set, includes MEANRESO and MEANSNR from the delta's headers, to the los_table

    Return:
    -------
    los_table: Table
    los_info_table_list: Table or list of tables where each corresponds to one redshift bin
    In each of the tables: each row corresponds to a QSO, containing [ra, dec, delta_los, wavelength, TARGETID] and possibly [MEANSNR, MEANRESO]
    """
    
    # Checking if z_center is array or float
    if hasattr(z_center,'__len__') is False:
        z_center = np.array([z_center])
        
    # Checking if lambda_pixelmask_min and lambda_pixelmask_max are arrays or floats, only if they're not none
    if hasattr(lambda_pixelmask_min,'__len__') is False:
        lambda_pixelmask_min = np.array([lambda_pixelmask_min])
    if hasattr(lambda_pixelmask_max,'__len__') is False:
        lambda_pixelmask_max = np.array([lambda_pixelmask_max])

    # Reading the TARGETID of each quasar in the catalog
    qso_tid = np.array(qso_cat['TARGETID'])

    # Opening delta_file
    delta_file = fitsio.FITS(delta_file_name)
    n_hdu = len(delta_file)-1 # Each delta file contains many hdu (don't take into account HDU0)
    print("DESI delta file ", delta_file_name, ":", n_hdu, "HDUs")
    n_masked = 0

    # Reference DESI wavelength grid
    wavelength_ref_min = 3600.  # AA
    wavelength_ref_max = 9824.  # AA
    delta_lambda = 0.8  # AA
    wavelength_ref = np.arange(wavelength_ref_min, wavelength_ref_max+0.01, delta_lambda)

    # This part is to initialize a list of tables where each table corresponds to one redshift bin
    los_table_list = []
    for j in range(len(z_center)):
        # Selecting the part of wavelength_ref that corresponds to the z bin
        wavelength_ref_zbin = wavelength_ref.copy() # Where wavelength_ref is one for all zbins
        mask_wavelength_ref_zbin = (wavelength_ref_zbin > lambda_min[j]) & (wavelength_ref_zbin < lambda_max[j])
        wavelength_ref_zbin = wavelength_ref_zbin[mask_wavelength_ref_zbin]

        # Initializing table los_info_table
        los_table = Table()
        los_table['z_center'] = np.ones(n_hdu) * z_center[j]
        los_table['ra'] = np.ones(n_hdu) * np.nan
        los_table['dec'] = np.ones(n_hdu) * np.nan
        los_table['TARGETID'] = np.zeros(n_hdu, dtype='>i8')
        los_table['delta_los'] = np.zeros((n_hdu, len(wavelength_ref_zbin))) # len of delta_los and wavelength is = len of wavelength_ref_zbin before masking pixels due to skylines since deltas will be patched
        los_table['wavelength'] = np.zeros((n_hdu, len(wavelength_ref_zbin)))
        if include_snr_reso:
            los_table['MEANRESO'] = np.zeros(n_hdu)
            los_table['MEANSNR'] = np.zeros(n_hdu)
        los_table_list.append(los_table)

    # Looping over hdus    
    for i in range(n_hdu):
        if i%100==0 :
            print(delta_file_name,": HDU",i,"/",n_hdu)

        delta_i_header = delta_file[i+1].read_header()
        delta_ID = delta_i_header['TARGETID']
        
        if delta_ID in qso_tid:
            # Reading data
            try:
                delta_los = delta_file[i+1]['DELTA'][:].astype(float) 
            except:
                delta_los = delta_file[i+1]['DELTA_BLIND'][:].astype(float) 
            wavelength = delta_file[i+1]['LAMBDA'][:].astype(float)

        # Looping over zbins 
        for j in range(len(z_center)): # los_info_table_list must have n_zbins lists
            # Selecting wavelength_ref corresponding to zbin
            wavelength_ref_zbin = wavelength_ref.copy() # Where wavelength_ref is one for all zbins
            mask_wavelength_ref_zbin = (wavelength_ref_zbin > lambda_min[j]) & (wavelength_ref_zbin < lambda_max[j])
            wavelength_ref_zbin = wavelength_ref_zbin[mask_wavelength_ref_zbin]

            # Masking pixels in wavelength_ref_zbin if masks are not none
            if (lambda_pixelmask_min is not None) & (lambda_pixelmask_max is not None):
                if (len(lambda_pixelmask_min)==len(lambda_pixelmask_max)):
                    for i_pixelmask in range(len(lambda_pixelmask_min)): # or lambda_pixelmask_max, it's the same
                        pixels_to_mask = (wavelength_ref_zbin > lambda_pixelmask_min[i_pixelmask]) & (wavelength_ref_zbin < lambda_pixelmask_max[i_pixelmask])
                        wavelength_ref_zbin = wavelength_ref_zbin[~pixels_to_mask]
                else:
                    print('lambda_pixelmask_min and lambda_pixelmask_max have different lengths, therefore the mask is not taken into acccount')
            
            # This part is to check if the delta must be included in the redshift bin or not:
            # Checking if LAMBDA.min < lambda_min & LAMBDA.max > lambda_max
            if (wavelength.min() < lambda_min[j]) and (wavelength.max() > lambda_max[j]):
                # Define wavelength mask
                mask_wavelength = (wavelength > lambda_min[j]) & (wavelength < lambda_max[j])

                # Checking that the masked wavelength and wavelength_ref have the same shape
                # otherwise it means that there are masked pixels and we don't want to consider this delta in the calculation
                if len(wavelength[mask_wavelength]) == len(wavelength_ref_zbin):
                    if np.allclose(wavelength[mask_wavelength], wavelength_ref_zbin):
                        los_table_list[j][i]['ra'] = delta_i_header['RA'] * 180 / np.pi  # must convert rad --> dec.
                        los_table_list[j][i]['dec'] = delta_i_header['DEC'] * 180 / np.pi
                        los_table_list[j][i]['TARGETID'] = delta_ID

                        # Here we must patch wavelength and delta_los before adding to the table
                        wavelength_patched, delta_los_patched = patch_deltas(wavelength[mask_wavelength], delta_los[mask_wavelength], delta_lambda)

                        # Adding patched arrays to table
                        los_table_list[j][i]['delta_los'] = delta_los_patched
                        los_table_list[j][i]['wavelength'] = wavelength_patched

                        # Adding MEANSNR and MEANRESO of LOS to los_table
                        if include_snr_reso:
                            if ('MEANSNR' in delta_i_header) and ('MEANRESO' in delta_i_header):
                                los_table_list[j][i]['MEANSNR'] = delta_i_header['MEANSNR']
                                los_table_list[j][i]['MEANRESO'] = delta_i_header['MEANRESO']
                            else:
                                print('Warning, no MEANSNR/MEANRESO in delta header.')
                    else:
                        print('Warning')  # should not happen in principle
                else:
                    # print('Masked LOS')
                    n_masked += 1

    # Closing delta_file
    delta_file.close()

    # Removing rows belonging to LOS with masks that were discarded
    for j in range(len(z_center)):
        mask_los_used = ~np.isnan(los_table_list[j]['ra'])
        los_table_list[j] = los_table_list[j][mask_los_used]
        print("DESI delta file", delta_file_name,":",len(los_table_list[j]),"LOS used")
        if n_masked>0:
            print("    (",n_masked,"LOS not used presumably due to masked pixels)")

    return los_table_list


def get_los_table_desi(qso_cat, deltas_dir, lambda_min, lambda_max, z_center, outputdir, outputfilename, lambda_pixelmask_min=None, lambda_pixelmask_max=None, ncpu='all', include_snr_reso=False):
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
    
    z_center: Float or array of floats
    Value of z center per redshift bin
    PS: lambda_min, lambda_max and z_center must have the same length
    
    outputdir: string, default None
    Write LOS table to file
    
    outputfilename: string, default None
    Name of the file
    
    lambda_pixelmask_min: Float or array of floats
    Minimum of wavelength intervals to be masked. This will be applied not on data, but on wavelength_ref
    
    lambda_pixelmask_max: Float or array of floats
    Maximum of wavelength intervals to be masked. This will be applied not on data, but on wavelength_ref

    ncpu: int or 'all'
    For multiprocessing.Pool
    
    include_snr_reso: bool, default False
    If set, includes MEANRESO and MEANSNR from the delta's headers, to the los_table

    Return:
    -------
    los_allfiles_allz: Table or list of tables where each corresponds to one redshift bin
    In each of the tables: each row corresponds to a QSO, containing [ra, dec, delta_los, wavelength, TARGETID] and possibly [MEANSNR, MEANRESO]
    
    """

    searchstr = '*'
    deltafiles = glob.glob(os.path.join(deltas_dir, f"delta{searchstr}.fits.gz"))

    if ncpu=='all':
        ncpu = multiprocessing.cpu_count()

    print("Nb of delta files:", len(deltafiles))
    print("Number of cpus:", multiprocessing.cpu_count())

    # with Pool(ncpu) as pool:
    #     output_get_desi_deltas_singlefile = pool.starmap(
    #         get_desi_deltas_singlefile,
    #         [[f, qso_cat, lambda_min, lambda_max, include_snr_reso] for f in deltafiles]
    #     )
    
    # This does the parallelization over all the delta files in deltas_dir, 
    # and the output_get_los_info_singlefile is a list of los_info_table(s), each table corresponding to one z bin

    with Pool(ncpu) as pool:
        output_get_desi_los_singlefile = pool.starmap(
            get_desi_los_singlefile,
            [[f, qso_cat, lambda_min, lambda_max, z_center, lambda_pixelmask_min, lambda_pixelmask_max, include_snr_reso] for f in deltafiles]
        )
        
    for x in output_get_desi_los_singlefile:
        if x is None: print("output of get_desi_los_singlefile is None")  # should not happen in principle

    # for x in output_get_desi_deltas_singlefile:
    #     if x is None: print("output of get_desi_deltas_singlefile is None")  # should not happen in principle
    
    los_allfiles_allz = [] # A list of tables each corresponding to one redshift bin
    for j in range(len(z_center)):
        output_get_desi_los_singlefile_onez = [x[j] for x in output_get_desi_los_singlefile if x[j] is not None] # Here it is a list of tables
        los_table_onez = vstack([output_get_desi_los_singlefile_onez[i] for i in range(len(output_get_desi_los_singlefile_onez))])
        # Writing table for one z bin
        outputfile = os.path.join(outputdir, 'output_'+str(z_center[j]), outputfilename)
        los_table_onez.write(outputfile)

        los_allfiles_allz.append(los_table_onez)

    return los_allfiles_allz

#     output_get_desi_deltas_singlefile = [x for x in output_get_desi_deltas_singlefile if x is not None]
#     los_table = vstack([output_get_desi_deltas_singlefile[i] for i in range(len(output_get_desi_deltas_singlefile))])

#     if outputfile is not None:
#         los_table.write(outputfile)

#     return los_table


def patch_deltas(lambda_array, delta_array, delta_lambda):
    """ This function patches deltas of masked LOS by zeros and the correspinding wavelength by the value of the wavelength of the masked pixel.
    Having a regular grid is required for the fft analysis in the P_cross computation.
    PS: THIS CODE IS COPIED FROM PICCA!!

    Arguments:
    ----------
    lambda_array: Array
    Array of LOS wavelength. It can be both linear or log spaced.

    delta_array: Array
    Array of LOS delta.

    delta_lambda: Float
    Pixel size. Equal to 0.8 AA in the case of DESI.

    Return:
    -------
    lambda_array_new: Array
    Array of LOS wavelength after patching.

    delta_array_new: Array
    Array of LOS delta after patching.
    """

    lambda_array_index = lambda_array.copy()
    lambda_array_index -= lambda_array[0]
    lambda_array_index /= delta_lambda
    lambda_array_index += 0.5
    lambda_array_index = np.array(lambda_array_index, dtype=int)
    index_all = range(lambda_array_index[-1] + 1)
    index_ok = np.in1d(index_all, lambda_array_index)

    # Patching delta
    delta_array_new = np.zeros(len(index_all))
    delta_array_new[index_ok] = delta_array

    # Patching lambda
    lambda_array_new = np.array(index_all, dtype=float)
    lambda_array_new *= delta_lambda
    lambda_array_new += lambda_array[0]

    return lambda_array_new, delta_array_new

