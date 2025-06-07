""" Functions to read DESI delta files (Y1), creating a los_table to be used later by the pipeline """

import numpy as np
import os, sys, glob
import multiprocessing
from multiprocessing import Pool
import fitsio
from astropy.io import fits
from astropy.table import Table, vstack
import scipy
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# from .tools import SPEED_LIGHT
sys.path.insert(0, os.environ['HOME']+'/Software/LyaP3D')
from tools import SPEED_LIGHT, LAMBDA_LYA, DEFAULT_MINWAVE_SKYMASK, DEFAULT_MAXWAVE_SKYMASK, DEFAULT_DWAVE_SKYMASK


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


def DESI_resolution_correction(z_measurement, k_parallel):
    # Using the final P1D, but without BAL AI cuts
    P1D_Y1_file = fits.open('/global/cfs/cdirs/desi/science/lya/y1-p1d/fft_measurement/v0/data/p1d_out/p1d_lya_noisepipeline_linesDESIEDR_catiron_highz_20240305_nobal_BI_dlairon_highz_CNN_GP_naimcuts_balNone/pk1d_SNRcut1/mean_Pk1d_fit_snr_medians.fits.gz')
    P1D_Y1 = P1D_Y1_file[1].data
    zbins = np.unique(P1D_Y1['zbin'])[:11]
    n_z = len(zbins)
    n_k = len(P1D_Y1['meank'][P1D_Y1['zbin'] == zbins[0]])
    
    # Creating ResoCorr_table from P1D file for z and k of P1D
    ResoCorr_table = Table()
    ResoCorr_table['Z'] = np.zeros(n_z)
    ResoCorr_table['K'] = np.zeros((n_z, n_k))
    ResoCorr_table['Reso_corr'] = np.zeros((n_z, n_k))
    for iz, z in enumerate(zbins): # z bin
        select_z = (P1D_Y1['zbin'] == z)
        K = P1D_Y1['meank'][select_z]
        Reso_corr = P1D_Y1['meancor_reso'][select_z]
    
        # Filling table
        ResoCorr_table['Z'][iz] = z
        ResoCorr_table['K'][iz] = K
        ResoCorr_table['Reso_corr'][iz] = Reso_corr

    # Interpolating the resolution correction
    ## Build a big list of (z,k) points and corresponding correction values:
    points_list = []
    values_list = []
    for z, k_arr, corr_arr in zip(ResoCorr_table['Z'], ResoCorr_table['K'], ResoCorr_table['Reso_corr']):
        # stack z with each k in this row
        pts = np.column_stack((np.full_like(k_arr, z), k_arr))  # shape (n_k_row, 2)
        points_list.append(pts)
        values_list.append(corr_arr)
    points = np.vstack(points_list)       # shape (total_points, 2)
    values = np.concatenate(values_list)  # shape (total_points,)

    ## Create a reusable 2D scattered-data interpolator:
    # ResoCorr_interpolator = LinearNDInterpolator(points, values, fill_value=np.nan)
    ResoCorr_interpolator = NearestNDInterpolator(points, values) # avoids nan when extrapolated

    ## Interpolation
    z_arr = np.full_like(k_parallel, z_measurement)
    z_min, z_max = ResoCorr_table['Z'].min(), ResoCorr_table['Z'].max()
    z_arr = np.clip(z_arr, z_min, z_max)  # if z_measurement outside the z domain in which I got my interpolator, it will take the smallest z, will be happening only for z=2.15
    reso_corr_output = ResoCorr_interpolator(z_arr, k_parallel)

    return reso_corr_output


def subtract_SB_power_spectrum(pcross_table, pcross_sb_table):
    # Adding pcross_sb and error_pcross_sb columns to pcross_table
    pcross_table['power_spectrum_SB'] = pcross_sb_table['corrected_power_spectrum']
    pcross_table['error_power_spectrum_SB'] = pcross_sb_table['error_corrected_power_spectrum']

    # Adding the SB_corrected_power_spectrum, error_SB_corrected_power_spectrum, covmat_SB_corrected_power_spectrum columns

    ## Lengths definitions
    ang_sep_bin_centers = pcross_table['ang_sep_bin_centers']
    n_ang_sep_bins = len(ang_sep_bin_centers)
    k_parallel = pcross_table['k_parallel'][0]
    Nk = len(k_parallel)

    ## Columns initialization
    pcross_table['SB_corrected_power_spectrum'] = np.zeros((n_ang_sep_bins, Nk))
    pcross_table['error_SB_corrected_power_spectrum'] = np.zeros((n_ang_sep_bins, Nk))
    pcross_table['covmat_SB_corrected_power_spectrum'] = np.zeros((n_ang_sep_bins, Nk, Nk))

    ## Filling the table
    for i_ang_sep, ang_sep in enumerate(ang_sep_bin_centers):
        pcross_table['SB_corrected_power_spectrum'][i_ang_sep] = pcross_table['corrected_power_spectrum'][i_ang_sep] - pcross_table['power_spectrum_SB'][i_ang_sep]
        var_SB_corrected_power_spectrum = pcross_table['error_corrected_power_spectrum'][i_ang_sep]**2 + pcross_table['error_power_spectrum_SB'][i_ang_sep]**2
        pcross_table['error_SB_corrected_power_spectrum'][i_ang_sep] = np.sqrt(var_SB_corrected_power_spectrum)
        pcross_table['covmat_SB_corrected_power_spectrum'][i_ang_sep,:,:] = np.diag(var_SB_corrected_power_spectrum)

    return pcross_table


## Copied from picca.pk1d but adapted to my Px measurement case
def skyline_mask_matrices_desi(lmin, lmax, skyline_mask_file,
                               minwave=DEFAULT_DWAVE_SKYMASK,
                               maxwave=DEFAULT_MAXWAVE_SKYMASK,
                               dwave=DEFAULT_DWAVE_SKYMASK):
    """ Compute matrices to correct for the masking effect on FFTs,
    when chunks are defined with the `--parts-in-redshift` option in picca_Pk1D.
    Only implemented for DESI data, ie delta lambda = 0.8 and no rebinning in the deltas.

    Arguments
    ---------
    lmin, lmax of my bin

    skyline_mask_file: str
    Name of file containing the list of skyline masks

    minwave, maxwave, dwave: float
    Parameter defining the wavelength grid used for the skymask. Should be consistent with
    arguments lambda min, lambda max, delta lambda used in picca_delta_extraction.py.

    Return
    ------
    inv_matrix of the correction to be directly multiplied by the Pcross
    """
    skyline_list = np.genfromtxt(skyline_mask_file,
                                 names=('type', 'wave_min', 'wave_max', 'frame'))
    ref_wavegrid = np.arange(minwave, maxwave, dwave)
    wave = ref_wavegrid[ (ref_wavegrid > lmin) & (ref_wavegrid < lmax) ]
    npts = len(wave)
    print(npts)
    skymask = np.ones(npts)
    selection = ( (skyline_list['wave_min']<=lmax) & (skyline_list['wave_min']>=lmin)
                ) | ( (skyline_list['wave_max']<=lmax) & (skyline_list['wave_max']>=lmin) ) # check if it should be modified
    print('List of skylines', skyline_list[selection])

    for skyline in skyline_list[selection]:
        skymask[(wave>skyline['wave_min']) & (wave<skyline['wave_max'])] = 0
    skymask_tilde = np.fft.fft(skymask)/npts
    mask_matrix = np.zeros((npts, npts))
    for j in range(npts):
        for l in range(npts):
            index_mask = j-l if j>=l else j-l+npts
            mask_matrix[j, l] = (
                skymask_tilde[index_mask].real ** 2
                + skymask_tilde[index_mask].imag ** 2
            )
    try:
        inv_matrix = np.linalg.inv(mask_matrix)
    except np.linalg.LinAlgError:
        userprint(
            """Warning: cannot invert sky mask matrix """
        )
        userprint("No correction will be applied for this bin")
        inv_matrix = np.eye(npts)

    return inv_matrix


def DESI_skyline_correction(pcross_table, lmin, lmax, skyline_mask_file):

    # Computing the skyline mask correction matrix
    skyline_mask_correction_matrix = skyline_mask_matrices_desi(lmin, lmax, skyline_mask_file)

    # Adding columns to power_spectrum_table

    ## Lengths definitions
    ang_sep_bin_centers = pcross_table['ang_sep_bin_centers']
    n_ang_sep_bins = len(ang_sep_bin_centers)
    k_parallel = pcross_table['k_parallel'][0]
    Nk = len(k_parallel)

    ## Cutting the matrix
    skyline_mask_correction_matrix = np.copy(skyline_mask_correction_matrix[0:Nk, 0:Nk])

    ## Columns initialization
    pcross_table['Full_corrected_power_spectrum'] = np.zeros((n_ang_sep_bins, Nk))
    pcross_table['error_Full_corrected_power_spectrum'] = np.zeros((n_ang_sep_bins, Nk))
    pcross_table['covmat_Full_corrected_power_spectrum'] = np.zeros((n_ang_sep_bins, Nk, Nk))

    ## Filling the table
    for i_ang_sep, ang_sep in enumerate(ang_sep_bin_centers):
        pcross_table['Full_corrected_power_spectrum'][i_ang_sep] = skyline_mask_correction_matrix @ pcross_table['SB_corrected_power_spectrum'][i_ang_sep]
        pcross_table['covmat_Full_corrected_power_spectrum'][i_ang_sep,:,:] = skyline_mask_correction_matrix @ pcross_table['covmat_SB_corrected_power_spectrum'][i_ang_sep,:,:] @ skyline_mask_correction_matrix.T
        pcross_table['error_Full_corrected_power_spectrum'][i_ang_sep] = np.sqrt(np.diag(pcross_table['covmat_Full_corrected_power_spectrum'][i_ang_sep,:,:]))

    return pcross_table
