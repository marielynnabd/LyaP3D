""" This module provides a set of functions to get the cross power spectrum 

    P_cross(z, delta_theta, k_parallel) from quasar's Lyman-alpha forests """
 

import numpy as np
import sys, os
import glob
from astropy.table import Table, vstack
from multiprocessing import Pool
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

sys.path.insert(0, os.environ['HOME']+'/Software')
from LyaP3D.tools import SPEED_LIGHT, LAMBDA_LYA, find_bin_edges, convert_units, fitfunc_std_fftproduct, fitfunc_variance_pk1d
from LyaP3D.eBOSS_dr16_analysis import boss_resolution_correction
from LyaP3D.pairs_computation import compute_pairs
from LyaP3D.DESI_Y1_analysis import DESI_resolution_correction


def compute_mean_p_cross(all_los_table, los_pairs_table, ang_sep_bin_edges, data_type, units, weight_method='no_weights',
                         min_snr_p_cross=None, max_resolution_p_cross=None,
                         resolution_correction=False, reshuffling=False, with_covmat=True):
    """ This function computes mean power spectrum for pairs with angular separations > 0 (called cross power spectrum):
          - Takes mock and corresponding los_pairs_table
          _ Computes cross power spectrum for each pair
          - Averages over all of them to get one p_cross(k_parallel) per angular separation bin
    
    Arguments:
    ----------
    all_los_table: Table
    Mock.
    
    los_pairs_table: Table
    Each row corresponds to the indices of the pixels forming the pair, and the angular separation between them.
    
    ang_sep_bin_edges: Array of floats
    Edges of the angular separation bins we want to use.
    
    min_snr_p_cross: Float, Default is None
    The value of minimum snr desired.
    
    max_resolution_cross: Float, Default is None
    The value of maximum resolution desired.
    
    resolution_correction: Boolean, Default is False
    If we want to apply a resolution correction or not, this is only in the real data case.
    
    reshuffling: Boolean, Default is False
    This is done in case we want to compute a cross spectrum wihout signal, ie. by correlating pixels that aren't correlated.
    
    with_covmat: Boolean, Default is True
    Switch on/off covariance matrix computation.

    data_type: String, Options: 'mocks', 'DESI', 'eBOSS'
    The type of data set on which we want to run the cross power spectrum computation.
        - In the case of mocks: The cross power spectrum will be computed in [Angstrom] by default,
        because when we draw LOS to create mocks, wavelength = (1 + refshift) * lambda_lya [Angstrom].
        If another unit is desired, this must be specified in the argument units.
        PS: In the case of mocks: min_snr_p_cross, max_resolution_cross, and resolution_correction must be set to default !
        - In the case of real data: The cross power spectrum will be first computed unitless,
        because wavelength = LOGLAM, therefore it is mandatory to multiply it my a factor c, and the output will be in [km/s].
    
    units: String, Options: 'Mpc/h', 'Angstrom', 'km/s'.
    Units in which to compute power spectrum.
    
    weight_method: String
    3 possible options:
        'no_weights': Compute mean cross power spectrum without weights (Used for mocks, or for data when min_snr_p_cross is not None).
        'forest_snr': Compute mean cross power spectrum with weights = w_i * w_j with w_i = SNR_i, with i and j being the LOS forming each pair.
        'fit_forest_snr': Compute mean P1D with weights = w_i * w_j with w_i and w_j estimated by fitting dispersion of (delta_i * conj(delta_i)) vs SNR.
                          PS: This weight_method doesn't allow forests having snr < 1, if such data is being used, must do min_snr_p_cross = 1.

    Return:
    -------
    p_cross_table: Table
    Each row corresponds to the p_cross in one angular separation bin.
    """

    ## TODO: cosmo should be args
    # Computing cosmo used for conversions
    Omega_m = 0.3153
    h = 0.7
    cosmo = FlatLambdaCDM(H0=100*h, Om0=Omega_m)
    
    # Centers of angular separation bins
    ang_sep_bin_centers = np.around((ang_sep_bin_edges[1:] + ang_sep_bin_edges[:-1]) / 2, 5)
    n_ang_sep_bins = len(ang_sep_bin_centers)

    print('Pcross computation')
    
    # Parameters definitions
    delta_lambda = all_los_table['wavelength'][0][1] - all_los_table['wavelength'][0][0]
    mean_wavelength = np.mean(all_los_table['wavelength'][0])
    z = mean_wavelength / LAMBDA_LYA - 1
    
    ## if we're in eBOSS case, we have log(lambda) and not lambda so a conversion is required [km/s]
    if data_type == 'eBOSS':
        mean_wavelength = np.mean(10.**(all_los_table['wavelength'][0]))
        z = mean_wavelength / LAMBDA_LYA - 1
        delta_lambda *= SPEED_LIGHT * np.log(10.)

    print("Px: z=", z)

    delta_los = all_los_table['delta_los']
    Npix = len(delta_los[0])
    print('Npix', Npix)

    # Applying snr mask
    if min_snr_p_cross is not None:
        print('snr cut applied')
        snr_los1 = all_los_table['MEANSNR'][ los_pairs_table['index_los1'] ]
        snr_los2 = all_los_table['MEANSNR'][ los_pairs_table['index_los2'] ]
        snr_mask = (snr_los1 > min_snr_p_cross) & (snr_los2 > min_snr_p_cross)
        if weight_method != 'no_weights':
            print('Warning, both snr cut and weighting will be applied on p_cross')
    else:
        snr_mask = np.ones(len(los_pairs_table), dtype=bool)

    # Applying resolution mask    
    if max_resolution_p_cross is not None:
        if data_type == 'mocks':
            print('Warning, no resolution cut will be applied on p_cross since it is a mock case')
            reso_mask = np.ones(len(los_pairs_table), dtype=bool)
        elif data_type == 'eBOSS':
            print("reso cut applied")
            reso_los1 = all_los_table['MEANRESOLUTION'][ los_pairs_table['index_los1'] ]
            reso_los2 = all_los_table['MEANRESOLUTION'][ los_pairs_table['index_los2'] ]
            reso_mask = (reso_los1 < max_resolution_p_cross) & (reso_los2 < max_resolution_p_cross)
        elif data_type == 'DESI':
            print("reso cut applied")
            reso_los1 = all_los_table['MEANRESO'][ los_pairs_table['index_los1'] ]
            reso_los2 = all_los_table['MEANRESO'][ los_pairs_table['index_los2'] ]
            reso_mask = (reso_los1 < max_resolution_p_cross) & (reso_los2 < max_resolution_p_cross)
    else:
        reso_mask = np.ones(len(los_pairs_table), dtype=bool)

    los_pairs_table = los_pairs_table[ (snr_mask & reso_mask) ]
    
    # This is done if we want to compute a P_cross without signal ie. correlating pixels that aren't correlated
    if reshuffling==True:
        los_pairs_table['index_los2'] = np.random.permutation(los_pairs_table['index_los2'])

    # FFT of deltas
    fft_delta = np.fft.rfft(delta_los)
    Nk = fft_delta.shape[1] # bcz of rfft, otherwise Nk = Npix if we do fft
    print('Nk', Nk)
    
    # Initializing p_cross_table
    p_cross_table = Table()
    p_cross_table['ang_sep_bin_centers'] = np.array(ang_sep_bin_centers)
    p_cross_table['mean_ang_separation'] = np.zeros(n_ang_sep_bins)
    p_cross_table['N'] = np.zeros(n_ang_sep_bins)
    p_cross_table['k_parallel'] = np.zeros((n_ang_sep_bins, Nk))
    p_cross_table['mean_power_spectrum'] = np.zeros((n_ang_sep_bins, Nk))
    p_cross_table['error_power_spectrum'] = np.zeros((n_ang_sep_bins, Nk))
    p_cross_table['resolution_correction'] = np.zeros((n_ang_sep_bins, Nk))
    p_cross_table['corrected_power_spectrum'] = np.zeros((n_ang_sep_bins, Nk))
    p_cross_table['error_corrected_power_spectrum'] = np.zeros((n_ang_sep_bins, Nk))
    if with_covmat:
        p_cross_table['covmat_power_spectrum'] = np.zeros((n_ang_sep_bins, Nk, Nk))
        p_cross_table['covmat_corrected_power_spectrum'] = np.zeros((n_ang_sep_bins, Nk, Nk))

    # p_cross computation
    for i_ang_sep, ang_sep in enumerate(ang_sep_bin_edges[:-1]):
        print('angular separation bin edges', ang_sep_bin_edges[i_ang_sep], ang_sep_bin_edges[i_ang_sep+1], 
              'corresponding center', ang_sep_bin_centers[i_ang_sep])

        select = (los_pairs_table['ang_separation'] > ang_sep_bin_edges[i_ang_sep]) & (
                los_pairs_table['ang_separation'] <= ang_sep_bin_edges[i_ang_sep+1])

        N_pairs = np.sum(select)
        mean_ang_separation = np.mean(los_pairs_table['ang_separation'][select])
        p_cross_table['N'][i_ang_sep] = N_pairs
        p_cross_table['mean_ang_separation'][i_ang_sep] = mean_ang_separation

        index_los1 = los_pairs_table['index_los1'][select]
        index_los2 = los_pairs_table['index_los2'][select]

        # Computation of p_cross in Ansgtrom units
        p_cross = (fft_delta[index_los1] * np.conj(fft_delta[index_los2])) * delta_lambda / Npix # same units as delta_lambda
        k_parallel = 2 * np.pi * np.fft.rfftfreq(Npix, delta_lambda) # same units as delta_lambda

        if data_type == 'eBOSS':
            p_cross = convert_units(p_cross, 'km/s', units, z, inverse_units=False)
            k_parallel = convert_units(k_parallel, 'km/s', units, z, inverse_units=True)
        else: # This is the case of 'mocks' and 'DESI' where pcross is first computed in Angstrom
            p_cross = convert_units(p_cross, 'Angstrom', units, z, inverse_units=False)
            k_parallel = convert_units(k_parallel, 'Angstrom', units, z, inverse_units=True)

        # Preparing for the weighting scheme
        if weight_method != 'no_weights': # Either 'forest_snr' or 'fit_forest_snr'
            # These are the snr of the LOS that fall into this bin and contribute to the measurement
            snr_los1 = all_los_table['MEANSNR'][index_los1]
            snr_los2 = all_los_table['MEANSNR'][index_los2]

        if weight_method == 'fit_forest_snr':
            fft_product_los1 = (fft_delta[index_los1].real**2 + fft_delta[index_los1].imag**2) * delta_lambda / Npix
            fft_product_los2 = (fft_delta[index_los2].real**2 + fft_delta[index_los2].imag**2) * delta_lambda / Npix
            if data_type == 'eBOSS':
                fft_product_los1 = convert_units(fft_product_los1, 'km/s', units, z, inverse_units=False)
                fft_product_los2 = convert_units(fft_product_los2, 'km/s', units, z, inverse_units=False)
            else: # This is the case of 'mocks' and 'DESI' where pcross is first computed in Angstrom
                fft_product_los1 = convert_units(fft_product_los1, 'Angstrom', units, z, inverse_units=False)
                fft_product_los2 = convert_units(fft_product_los2, 'Angstrom', units, z, inverse_units=False)

        # resolution correction computation
        if resolution_correction == True:
            if data_type == 'eBOSS':
                delta_v = delta_lambda
                resolution_los1 = all_los_table['MEANRESOLUTION'][index_los1]
                resolution_los2 = all_los_table['MEANRESOLUTION'][index_los2]
                resgrid, kpargrid = np.meshgrid(resolution_los1, k_parallel, indexing='ij')
                resolution_correction_los1 = boss_resolution_correction(resgrid, kpargrid, delta_v)
                resgrid, kpargrid = np.meshgrid(resolution_los2, k_parallel, indexing='ij')
                resolution_correction_los2 = boss_resolution_correction(resgrid, kpargrid, delta_v)
                resolution_correction_p_cross = resolution_correction_los1 * resolution_correction_los2
            elif data_type == 'DESI': # Interpolated correction from that of P1D, code to be improved
                resolution_correction_p_cross = DESI_resolution_correction(z, k_parallel)
        else:
            resolution_correction_p_cross = np.ones((len(index_los1), len(k_parallel)))

        # mean_p_cross computation
        mean_p_cross = np.zeros(Nk)
        mean_resolution_correction_p_cross = np.zeros(Nk)
        error_p_cross = np.zeros(Nk)

        if weight_method == 'fit_forest_snr':
            weights_pairs = np.ones((len(index_los1), len(k_parallel)))
        else:
            weights_pairs = None

        for i in range(Nk):
            p_cross_array = np.array(p_cross[:,i]) # This contains the values of p_cross we want to average at one k_parallel
            # Applying weighting scheme
            if weight_method == 'fit_forest_snr':
                # Selecting the points corresponding to the same k_parallel value from the individual fft products of the LOS 
                fft_product_los1_array = np.array(fft_product_los1[:,i])
                fft_product_los2_array = np.array(fft_product_los2[:,i])
                # Computing the fit of the standard deviation these points
                snr_bin_edges = np.arange(1, 10 + 1, 1)
                snr_bins = (snr_bin_edges[:-1] + snr_bin_edges[1:]) / 2
                standard_dev_los1, _, _ = binned_statistic(snr_los1, fft_product_los1_array, statistic="std", bins=snr_bin_edges)
                standard_dev_los2, _, _ = binned_statistic(snr_los2, fft_product_los2_array, statistic="std", bins=snr_bin_edges)
                mask_nan_in_std_los1 = ~np.isnan(standard_dev_los1)
                mask_nan_in_std_los2 = ~np.isnan(standard_dev_los2)
                standard_dev_los1 = standard_dev_los1[mask_nan_in_std_los1]
                standard_dev_los2 = standard_dev_los2[mask_nan_in_std_los2]
                snr_bins_los1 = snr_bins[mask_nan_in_std_los1]
                snr_bins_los2 = snr_bins[mask_nan_in_std_los2]
                coef_los1, *_ = curve_fit(fitfunc_std_fftproduct, snr_bins_los1, standard_dev_los1, bounds=(0, np.inf))
                coef_los2, *_ = curve_fit(fitfunc_std_fftproduct, snr_bins_los2, standard_dev_los2, bounds=(0, np.inf))
                # Fixing high and low snr values of data
                snr_los1[snr_los1 > 10] = 10
                snr_los1[snr_los1 < 1.01] = 1.01
                snr_los2[snr_los2 > 10] = 10
                snr_los2[snr_los2 < 1.01] = 1.01
                # Estimated std
                standard_dev_los1_estimated = fitfunc_std_fftproduct(snr_los1, *coef_los1)
                standard_dev_los2_estimated = fitfunc_std_fftproduct(snr_los2, *coef_los2)
                # Weights
                weights_los1 = 1 / standard_dev_los1_estimated
                weights_los2 = 1 / standard_dev_los2_estimated
                weights_p_cross_array = weights_los1 * weights_los2
                weights_pairs[:,i] = weights_p_cross_array
                # Computing weighted average
                mean_p_cross[i] = np.average(p_cross_array.real, weights=weights_p_cross_array)
                # error_p_cross[i] = np.sqrt(1.0 / np.sum(weights_p_cross_array))  # wrong if weights are not optimal
                # error bar => (B.8) from arxiv:2505.09493
                w2 = weights_p_cross_array**2
                sw, sw2 = np.sum(weights_p_cross_array), np.sum(w2)
                avg2 = np.average((p_cross_array.real)**2, weights=w2)
                variance_pcross = (avg2 - (mean_p_cross[i])**2) / (sw**2/sw2 - 1)
                error_p_cross[i] = np.sqrt(variance_pcross)
            elif weight_method == 'forest_snr':
                weights_p_cross_array = snr_los1 * snr_los2
                # Computing weighted average
                mean_p_cross[i] = np.average(p_cross_array.real, weights=weights_p_cross_array)
                error_p_cross[i] = np.sqrt(1.0 / np.sum(weights_p_cross_array))
            else:
                mean_p_cross[i] = np.mean(p_cross_array.real)
                error_p_cross[i] = np.std(p_cross_array.real) / np.sqrt(N_pairs - 1)
            try:    
                mean_resolution_correction_p_cross[i] = np.mean(resolution_correction_p_cross[:,i])
            except: # Because in the case of DESI it's the same correction for all LOS
                mean_resolution_correction_p_cross[i] = resolution_correction_p_cross[i]

        # Smoothing errors:
        smoothed_error_p_cross = savgol_filter(error_p_cross, window_length=15, polyorder=1)

        # Filling table
        p_cross_table['k_parallel'][i_ang_sep, :] = k_parallel
        p_cross_table['mean_power_spectrum'][i_ang_sep, :] = mean_p_cross  
        p_cross_table['error_power_spectrum'][i_ang_sep, :] = smoothed_error_p_cross
        p_cross_table['resolution_correction'][i_ang_sep, :] = mean_resolution_correction_p_cross
        p_cross_table['corrected_power_spectrum'][i_ang_sep, :] = mean_p_cross / mean_resolution_correction_p_cross
        p_cross_table['error_corrected_power_spectrum'][i_ang_sep, :] = error_p_cross / mean_resolution_correction_p_cross

        if with_covmat:  #- covariance matrix:
            if weight_method == 'fit_forest_snr':
                # Eqn (B.7) from arxiv:2505.09493, same notation
                covmat = np.zeros((Nk, Nk))
                for i in range(Nk):
                    v, X = weights_pairs[:,i], np.array(p_cross[:,i].real)
                    sv = np.sum(v)
                    for j in range(Nk):
                        w, Y = weights_pairs[:,j], np.array(p_cross[:,j].real)
                        vw = v*w
                        sw, s2 = np.sum(w), np.sum(vw)
                        avg2 = np.average(X*Y, weights=vw)
                        covmat[i,j] = (avg2 - mean_p_cross[i]*mean_p_cross[j]) / (sv*sw/s2 - 1)
            else:
                covmat = np.cov(p_cross.real, rowvar=False)
            p_cross_table['covmat_power_spectrum'][i_ang_sep, :, :] = covmat
            p_cross_table['covmat_corrected_power_spectrum'][i_ang_sep, :, :] = covmat / np.outer(
                mean_resolution_correction_p_cross, mean_resolution_correction_p_cross)  # cov_ij / Wi*Wj

    return p_cross_table


def compute_mean_p_auto(all_los_table, data_type, units, weight_method='no_weights', 
                        min_snr_p_auto=None, max_resolution_p_auto=None, resolution_correction=True, 
                        p_noise=0, with_covmat=True):
    """ This function computes mean power spectrum for angular separation = 0 (Lya forest and itself, called auto power spectrum):
          - Takes all_los_table
          - Computes auto power spectrum for each LOS 
          - Averages over all of them to get one p_auto(k_parallel) at ang_sep_bin = 0
          PS: This function has to be updated: Reso Corr DESI missing, error bars... missing all modifs added to compute_mean_p_cross on 4th on June 2025!

    Arguments:
    ----------
    all_los_table: Table
    Mock.

    min_snr_p_auto: Float, Default is None
    The value of minimum snr desired.

    max_resolution_p_auto: Float, Default is None
    The value of maximum resolution desired.

    resolution_correction: Boolean, Default is True
    If we want to apply a resolution correction or not.

    p_noise: Float, Default is 0
    Value of Pnoise that we want to substract from p_auto. This is only for p_auto, in the case of p_cross, noise effect is zero.

    with_covmat: Boolean, Default is True
    Switch on/off covariance matrix computation

    data_type: String, Options: 'mocks', 'DESI', 'eBOSS'
    The type of data set on which we want to run the auto power spectrum computation.
        - In the case of mocks: The auto power spectrum will be computed in [Angstrom] by default,
        because when we draw LOS to create mocks, wavelength = (1 + refshift) * lambda_lya [Angstrom].
        If another unit is desired, this must be specified in the argument units.
        PS: In the case of mocks: min_snr_p_auto, max_resolution_auto, and resolution_correction must be set to default !
        - In the case of real data: The auto power spectrum will be first computed unitless,
        because wavelength = LOGLAM, therefore it is mandatory to multiply it my a factor c, and the output will be in [km/s].

    units: String, Options: 'Mpc/h', 'Angstrom', 'km/s'.
    Units in which to compute power spectrum.

    weight_method: String
    3 possible options:
        'no_weights': Compute mean cross power spectrum without weights (Used for mocks, or for data when min_snr_p_cross is not None).
        'forest_snr': Compute mean cross power spectrum with weights = w_i * w_j with w_i = SNR_i, with i and j being the LOS forming each pair.
        'fit_forest_snr': Compute mean P1D with weights = w_i * w_j with w_i and w_j estimated by fitting dispersion of (delta_i * conj(delta_i)) vs SNR.
                          PS: This weight_method doesn't allow forests having snr < 1, if such data is being used, must do min_snr_p_auto = 1.

    Return:
    -------
    p_auto_table: Table
    One row table corresponding to average p_auto computed in ang_sep_bin = 0
    """
    
    ## TODO: cosmo should be args
    # Computing cosmo used for conversions
    Omega_m = 0.3153
    h = 0.7
    cosmo = FlatLambdaCDM(H0=100*h, Om0=Omega_m)

    print('P_auto computation')
    
    # Parameters definitions
    delta_lambda = all_los_table['wavelength'][0][1] - all_los_table['wavelength'][0][0]
    mean_wavelength = np.mean(all_los_table['wavelength'][0])
    z = mean_wavelength / LAMBDA_LYA - 1

    ## if we're in eBOSS case, we have log(lambda) and not lambda so a conversion is required [km/s]
    if data_type == 'eBOSS':
        mean_wavelength = np.mean(10.**(all_los_table['wavelength'][0]))
        z = mean_wavelength / LAMBDA_LYA - 1
        delta_lambda *= SPEED_LIGHT * np.log(10.)

    delta_los = all_los_table['delta_los']
    Npix = len(delta_los[0])
    print('Npix', Npix)
    
    # Applying snr mask
    if min_snr_p_auto is not None:
        print('snr cut applied')
        snr_mask = (all_los_table['MEANSNR'] > min_snr_p_auto)
        if weight_method != 'no_weights':
            print('Warning, both snr cut and weighting will be applied on p_auto')
    else:
        snr_mask = np.ones(len(all_los_table), dtype=bool)
    
    # Applying resolution mask
    if max_resolution_p_auto is not None:
        if data_type == 'mocks':
            print('Warning, no resolution cut will be applied on p_auto since it is a mock case')
            reso_mask = np.ones(len(all_los_table), dtype=bool)
        elif data_type == 'eBOSS':
            print("reso cut applied")
            reso_mask = (all_los_table['MEANRESOLUTION'] < max_resolution_p_auto)
        elif data_type == 'DESI':
            print("reso cut applied")
            reso_mask = (all_los_table['MEANRESO'] < max_resolution_p_auto)     
    else:
        reso_mask = np.ones(len(all_los_table), dtype=bool)

    delta_los = delta_los[ (snr_mask & reso_mask) ]
    Nlos = len(delta_los)

    # FFT of deltas
    fft_delta = np.fft.rfft(delta_los)
    Nk = fft_delta.shape[1]
    print('Nk', Nk)
            
    # Preparing for the weighting scheme
    if weight_method != 'no_weights': 
        snr_los = all_los_table['MEANSNR'][ (snr_mask & reso_mask) ]

    # Initializing p_auto_table
    p_auto_table = Table()
    p_auto_table['ang_sep_bin_centers'] = np.zeros((1))
    p_auto_table['mean_ang_separation'] = np.zeros((1))
    p_auto_table['N'] = np.zeros((1))
    p_auto_table['k_parallel'] = np.zeros((1, Nk))
    p_auto_table['mean_power_spectrum'] = np.zeros((1, Nk))
    p_auto_table['error_power_spectrum'] = np.zeros((1, Nk))
    p_auto_table['resolution_correction'] = np.zeros((1, Nk))
    p_auto_table['corrected_power_spectrum'] = np.zeros((1, Nk))
    p_auto_table['error_corrected_power_spectrum'] = np.zeros((1, Nk))
    if with_covmat:
        p_auto_table['covmat_power_spectrum'] = np.zeros((1, Nk, Nk))
        p_auto_table['covmat_corrected_power_spectrum'] = np.zeros((1, Nk, Nk))

    # p_auto computation in Angstrom units
    p_auto = (fft_delta.real**2 + fft_delta.imag**2) * delta_lambda / Npix # same units as delta_lambda
    k_parallel = 2 * np.pi * np.fft.rfftfreq(Npix, delta_lambda) # same units as delta_lambda

    if data_type == 'eBOSS':
        p_auto = convert_units(p_auto, 'km/s', units, z, inverse_units=False)
        k_parallel = convert_units(k_parallel, 'km/s', units, z, inverse_units=True)
    else:
        p_auto = convert_units(p_auto, 'Angstrom', units, z, inverse_units=False)
        k_parallel = convert_units(k_parallel, 'Angstrom', units, z, inverse_units=True)

    # resolution correction computation
    if resolution_correction == True:
        if data_type == 'eBOSS':
            delta_v = delta_lambda
            resolution_los = all_los_table['MEANRESOLUTION'][ snr_mask & reso_mask ]
            resgrid, kpargrid = np.meshgrid(resolution_los, k_parallel, indexing='ij')
            resolution_correction_los = boss_resolution_correction(resgrid, kpargrid, delta_v)
            resolution_correction_p_auto = resolution_correction_los**2
        elif data_type == 'DESI': # Just for now and must be modified later == no correction on DESI
            # resolution_correction_p_auto = np.ones((Nlos, len(k_parallel)))
            resolution_correction_p_auto = DESI_resolution_correction(z, k_parallel)
    else:
        resolution_correction_p_auto = np.ones((Nlos, len(k_parallel)))

    # mean_p_auto computation
    mean_p_auto = np.zeros(Nk)
    error_p_auto = np.zeros(Nk)
    mean_resolution_correction_p_auto = np.zeros(Nk)

    for i in range(Nk):
        p_auto_array = np.array(p_auto[:,i])
        # Applying weighting scheme
        if weight_method == 'fit_forest_snr':
            # Computing the fit of the variance of p_auto_array 
            snr_bin_edges = np.arange(1, 10 + 1, 1)
            snr_bins = (snr_bin_edges[:-1] + snr_bin_edges[1:]) / 2
            standard_dev_los, _, _ = binned_statistic(snr_los, p_auto_array, statistic="std", bins=snr_bin_edges)
            mask_nan_in_std_los = ~np.isnan(standard_dev_los)
            standard_dev_los = standard_dev_los[mask_nan_in_std_los]
            snr_bins = snr_bins[mask_nan_in_std_los]
            coef_los, *_ = curve_fit(fitfunc_variance_pk1d, snr_bins, standard_dev_los**2, bounds=(0, np.inf))
            # Fixing high and low snr values
            snr_los[snr_los > 10] = 10
            snr_los[snr_los < 1.01] = 1.01
            # Estimated variance
            variance_los_estimated = fitfunc_variance_pk1d(snr_los, *coef_los)
            # Weights
            weights_p_auto_array = 1 / variance_los_estimated
            # Computing weighted average
            mean_p_auto[i] = np.average(p_auto_array, weights=weights_p_auto_array)
            # error_p_auto[i] = np.sqrt(1.0 / np.sum(weights_p_auto_array))
            # Updated error bar using the same formula as in Eq. (B.8) of arXiv:2505.09493
            w2 = weights_p_auto_array**2
            sw = np.sum(weights_p_auto_array)
            sw2 = np.sum(w2)
            avg2 = np.average(p_auto_array**2, weights=w2)
            variance_pauto = (avg2 - (mean_p_auto[i])**2) / (sw**2/sw2 - 1)
            error_p_auto[i] = np.sqrt(variance_pauto)
        elif weight_method == 'forest_snr':
            weights_p_auto_array = snr_los**2
            # Computing weighted average
            mean_p_auto[i] = np.average(p_auto_array, weights=weights_p_auto_array)
            error_p_auto[i] = np.sqrt(1.0 / np.sum(weights_p_auto_array))
        else:
            mean_p_auto[i] = np.mean(p_auto_array)
            error_p_auto[i] = np.std(p_auto_array) / np.sqrt(Nlos - 1)
        try:    
            mean_resolution_correction_p_auto[i] = np.mean(resolution_correction_p_auto[:, i])
        except: # Because in the case of DESI it's the same correction for all LOS
            mean_resolution_correction_p_auto[i] = resolution_correction_p_auto[i]

        # Smoothing errors:
        smoothed_error_p_auto = savgol_filter(error_p_auto, window_length=15, polyorder=1)

    p_auto_table['k_parallel'][0, :] = k_parallel
    p_auto_table['mean_power_spectrum'][0, :] = mean_p_auto  
    p_auto_table['error_power_spectrum'][0, :] = smoothed_error_p_auto
    p_auto_table['resolution_correction'][0, :] = mean_resolution_correction_p_auto
    p_auto_table['corrected_power_spectrum'][0, :] = (mean_p_auto - p_noise) / mean_resolution_correction_p_auto
    p_auto_table['error_corrected_power_spectrum'][0, :] = error_p_auto / mean_resolution_correction_p_auto
    p_auto_table['N'][0] = len(p_auto)

    if with_covmat:  #- covariance matrix
        covmat = np.cov(p_auto.real, rowvar=False)
        p_auto_table['covmat_power_spectrum'][0, :, :] = covmat
        p_auto_table['covmat_corrected_power_spectrum'][0, :, :] = covmat / np.outer(
                mean_resolution_correction_p_auto, mean_resolution_correction_p_auto)  # cov_ij / Wi*Wj

    return p_auto_table


def compute_mean_power_spectrum(all_los_table, los_pairs_table, ang_sep_bin_edges, data_type,
                                units, weight_method='no_weights', p_noise=0, min_snr_p_cross=None, min_snr_p_auto=None,
                                max_resolution_p_cross=None, max_resolution_p_auto=None,
                                resolution_correction=False, reshuffling=False, with_covmat=True):
    """ - This function computes mean_power_spectrum:
            - Takes all_los_table and pairs (1 mock)
            - Computes mean_p_auto and mean_p_cross using above functions
            - Stacks them both in one table called mean_power_spectrum

    Arguments:
    ----------
    Arguments are as defined above
    
    Return:
    -------
    mean_power_spectrum: Table
    Each row corresponds to the computed power spectrum in an angular spearation bin
    """

    p_cross_table = compute_mean_p_cross(all_los_table=all_los_table, los_pairs_table=los_pairs_table, 
                                         ang_sep_bin_edges=ang_sep_bin_edges,
                                         min_snr_p_cross=min_snr_p_cross,
                                         max_resolution_p_cross=max_resolution_p_cross,
                                         resolution_correction=resolution_correction,
                                         reshuffling=reshuffling,
                                         with_covmat=with_covmat,
                                         data_type=data_type, 
                                         units=units,
                                         weight_method=weight_method)

    p_auto_table = compute_mean_p_auto(all_los_table=all_los_table, 
                                       min_snr_p_auto=min_snr_p_auto, 
                                       max_resolution_p_auto=max_resolution_p_auto,
                                       resolution_correction=resolution_correction,
                                       with_covmat=with_covmat,
                                       p_noise=p_noise,
                                       data_type=data_type, 
                                       units=units,
                                       weight_method=weight_method)
    mean_power_spectrum = vstack([p_auto_table, p_cross_table])

    return mean_power_spectrum


def wavenumber_rebin_power_spectrum(power_spectrum_table, k_scale, n_kbins=None):
    """ This function rebins the cross power spectrum into parallel wavenumber bins

    Arguments:
    ----------
    power_spectrum_table: Table
    Table of mean power spectrum

    n_kbins: Integer
    Number of k bins we want after rebinning
    
    k_scale: String
    Scale of wavenumber array to be rebinned. Options: 'linear', 'log', 'custom_DR1'. n_kbins not needed for custom_DR1.

    Return:
    -------
    power_spectrum_table: Table
    Same table as in input, but with rebinned power spectrum columns added to the table
    """

    if k_scale == 'log': # Used in mocks case
        k_bin_edges = np.logspace(-2, np.log10(np.max(power_spectrum_table['k_parallel'][0])), 
                                  num=n_kbins+1)
    elif k_scale == 'linear': # Used in data case
        k_bin_edges = np.linspace(np.min(power_spectrum_table['k_parallel'][0]), 
                                  np.max(power_spectrum_table['k_parallel'][0]), num=n_kbins+1)
    elif k_scale == 'custom_DR1':
        k_bin_edges = np.arange(0, 3.1, 0.1)
    else:
        sys.exit('Must input k_scale: log, linear or custrom_DR1')

    k_bin_centers = np.around((k_bin_edges[1:] + k_bin_edges[:-1]) / 2, 5) # same units as k_parallel

    # Add columns to power_spectrum_table
    power_spectrum_table['k_parallel_rebinned'] = np.zeros((len(power_spectrum_table), len(k_bin_centers)))
    power_spectrum_table['mean_power_spectrum_rebinned'] = np.zeros((len(power_spectrum_table), len(k_bin_centers)))
    power_spectrum_table['error_power_spectrum_rebinned'] = np.zeros((len(power_spectrum_table), len(k_bin_centers)))
    if 'resolution_correction' in power_spectrum_table.keys():
        power_spectrum_table['resolution_correction_rebinned'] = np.zeros((len(power_spectrum_table), len(k_bin_centers)))
    if 'corrected_power_spectrum' in power_spectrum_table.keys():
        power_spectrum_table['corrected_power_spectrum_rebinned'] = np.zeros((len(power_spectrum_table), len(k_bin_centers)))
        power_spectrum_table['error_corrected_power_spectrum_rebinned'] = np.zeros((len(power_spectrum_table), len(k_bin_centers)))
    if 'power_spectrum_SB' in power_spectrum_table.keys():
        power_spectrum_table['power_spectrum_SB_rebinned'] = np.zeros((len(power_spectrum_table), len(k_bin_centers)))
        power_spectrum_table['error_power_spectrum_SB_rebinned'] = np.zeros((len(power_spectrum_table), len(k_bin_centers)))
    if 'SB_corrected_power_spectrum' in power_spectrum_table.keys():
        power_spectrum_table['SB_corrected_power_spectrum_rebinned'] = np.zeros((len(power_spectrum_table), len(k_bin_centers)))
        power_spectrum_table['error_SB_corrected_power_spectrum_rebinned'] = np.zeros((len(power_spectrum_table), len(k_bin_centers)))
    if 'Full_corrected_power_spectrum' in power_spectrum_table.keys():
        power_spectrum_table['Full_corrected_power_spectrum_rebinned'] = np.zeros((len(power_spectrum_table), len(k_bin_centers)))
        power_spectrum_table['error_Full_corrected_power_spectrum_rebinned'] = np.zeros((len(power_spectrum_table), len(k_bin_centers)))

    for j in range(len(power_spectrum_table)):

        for ik_bin, k_bin in enumerate(k_bin_edges[:-1]):

            select_k = (power_spectrum_table['k_parallel'][j] > k_bin_edges[ik_bin]) & (
                power_spectrum_table['k_parallel'][j] <= k_bin_edges[ik_bin+1])
            
            mean_k_parallel_in_bin = np.mean(power_spectrum_table['k_parallel'][j][select_k])
            mean_power_spectrum_rebinned = np.mean(power_spectrum_table['mean_power_spectrum'][j][select_k])
            # error_power_spectrum_rebinned = np.mean(power_spectrum_table['error_power_spectrum'][j][select_k]) / np.sqrt(np.sum(select_k))
            error_power_spectrum_rebinned = np.sqrt(np.sum(power_spectrum_table['error_power_spectrum'][j][select_k]**2)) / np.sum(select_k)
            power_spectrum_table['k_parallel_rebinned'][j,ik_bin] = mean_k_parallel_in_bin
            power_spectrum_table['mean_power_spectrum_rebinned'][j,ik_bin] = mean_power_spectrum_rebinned 
            power_spectrum_table['error_power_spectrum_rebinned'][j,ik_bin] = error_power_spectrum_rebinned
            
            try:
                resolution_correction_rebinned = np.mean(power_spectrum_table['resolution_correction'][j][select_k])
                corrected_power_spectrum_rebinned = np.mean(power_spectrum_table['corrected_power_spectrum'][j][select_k])
                error_corrected_power_spectrum_rebinned = np.sqrt(np.sum(power_spectrum_table['error_corrected_power_spectrum'][j][select_k]**2)) / np.sum(select_k)
                power_spectrum_table['resolution_correction_rebinned'][j,ik_bin] = resolution_correction_rebinned
                power_spectrum_table['corrected_power_spectrum_rebinned'][j,ik_bin] = corrected_power_spectrum_rebinned
                power_spectrum_table['error_corrected_power_spectrum_rebinned'][j,ik_bin] = error_corrected_power_spectrum_rebinned
            except:
                pass

            try:
                power_spectrum_SB_rebinned = np.mean(power_spectrum_table['power_spectrum_SB'][j][select_k])
                error_power_spectrum_SB_rebinned = np.sqrt(np.sum(power_spectrum_table['error_power_spectrum_SB'][j][select_k]**2)) / np.sum(select_k)
                power_spectrum_table['power_spectrum_SB_rebinned'][j,ik_bin] = power_spectrum_SB_rebinned
                power_spectrum_table['error_power_spectrum_SB_rebinned'][j,ik_bin] = error_power_spectrum_SB_rebinned
            except:
                pass

            try:
                SB_corrected_power_spectrum_rebinned = np.mean(power_spectrum_table['SB_corrected_power_spectrum'][j][select_k])
                error_SB_corrected_power_spectrum_rebinned = np.sqrt(np.sum(power_spectrum_table['error_SB_corrected_power_spectrum'][j][select_k]**2)) / np.sum(select_k)
                power_spectrum_table['SB_corrected_power_spectrum_rebinned'][j,ik_bin] = SB_corrected_power_spectrum_rebinned
                power_spectrum_table['error_SB_corrected_power_spectrum_rebinned'][j,ik_bin] = error_SB_corrected_power_spectrum_rebinned
            except:
                pass

            try:
                Full_corrected_power_spectrum_rebinned = np.mean(power_spectrum_table['Full_corrected_power_spectrum'][j][select_k])
                error_Full_corrected_power_spectrum_rebinned = np.sqrt(np.sum(power_spectrum_table['error_Full_corrected_power_spectrum'][j][select_k]**2)) / np.sum(select_k)
                power_spectrum_table['Full_corrected_power_spectrum_rebinned'][j,ik_bin] = Full_corrected_power_spectrum_rebinned
                power_spectrum_table['error_Full_corrected_power_spectrum_rebinned'][j,ik_bin] = error_Full_corrected_power_spectrum_rebinned
            except:
                pass

    return power_spectrum_table


def redshift_rebin_power_spectrum(power_spectrum_tables_list, ps_columns=None, error_columns=None):
    """ This function rebins the cross power spectrum in redshift, it only works for averaging power spectra that have been rebinned with k_scale = 'custom_DR1,
    Otherwise, it might raise some errors or give inaccurate results.
    PS: it won't work on power spectra that weren't rebinned anyway, because it reads _rebinned arguments only, and these are added at the level of wavenumber_rebin_power_spectrum function

    Arguments:
    ----------
    power_spectrum_tables_list: List of Tables
    List of the mean power spectrum tables

    Return:
    -------
    power_spectrum_table_z_rebinned: Table
    New power spectrum table at rebinned redshift value
    """

    N_ps = len(power_spectrum_tables_list)

    # Since all power spectra have same n_ang_sep_bins and Nk, I'll read first table to define table dimensions only here
    ang_sep_bin_centers = power_spectrum_tables_list[0]['ang_sep_bin_centers'] # This is unchanged from one ps to another
    n_ang_sep_bins = len(ang_sep_bin_centers)
    Nk = len(power_spectrum_tables_list[0]['k_parallel_rebinned'])

    # Initializing power_spectrum_table_rebinned
    power_spectrum_table_z_rebinned = Table()

    # Defining columns to be averaged
    if ps_columns is None:
        ps_columns = ['mean_ang_separation', 'k_parallel_rebinned', 'mean_power_spectrum_rebinned', 'corrected_power_spectrum_rebinned', 'SB_corrected_power_spectrum_rebinned', 'Full_corrected_power_spectrum_rebinned']
    if error_columns is None:
        error_columns = ['error_power_spectrum_rebinned', 'error_corrected_power_spectrum_rebinned', 'error_SB_corrected_power_spectrum_rebinned', 'error_Full_corrected_power_spectrum_rebinned']

    # Stacking
    for col in ps_columns + error_columns:
        stacked = np.stack([table[col] for table in power_spectrum_tables_list], axis=0)
        if col in error_columns:
            averaged = np.sqrt(np.sum(stacked**2, axis=0)) / N_ps
        else:
            averaged = np.mean(stacked, axis=0)
        power_spectrum_table_z_rebinned[col] = averaged

    return power_spectrum_table_z_rebinned


def run_compute_mean_power_spectrum(mocks_dir, ncpu, ang_sep_max, n_kbins, k_scale, data_type, units, weight_method='no_weights',
                                    ang_sep_bin_edges=None, ang_sep_bin_centers=None,
                                    p_noise=0, min_snr_p_cross=None, min_snr_p_auto=None,
                                    max_resolution_p_cross=None, max_resolution_p_auto=None,
                                    resolution_correction=False, reshuffling=False, with_covmat=True,
                                    k_binning=False, radec_names=['ra', 'dec']): 
    """ - This function computes all_mocks_mean_power_spectrum:
            - Takes all mocks or one mock
            - Gets pairs table for each mock separately
            - Computes mean_p_auto and mean_p_cross for each mock separately
            - Averages over all mocks (mean of mean)
            - Does wavenumber rebinning if k_binning==True

    Arguments:
    ----------
    mocks_dir: String
    Directory where mocks fits files are located
    
    ncpu: Integer
    
    ang_sep_max: Same definition as in function get_possible_pairs
    
    ang_sep_bin_edges: Array, Default to None
    Array of angular separation bin edges.
    If this array wasn't given, it can be determined automatically if the desired bin centers were given.
    Either this array or ang_sep_bin_centers must be given.

    ang_sep_bin_centers: Array, Default to None
    Array of angular separation bin centers determined automatically if the desired bin edges were given.
    Either this array or ang_sep_bin_edges must be given.

    min_snr_p_cross, min_snr_p_auto: Floats, Defaults are None
    The values of minimum snr required for both p_cross and p_auto computation.
    
    k_binning: Boolean, Default to False
    Rebin power spectrum using wavenumber_rebin_power_spectrum function
    
    k_scale: String
    Scale of wavenumber array to be rebinned if k_binning. Options: 'linear', 'log'
    
    data_type: String, Options: 'mocks', 'DESI', 'eBOSS'
    The type of data set on which we want to run the power spectrum computation.
        - In the case of mocks: The power spectrum will be computed in [Angstrom] by default,
        because when we draw LOS to create mocks, wavelength = (1 + refshift) * lambda_lya [Angstrom].
        If another unit is desired, this must be specified in the argument units.
        - In the case of real data: The power spectrum will be first computed unitless,
        because wavelength = LOGLAM, therefore it is mandatory to multiply it my a factor c, and the output will be in [km/s].
    
    units: String, Options: 'Mpc/h', 'Angstrom', 'km/s', Default is Angstrom
    Units in which to compute power spectrum. This argument must be specified if data_type is 'mocks'.
    
    radec_names: List of str, Default: ['ra', 'dec']
    ra dec keys in mocks or data table
    Options: - ['ra', 'dec']: my mocks
             - ['RA', 'DEC']: eBOSS data
             - ['TARGET_RA', 'TARGET_DEC']: IRON
             - ['x', 'y']: Mpc/h analysis

    Return:
    -------
    final_power_spectrum: Table
    Each row corresponds to the computed mean power spectrum over all mocks in one angular spearation bin
    """
    
    # Conversion from degree to Mpc
    ## TODO: cosmo/z should be args
    Omega_m = 0.3153
    h = 0.7
    cosmo = FlatLambdaCDM(H0=100*h, Om0=Omega_m)
    z = 2.59999
    deg_to_Mpc = cosmo.comoving_distance(z).value * np.pi / 180

    # Loading mocks files
    searchstr = '*'
    files = glob.glob(os.path.join(mocks_dir, f"{searchstr}.fits.gz"))
    
    # Initializing all mocks mean power spectrum table
    all_mocks_mean_power_spectrum = Table()
    
    for i_f, f in enumerate(files):
    
        # Reading mock (file)
        print('Reading file test'+str(f))
        all_los_table = Table.read(f)

        # Computing los pairs
        los_pairs_table = compute_pairs(all_los_table, ang_sep_max,
                                        radec_names=radec_names, ncpu=ncpu)

        # Defining edges of angular separation bins used for next step
        if (ang_sep_bin_edges is None) and (ang_sep_bin_centers is not None):
            ang_sep_bin_edges = find_bin_edges(los_pairs_table['ang_separation'],
                                               mean_values_target=ang_sep_bin_centers,
                                               debug=True)

        elif (ang_sep_bin_edges is not None) and (ang_sep_bin_centers is not None):
            print("WARNING! both ang_sep_bin_edges and ang_sep_bin_centers were assigned,"+
                  "in this case new ang_sep_bin_edges will be calculated based on the ang_sep_bin_centers input array")

            ang_sep_bin_edges = find_bin_edges(los_pairs_table['ang_separation'],
                                               mean_values_target=ang_sep_bin_centers,
                                               debug=True)

        elif (ang_sep_bin_edges is None) and (ang_sep_bin_centers is None):
            raise ValueError("At least one of ang_sep_bin_edges or ang_sep_bin_centers arrays must be given!")

        if radec_names == ['x', 'y']:
            print("WARNING! converting distang with z=",z)
            ang_sep_bin_edges *= deg_to_Mpc * h

        # Computing the mean_p_cross for each mock

        print('Computing mean power spectrum in input units')

        mock_mean_power_spectrum = compute_mean_power_spectrum(all_los_table=all_los_table, 
                                                               los_pairs_table=los_pairs_table,
                                                               ang_sep_bin_edges=ang_sep_bin_edges,
                                                               data_type=data_type, units=units,
                                                               weight_method=weight_method,
                                                               p_noise=p_noise, 
                                                               min_snr_p_cross=min_snr_p_cross, 
                                                               min_snr_p_auto=min_snr_p_auto,
                                                               max_resolution_p_cross=max_resolution_p_cross,
                                                               max_resolution_p_auto=max_resolution_p_auto, 
                                                               resolution_correction=resolution_correction, 
                                                               reshuffling=reshuffling,
                                                               with_covmat=with_covmat)

        # Stacking power spectra of all mocks in one table
        all_mocks_mean_power_spectrum = vstack([all_mocks_mean_power_spectrum, mock_mean_power_spectrum])  
        
    if k_binning:
        print('Wavenumber rebinning')
        all_mocks_mean_power_spectrum = wavenumber_rebin_power_spectrum(power_spectrum_table=all_mocks_mean_power_spectrum, 
                                                         n_kbins=n_kbins, k_scale=k_scale)

    return all_mocks_mean_power_spectrum

# Commented because we don't do the analysis on many mock realizations right now and must be updated
#     # Mean over all mocks
#     N_mocks = len(files)
#     if N_mocks == 1:
#         print('Averaging on '+str(N_mocks)+' mock')
#     else:
#         print('Averaging on '+str(N_mocks)+' mocks')

#     N_ang_sep_bins = int(len(all_mocks_mean_power_spectrum) / N_mocks)
#     k_parallel = all_mocks_mean_power_spectrum['k_parallel'][0] 
#     N_k_bins = len(k_parallel)
#     ang_sep_bin_centers = np.array(all_mocks_mean_power_spectrum['ang_sep_bin_centers'][:N_ang_sep_bins])
    
#     ### Initializing table
#     final_power_spectrum = Table()
#     final_power_spectrum['ang_sep_bin_centers'] = ang_sep_bin_centers
#     final_power_spectrum['mean_ang_separation'] = np.zeros(len(ang_sep_bin_centers))
#     final_power_spectrum['N'] = np.zeros(len(ang_sep_bin_centers))
#     final_power_spectrum['k_parallel'] = np.zeros((len(ang_sep_bin_centers), N_k_bins))
#     final_power_spectrum['mean_power_spectrum'] = np.zeros((len(ang_sep_bin_centers), N_k_bins))
#     final_power_spectrum['error_power_spectrum'] = np.zeros((len(ang_sep_bin_centers), N_k_bins))
#     final_power_spectrum['covmat_power_spectrum'] = np.zeros((1, N_k_bins, N_k_bins))
#     final_power_spectrum['resolution_correction'] = np.zeros((len(ang_sep_bin_centers), N_k_bins))
#     final_power_spectrum['corrected_power_spectrum'] = np.zeros((len(ang_sep_bin_centers), N_k_bins))
    
#     ### Averaging
#     for i_ang_sep, ang_sep in enumerate(ang_sep_bin_centers): 
    
#         select = (all_mocks_mean_power_spectrum['ang_sep_bin_centers'] == ang_sep)

#         N = np.sum(all_mocks_mean_power_spectrum['N'][select])
#         final_power_spectrum['N'][i_ang_sep] = N

#         mean_ang_separation = np.mean(all_mocks_mean_power_spectrum['mean_ang_separation'][select])
#         final_power_spectrum['mean_ang_separation'][i_ang_sep] = mean_ang_separation

#         mean = np.zeros(N_k_bins)
#         error = np.zeros(N_k_bins)

#         for i in range(N_k_bins):
#             if N_mocks>1:
#                 mean[i] = np.mean(all_mocks_mean_power_spectrum['mean_power_spectrum'][select][:, i])
#                 error[i] = np.mean(all_mocks_mean_power_spectrum['error_power_spectrum'][select][:,i]) / np.sqrt(N_mocks - 1)
#             else:
#                 mean[i] = all_mocks_mean_power_spectrum['mean_power_spectrum'][select][:, i]
#                 error[i] = all_mocks_mean_power_spectrum['error_power_spectrum'][select][:,i]
                

#         final_power_spectrum['k_parallel'][i_ang_sep, :] = k_parallel
#         final_power_spectrum['mean_power_spectrum'][i_ang_sep, :] = mean 
#         final_power_spectrum['error_power_spectrum'][i_ang_sep, :] = error
#         final_power_spectrum['covmat_power_spectrum'][i_ang_sep, :, :] = 
#         final_power_spectrum['resolution_correction'][i_ang_sep, :] = 
#         final_power_spectrum['corrected_power_spectrum'][i_ang_sep, :] = 
        
#     if k_binning:
#         print('Wavenumber rebinning')
#         final_power_spectrum = wavenumber_rebin_power_spectrum(power_spectrum_table=final_power_spectrum, rebin_factor=rebin_factor)
        
    # return final_power_spectrum

