""" This module provides a set of functions to get P3D from Pcross computed on data """

import numpy as np
import os, sys
import astropy.io.fits
from astropy.table import Table
import scipy
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

from tools import rebin_vector, SPEED_LIGHT, LAMBDA_LYA


# def wavenumber_rebin(p3d_table, rebin_factor):
def wavenumber_rebin(p3d_table, n_kbins):
    """ This function rebins the 3D power spectrum into k_parallel bins
    
    Arguments:
    ----------
    p3d_table: Table
    Table of P3D
    
    # rebin_factor: Integer
    # Rebin factor
    
    n_kbins: Integer
    Number of k bins we want after rebinning
    
    Return:
    -------
    p3d_table: Table
    Same table as in input, but with rebinned p3d columns added to the table
    """
    
    k_bin_edges = np.logspace(-2, np.log10(np.max(p3d_table['k_parallel'][0])), num=n_kbins) # same units as k_parallel
    k_bin_centers = np.around((k_bin_edges[1:] + k_bin_edges[:-1]) / 2, 5) # same units as k_parallel

#     # First rebin k_parallel array
#     k_parallel_rebinned = rebin_vector(p3d_table['k_parallel'][0], pack=2, rebin_opt='mean', verbose=False)
    
    # p3d_table['k_parallel_rebinned'] = np.zeros((len(p3d_table), len(k_parallel_rebinned)))
    # p3d_table['P3D_rebinned'] = np.zeros((len(p3d_table), len(k_parallel_rebinned)))
    # p3d_table['error_P3D_rebinned'] = np.zeros((len(p3d_table), len(k_parallel_rebinned)))
    p3d_table['k_parallel_rebinned'] = np.zeros((len(p3d_table), len(k_bin_centers)))
    p3d_table['P3D_rebinned'] = np.zeros((len(p3d_table), len(k_bin_centers)))
    p3d_table['error_P3D_rebinned'] = np.zeros((len(p3d_table), len(k_bin_centers)))
    
#     for j in range(len(p3d_table)):

#         P3D_rebinned = rebin_vector(p3d_table['P3D'][j], 
#                                                     pack=2, rebin_opt='mean', verbose=False)
#         error_P3D_rebinned = rebin_vector(p3d_table['error_P3D'][j], 
#                                                     pack=2, rebin_opt='mean', verbose=False) / np.sqrt(rebin_factor)
        
#         p3d_table['k_parallel_rebinned'][j,:] = k_parallel_rebinned
#         p3d_table['P3D_rebinned'][j,:] = P3D_rebinned 
#         p3d_table['error_P3D_rebinned'][j,:] = error_P3D_rebinned

    for j in range(len(p3d_table)):
    
        p3d_table['k_parallel_rebinned'][j,:] = k_bin_centers

        for ik_bin, k_bin in enumerate(k_bin_edges[:-1]):

            select_k = (p3d_table['k_parallel'][j] > k_bin_edges[ik_bin]) & (
                p3d_table['k_parallel'][j] <= k_bin_edges[ik_bin+1])

            P3D_rebinned = np.mean(p3d_table['P3D'][j][select_k])
            error_P3D_rebinned = np.mean(p3d_table['error_P3D'][j][select_k])
            
            p3d_table['P3D_rebinned'][j,ik_bin] = P3D_rebinned 
            p3d_table['error_P3D_rebinned'][j,ik_bin] = error_P3D_rebinned

    return p3d_table


def _pcross_interpolated(pcross_table, angular_separation_array, n_angsep=1000,
						interp_method='UnivariateSpline',
						add_noise=False):
	#- Compute Pcross_interpolated(kpar, angsep)
	#      add_noise: if True, add gaussian fluctuations on top of Pcross
	#				 (used to propagate Px error bars to P3D)
	#      other args are described in main routines

    # reading k_parallel from pcross_table
    k_parallel = np.array(pcross_table['k_parallel'][0])

    # Define a thinner binning of angular separations
    ang_sep_min = np.min(angular_separation_array)
    ang_sep_max = np.max(angular_separation_array)
    if interp_method == 'none':
        n_angsep = len(angular_separation_array)
        angular_separation_array_fine_binning = angular_separation_array
    else:
        angular_separation_array_fine_binning = np.linspace(ang_sep_min, ang_sep_max, n_angsep)

    # Reading Pcross from table
    try:
        Pcross = np.array(pcross_table['corrected_power_spectrum'])
    except:
        Pcross = np.array(pcross_table['mean_power_spectrum'])

    # Add fluctuations
    if add_noise:
        if 'covmat_power_spectrum' in pcross_table.keys():
            # One covariance matrix per angular separation bin:
            for i_angsep in range(len(angular_separation_array)):
                covmat = np.array(pcross_table['covmat_power_spectrum'][i_angsep,:,:])
                Pcross[i_angsep,:,:] = np.random.multivariate_normal(Pcross[i_angsep,:], covmat)
        else:
            error_Pcross = np.array(pcross_table['error_power_spectrum'])
            Pcross = np.random.normal(Pcross, error_Pcross)

    Pcross_interpolated = np.zeros( (n_angsep, len(k_parallel)) )
    for ik_par, k_par in enumerate(k_parallel):  # k_par in [h/Mpc]
        # Interpolating Pcross
        if interp_method == 'none':
            Pcross_interpolated[:, ik_par] = Pcross[:, ik_par]
        else:
            if interp_method == 'UnivariateSpline':
                interpolation_function_Pcross = scipy.interpolate.UnivariateSpline(
                    angular_separation_array, Pcross[:, ik_par], s=smoothing)
            elif interp_method == 'PchipInterpolator':
                interpolation_function_Pcross = scipy.interpolate.PchipInterpolator(
                    angular_separation_array, Pcross[:, ik_par])
            else:
                raise ValueError('Wrong interp_method')
            # Pcross_interpolated computation
            Pcross_interpolated[:, ik_par] = interpolation_function_Pcross(angular_separation_array_fine_binning)

    return (angular_separation_array_fine_binning, Pcross_interpolated)


def pcross_to_p3d_cartesian(pcross_table, k_perpendicular, units_k_perpendicular,
                  mean_redshift, interp_method='UnivariateSpline', smoothing=0, n_angsep=1000,
                  compute_errors=False, k_binning=False):
    """ This function computes the P3D out of the Pcross in cartesian coordinates:
          - It either computes the P3D out of Pcross by direct integration over the angular separation
          - Or it interpolates the Pcross with a spline function before integration
          - For error_P3D computation:
              - This function generates n_iterations of random normal Pcross
              - Interpolates each with a spline and computes random P3Ds
              - Error = std(n_iteration random P3Ds)

    Arguments:
    ----------
    pcross_table: Table
    Table of Pcross as function of K_parallel for different angular separation bins.

    k_peprendicular: Array
    Array of k_perpendicular we want to use, either in [Mpc/h]^-1 or [degree]^-1, must be specified in the following argument
    
    units_k_perpendicular: String, default: '[Mpc/h]^-1'
    Units of input k_perpendicular
    Options: - '[Mpc/h]^-1': usually in the case of mocks
             - '[degree]^-1': usually in the case of real data

    mean_redshift: Float
    Central redshift value
    
    interp_method: String, default: 'UnivariateSpline'
    Pcross interpolation method, before P3D computation
    Options: - 'UnivariateSpline': interpolating using scipy.interpolate.UnivariateSpline function
             - 'PchipInterpolator': interpolating using scipy.interpolate.PchipInterpolator function
             - 'none': no interpolation, compute integral directly on Pcross output
             
    smoothing: Float, default: 0
    The value of smoothing if 'spline interpolation method' only
    
    n_angsep: Float, default: 1000
    Number of angular separation bins for angular_separation_fine_binning_array required for interpolation methods only

    compute_errors: Boolean, default: False
    Compute error_P3D or not
    
    k_binning: Boolean, Default to False
    Rebin P3D using wavenumber_rebin function
    
    # The p3d output units will be the same as Pcross input
    
    Return:
    -------
    p3d_table: Table
    Table of P3D as function of K_parallel and K_perpendicular."""

    ## TODO: cosmo should be args
    # Computing cosmo used for conversions
    Omega_m=0.3153
    h = 0.7
    cosmo = FlatLambdaCDM(H0=100*h, Om0=Omega_m)

    # Conversion factor from degree to Mpc
    deg_to_Mpc = cosmo.comoving_distance(mean_redshift).value * np.pi / 180
    
    # mean_ang_separation is always saved in degrees it must be converted to Mpc/h if k_perp is defined in [Mpc/h]^-1
    if units_k_perpendicular == '[Mpc/h]^-1':
        angular_separation_array = np.array(pcross_table['mean_ang_separation']) * deg_to_Mpc * h # [Mpc/h]
    else:
        angular_separation_array = np.array(pcross_table['mean_ang_separation']) # [degree]
    k_parallel = np.array(pcross_table['k_parallel'][0])

    ang_sep_finebin, Pcross_interpolated = _pcross_interpolated(
                            pcross_table, angular_separation_array,
                            n_angsep=n_angsep, interp_method=interp_method)

    # Generating n_iterations of random Pcross and interpolating each before error_P3D computation
    if compute_errors == True:
        # Defining number of iterations of Pcross
        n_iterations = 100
        random_Pcross_interpolated = np.zeros( (n_iterations, n_angsep, len(k_parallel)) )
        for i in range(n_iterations):
            _, pcross_fluct = _pcross_interpolated(pcross_table,
                            angular_separation_array,
                            n_angsep=n_angsep,
                            interp_method=interp_method,
                            add_noise=True)
            random_Pcross_interpolated[i,:,:] = pcross_fluct

    # Initializing P3D table
    p3d_table = Table()
    p3d_table['k_perpendicular'] = np.array(k_perpendicular)
    p3d_table['k_parallel'] = np.zeros((len(k_perpendicular), len(k_parallel)))
    p3d_table['P3D'] = np.zeros((len(k_perpendicular), len(k_parallel)))
    p3d_table['error_P3D'] = np.zeros((len(k_perpendicular), len(k_parallel)))

    # Integrate Pcross_interpolated to compute P3D
    for ik_par, k_par in enumerate(k_parallel):  # k_par in [h/Mpc]
        for ik_perp, k_perp in enumerate(k_perpendicular): # k_perp in [h/Mpc]
            #  Defining integrand_Pcross
            integrand_Pcross = 2 * np.pi * ang_sep_finebin * scipy.special.j0(ang_sep_finebin * k_perp) * Pcross_interpolated[:,ik_par]

            # Computing integral to get P3D
            P3D = np.trapz(integrand_Pcross, ang_sep_finebin)

            # Filling table
            p3d_table['k_parallel'][ik_perp,ik_par] = k_par
            p3d_table['P3D'][ik_perp,ik_par] = P3D

            if compute_errors == True:
                # Defining integrand_random_Pcross
                integrand_random_Pcross = 2 * np.pi * ang_sep_finebin * scipy.special.j0(ang_sep_finebin * k_perp) * random_Pcross_interpolated[:,:,ik_par]

                # Computing integral to get P3D
                random_P3D = np.trapz(integrand_random_Pcross, angular_separation_array_fine_binning, axis=-1)
                error_P3D = np.std(random_P3D)

                # Filling table
                p3d_table['error_P3D'][ik_perp,ik_par] = error_P3D
                    
    if k_binning == True:
        n_kbins = 60
        p3d_table = wavenumber_rebin(p3d_table, n_kbins)
        # rebin_factor = 2
        # p3d_table = wavenumber_rebin(p3d_table, rebin_factor)

    return p3d_table


def pcross_to_p3d_polar(pcross_table, mu_array, mean_redshift, input_units='Mpc/h', output_units='Mpc/h',
                        interp_method='UnivariateSpline', smoothing=0, n_angsep=1000, compute_errors=False):
    """ This function computes the P3D out of the Pcross in polar coordinates:
          - It either computes the P3D out of Pcross by direct integration over the angular separation
          - Or it interpolates the Pcross with a spline function before integration
          - For error_P3D computation:
              - This function generates n_iterations of random normal Pcross
              - Interpolates each with a spline and computes random P3Ds
              - Error = std(n_iteration random P3Ds)

    Arguments:
    ----------
    pcross_table: Table
    Table of Pcross as function of K_parallel for different angular separation bins.

    mean_redshift: Float
    Central redshift value

    input_units: String, Default: 'Mpc/h'
    Units of input Pcross.
    In the case of real data, the input might be in [km/s] (or [Angstrom] in the case of DESI, will be implemented too)
    Options: - 'Mpc/h'
             - 'km/s'

    output_units: String, Default: 'Mpc/h'
    Units of output p3d.
    In the case of eBOSS, usually the desired output is in [km/s] while for DESI it is in [Angstrom]
    Options: - 'Mpc/h'
             - 'km/s'
             - 'Angstrom'

    interp_method: String, default: 'UnivariateSpline'
    Pcross interpolation method, before P3D computation
    Options: - 'UnivariateSpline': interpolating using scipy.interpolate.UnivariateSpline function
             - 'PchipInterpolator': interpolating using scipy.interpolate.PchipInterpolator function
             - 'none': no interpolation, compute integral directly on Pcross output

    smoothing: Float, default: 0
    The value of smoothing if 'spline interpolation method' only

    n_angsep: Float, default: 1000
    Number of angular separation bins for angular_separation_fine_binning_array required for interpolation methods only

    compute_errors: Boolean, default: False
    Compute error_P3D or not

    # The p3d output units will be the same as Pcross input

    Return:
    -------
    p3d_table: Table
    Table of P3D as function of Mu and K."""

    ## TODO: cosmo should be args
    # Computing cosmo used for conversions
    Omega_m=0.3153
    h = 0.7
    cosmo = FlatLambdaCDM(H0=100*h, Om0=Omega_m)

    # Conversion factor from degree to Mpc
    deg_to_Mpc = cosmo.comoving_distance(mean_redshift).value * np.pi / 180

    # mean_ang_separation is always saved in degrees it must be converted to Mpc/h
    angular_separation_array = np.array(pcross_table['mean_ang_separation']) * deg_to_Mpc * h # [Mpc/h]

    # converting k_parallel and pcross to Mpc/h if needed
    if input_units == 'km/s': # Must be converted to Mpc/h
        conversion_factor = (1 + mean_redshift) * h / cosmo.H(mean_redshift).value
    elif input_units == 'Angstrom':
        conversion_factor = SPEED_LIGHT * h / (cosmo.H(mean_redshift).value * LAMBDA_LYA)
    else:
        conversion_factor = 1

    pcross_table['mean_power_spectrum'] *= conversion_factor
    pcross_table['k_parallel'] /= conversion_factor

    # reading k_parallel from pcross_table
    k_parallel = np.array(pcross_table['k_parallel'][0])

    ang_sep_finebin, Pcross_interpolated = _pcross_interpolated(
                            pcross_table, angular_separation_array,
                            n_angsep=n_angsep, interp_method=interp_method)

    # Generating n_iterations of random Pcross and interpolating each before error_P3D computation
    if compute_errors == True:
        # Defining number of iterations of Pcross
        n_iterations = 100
        random_Pcross_interpolated = np.zeros( (n_iterations, n_angsep, len(k_parallel)) )
        for i in range(n_iterations):
            _, pcross_fluct = _pcross_interpolated(pcross_table,
                            angular_separation_array,
                            n_angsep=n_angsep,
                            interp_method=interp_method,
                            add_noise=True)
            random_Pcross_interpolated[i,:,:] = pcross_fluct

    # Initializing P3D table
    p3d_table = Table()
    p3d_table['mu'] = np.array(mu_array)
    p3d_table['k'] = np.zeros((len(mu_array), len(k_parallel))) # len of k and k_parallel is the same
    p3d_table['P3D'] = np.zeros((len(mu_array), len(k_parallel)))
    p3d_table['error_P3D'] = np.zeros((len(mu_array), len(k_parallel)))

    ## P3D computation
    for i_mu, mu in enumerate(mu_array):

        for ik_par, k_par in enumerate(k_parallel):  # k_par in [h/Mpc]

            # Computing k and k_perpendicular
            k = k_par / mu
            k_perp = np.sqrt(k**2 - k_par**2)

            # Defining integrand_Pcross
            integrand_Pcross = 2 * np.pi * ang_sep_finebin * scipy.special.j0(ang_sep_finebin * k_perp) * Pcross_interpolated[:,ik_par]

            # Computing integral to get P3D
            P3D = np.trapz(integrand_Pcross, ang_sep_finebin)

            # Filling table
            p3d_table['k'][i_mu,ik_par] = k
            p3d_table['P3D'][i_mu,ik_par] = P3D

            if compute_errors == True:
                integrand_random_Pcross = 2 * np.pi * ang_sep_finebin * scipy.special.j0(ang_sep_finebin * k_perp) * random_Pcross_interpolated[:,:,ik_par]
                random_P3D = np.trapz(integrand_random_Pcross, angular_separation_array_fine_binning, axis=-1)
                error_P3D = np.std(random_P3D)
                p3d_table['error_P3D'][i_mu,ik_par] = error_P3D

    # converting k_parallel and p3d to desired output units
    if output_units == 'km/s': # Must be converted from Mpc/h to km/s
        conversion_factor = (1 + mean_redshift) * h / cosmo.H(mean_redshift).value
    elif output_units == 'Angstrom': # Must be converted from Mpc/h to Angstrom
        conversion_factor = SPEED_LIGHT * h / (cosmo.H(mean_redshift).value * LAMBDA_LYA)
    else:
        conversion_factor = 1

    p3d_table['P3D'] /= conversion_factor**3
    p3d_table['error_P3D'] /= conversion_factor**3
    p3d_table['k'] *= conversion_factor

    return p3d_table


def plot_integrand(pcross_table):
    """ This function plots integrand used for P3D computation, by fitting a spline funtion to Pcross """
    
    z = 2.59999
    
    # Computing cosmo used for Conversions
    Omega_m=0.3153
    h = 0.7
    cosmo = FlatLambdaCDM(H0=100*h, Om0=Omega_m)

    # Conversion from degree to Mpc
    deg_to_Mpc = cosmo.comoving_distance(z).value * np.pi / 180
    
    # Choice of k_parallel and k_perpendicular
    k_parallel_min = np.min(pcross_table['k_parallel'][0])
    k_parallel_max = np.max(pcross_table['k_parallel'][0])
    k_parallel = np.linspace(k_parallel_min, k_parallel_max, 20)

    
    k_perpendicular_min = 0.17890103
    k_perpendicular_max = 37.64856358
    k_perpendicular = np.linspace(k_perpendicular_min, k_perpendicular_max, 12)
    
    # Knowing that mean_ang_separation is always saved in degrees it must be also converted to Mpc/h
    angular_separation_array = pcross_table['mean_ang_separation'] * deg_to_Mpc * h # [Mpc/h]
    
    # Initializing P3D table
    p3d_table = Table()
    p3d_table['k_perpendicular'] = np.array(k_perpendicular)
    p3d_table['k_parallel'] = np.zeros((len(k_perpendicular), len(k_parallel)))
    p3d_table['P3D'] = np.zeros((len(k_perpendicular), len(k_parallel)))
    p3d_table['error_P3D'] = np.zeros((len(k_perpendicular), len(k_parallel)))

    print("Computing P3D from Pcross")
    for ik_par, k_par in enumerate(k_parallel):  # k_par in [h/Mpc]
        
        fig, ([ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9], [ax10, ax11, ax12]) = plt.subplots(4, 3, sharex=True, sharey=True)
        fig.text(0.5, 0.04, '$\\theta$ [Mpc/h]', ha='center')
        ax_list = [[ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9], [ax10, ax11, ax12]]

        for ik_perp, k_perp in enumerate(k_perpendicular):  # k_perp in [h/Mpc]
            
            # P3D computation
            ## Reading Pcross from table
            Pcross = pcross_table['mean_power_spectrum'][:,ik_par]
            errorPcross = pcross_table['error_power_spectrum'][:,ik_par]
            
            ## Interpolating Pcross with a spline function
            spline = scipy.interpolate.UnivariateSpline(angular_separation_array, Pcross)
            ang_sep_min = np.min(angular_separation_array)
            ang_sep_max = np.max(angular_separation_array)
            angular_separation_array_fine_binning = np.linspace(ang_sep_min, ang_sep_max, 1000)
            Pcross_spline = spline(angular_separation_array_fine_binning)
            
            ## Defining integrand
            integrand_Pcross = 2 * np.pi * angular_separation_array_fine_binning * scipy.special.j0(angular_separation_array_fine_binning * k_perp) * Pcross_spline
            
            # Filling subplots
            if ik_perp < 3:
                ax_list[0][ik_perp].scatter(angular_separation_array, Pcross, label='$k_{\perp} = $'+str(k_perp)+' [h/Mpc]')
                ax_list[0][ik_perp].errorbar(angular_separation_array, Pcross,
                xerr = None, yerr = errorPcross, fmt = 'none')
                ax_list[0][ik_perp].plot(angular_separation_array_fine_binning, Pcross_spline, ls='dashed')
                ax_list[0][ik_perp].plot(angular_separation_array_fine_binning, integrand_Pcross)
            elif (ik_perp >= 3) & (ik_perp < 6):
                ax_list[1][ik_perp-3].scatter(angular_separation_array, Pcross, label='$k_{\perp} = $'+str(k_perp)+' [h/Mpc]')
                ax_list[1][ik_perp-3].errorbar(angular_separation_array, Pcross,
                xerr = None, yerr = errorPcross, fmt = 'none')
                ax_list[1][ik_perp-3].plot(angular_separation_array_fine_binning, Pcross_spline, ls='dashed')
                ax_list[1][ik_perp-3].plot(angular_separation_array_fine_binning, integrand_Pcross)
            elif (ik_perp >= 6) & (ik_perp < 9):
                ax_list[2][ik_perp-6].scatter(angular_separation_array, Pcross, label='$k_{\perp} = $'+str(k_perp)+' [h/Mpc]')
                ax_list[2][ik_perp-6].errorbar(angular_separation_array, Pcross,
                xerr = None, yerr = errorPcross, fmt = 'none')
                ax_list[2][ik_perp-6].plot(angular_separation_array_fine_binning, Pcross_spline, ls='dashed')
                ax_list[2][ik_perp-6].plot(angular_separation_array_fine_binning, integrand_Pcross)
            elif (ik_perp >= 9) & (ik_perp < 12):
                ax_list[3][ik_perp-9].scatter(angular_separation_array, Pcross, label='$k_{\perp} = $'+str(k_perp)+' [h/Mpc]')
                ax_list[3][ik_perp-9].errorbar(angular_separation_array, Pcross,
                xerr = None, yerr = errorPcross, fmt = 'none')
                ax_list[3][ik_perp-9].plot(angular_separation_array_fine_binning, Pcross_spline, ls='dashed')
                ax_list[3][ik_perp-9].plot(angular_separation_array_fine_binning, integrand_Pcross)

        custom_ylim = (-0.2, 0.4)
        plt.setp(ax_list, ylim=custom_ylim)
        fig.suptitle('$k_{\parallel}$ = '+str(k_par)+' [h/Mpc]')
        fig.legend(fontsize = 10, loc='upper left', bbox_to_anchor=(1, 0.5))
        fig.show()

