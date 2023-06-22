""" This module provides a set of functions to get P3D from Pcross computed on data """

import numpy as np
import os, sys
import astropy.io.fits
from astropy.table import Table
import scipy
import matplotlib.pyplot as plt

sys.path.insert(0, os.environ['HOME']+'/Software/picca/py')
from picca import constants
from picca.constants import SPEED_LIGHT # in km/s


def wavenumber_rebin(p3d_table, n_kbins):
    """ This function rebins the 3D power spectrum into k_parallel bins
    
    Arguments:
    ----------
    p3d_table: Table
    Table of P3D
    
    n_kbins: Integer
    Number of k_parallel bins
    
    Return:
    -------
    p3d_table: Table
    Same table as in input, but with rebinned p3d columns added to the table
    """
    
    k_bin_edges = np.logspace(-2, np.log10(np.max(p3d_table['k_parallel'][0])), num=n_kbins) # same units as k_parallel
    k_bin_centers = np.around((k_bin_edges[1:] + k_bin_edges[:-1]) / 2, 5) # same units as k_parallel
    
    p3d_table['k_parallel_rebinned'] = np.zeros((len(p3d_table), len(k_bin_centers))) # same units as k_parallel
    p3d_table['P3D_rebinned'] = np.zeros((len(p3d_table), len(k_bin_centers)))
    p3d_table['error_P3D_rebinned'] = np.zeros((len(p3d_table), len(k_bin_centers)))
    
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


def pcross_to_p3d_cartesian(pcross_table, k_parallel, k_perpandicular, 
                  mean_redshift, method='spline interpolation', smoothing=0, n_angsep=1000, 
                  compute_errors=False, k_binning=False):
    """ This function computes the P3D out of the Pcross:
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
    
    k_parallel: Array
    Array of k_parallel we want to use
    
    k_peprandicular: Array
    Array of k_peprandicular we want to use
    
    mean_redshift: Float
    Central redshift value
    
    method: String, default: 'spline interpolation'
    Pcross interpolation method, before P3D computation
    Options: - 'spline interpolation': interpolating using scipy.interpolate.UnivariateSpline function
             - 'pchip interpolation': interpolating using scipy.interpolate.PchipInterpolator function
             - 'no interpolation': no interpolation, compute integral directly on Pcross output
             
    smoothing: Float, default: 0
    The value of smoothing if 'spline interpolation method' only
    
    n_angsep: Float, default: 1000
    Number of angular separation bins for angular_separation_fine_binning_array required for interpolation methods only

    compute_errors: Boolean, default: False
    Compute error_P3D or not
    
    k_binning: Boolean, Default to False
    Rebin P3D using wavenumber_rebin function
    
    # For units we'll do only in Mpc/h for now
    
    Return:
    -------
    p3d_table: Table
    Table of P3D as function of K_parallel and K_perpandicular."""
    
    # Initializing P3D table
    p3d_table = Table()
    p3d_table['k_perpandicular'] = np.array(k_perpandicular)
    p3d_table['k_parallel'] = np.zeros((len(k_perpandicular), len(k_parallel)))
    p3d_table['P3D'] = np.zeros((len(k_perpandicular), len(k_parallel)))
    p3d_table['error_P3D'] = np.zeros((len(k_perpandicular), len(k_parallel)))
    
    # Computing cosmo used for conversions
    Omega_m=0.3153
    Omega_k=0.
    h = 0.7
    lambda_lya = 1215.67 # Angstrom
    Cosmo = constants.Cosmo(Omega_m, Omega_k, H0=100*h)
    rcomov = Cosmo.get_r_comov
    distang = Cosmo.get_dist_m
    hubble = Cosmo.get_hubble

    # Conversion factor from degree to Mpc
    deg_to_Mpc = distang(mean_redshift) * np.pi / 180
    
    # mean_ang_separation is always saved in degrees it must be always converted to Mpc/h before computation
    angular_separation_array = np.array(pcross_table['mean_ang_separation']) * deg_to_Mpc * h # [Mpc/h]
    
    # If interpolation method:
    if method != 'no interpolation':
        print('Interpolation of Pcross before P3D computation')
        
        ## Define a thinner binning of angular separations used for P3D and errorP3D computation
        ang_sep_min = np.min(angular_separation_array)
        ang_sep_max = np.max(angular_separation_array)
        angular_separation_array_fine_binning = np.linspace(ang_sep_min, ang_sep_max, n_angsep)
        
        ## P3D computation
        for ik_par, k_par in enumerate(k_parallel):  # k_par in [h/Mpc]
            # Reading Pcross from table
            Pcross = np.array(pcross_table['mean_power_spectrum'][:,ik_par])

            # Interpolating Pcross
            if method == 'spline interpolation':
                interpolation_function_Pcross = scipy.interpolate.UnivariateSpline(
                    angular_separation_array, Pcross, s=smoothing)
            else: # this means that method == 'pchip interpolation' since 'no interpolation has been already checked'
                interpolation_function_Pcross = scipy.interpolate.PchipInterpolator(
                    angular_separation_array, Pcross)

            # Pcross_interpolated computation
            Pcross_interpolated = interpolation_function_Pcross(angular_separation_array_fine_binning)

            # Generating n_iterations of random Pcross and interpolating each before error_P3D computation
            if compute_errors == True:
                # Reading error_Pcross from table
                error_Pcross = np.array(pcross_table['error_power_spectrum'][:,ik_par])
                
                # Defining number of iterations of Pcross
                n_iterations = 100
                random_Pcross = np.random.normal(Pcross, error_Pcross, (n_iterations, len(Pcross)))
                random_Pcross_interpolated = np.zeros((n_iterations, n_angsep))
                for i in range(n_iterations):
                    if method == 'spline interpolation':
                        interpolation_function_random_Pcross = scipy.interpolate.UnivariateSpline(
                            angular_separation_array, random_Pcross[i,:], s=smoothing)
                    else: # this means that method == 'pchip interpolation' since 'no interpolation has been already checked'
                        interpolation_function_random_Pcross = scipy.interpolate.PchipInterpolator(
                            angular_separation_array, random_Pcross[i,:])

                    random_Pcross_interpolated[i,:] = interpolation_function_random_Pcross(
                        angular_separation_array_fine_binning)

            for ik_perp, k_perp in enumerate(k_perpandicular): # k_perp in [h/Mpc]
                #  Defining integrand_Pcross
                integrand_Pcross = 2 * np.pi * angular_separation_array_fine_binning * scipy.special.j0(angular_separation_array_fine_binning * k_perp) * Pcross_interpolated

                # Computing integral to get P3D
                P3D = np.trapz(integrand_Pcross, angular_separation_array_fine_binning)

                # Filling table
                p3d_table['k_parallel'][ik_perp,ik_par] = k_par
                p3d_table['P3D'][ik_perp,ik_par] = P3D

                if compute_errors == True:
                    # Defining integrand_random_Pcross
                    integrand_random_Pcross = 2 * np.pi * angular_separation_array_fine_binning * scipy.special.j0(angular_separation_array_fine_binning * k_perp) * random_Pcross_interpolated

                    # Computing integral to get P3D
                    random_P3D = np.trapz(integrand_random_Pcross, angular_separation_array_fine_binning, axis=-1)
                    error_P3D = np.std(random_P3D)
                    
                    # Filling table
                    p3d_table['error_P3D'][ik_perp,ik_par] = error_P3D
             
    else: # if method == 'no interpolation'
        ## P3D computation
        for ik_par, k_par in enumerate(k_parallel):  # k_par in [h/Mpc]
            # Reading Pcross and error_Pcross from table
            Pcross = np.array(pcross_table['mean_power_spectrum'][:,ik_par])
            
            if compute_errors == True:
                error_Pcross = np.array(pcross_table['error_power_spectrum'][:,ik_par])
            
            for ik_perp, k_perp in enumerate(k_perpandicular): # k_perp in [h/Mpc]
                # Defining integrand_Pcross
                integrand_Pcross = 2 * np.pi * angular_separation_array * scipy.special.j0(
                    angular_separation_array * k_perp) * Pcross
                
                # Computing integral to get P3D
                P3D = np.trapz(integrand_Pcross, angular_separation_array)
                
                # Filling table
                p3d_table['k_parallel'][ik_perp,ik_par] = k_par
                p3d_table['P3D'][ik_perp,ik_par] = P3D
                
                if compute_errors == True:
                    # Defining integrand_error_Pcross
                    integrand_error_Pcross = 2 * np.pi * angular_separation_array * scipy.special.j0(
                        angular_separation_array * k_perp) * error_Pcross
                
                    # Computing integral error_P3D
                    error_P3D = np.trapz(integrand_error_Pcross, angular_separation_array)

                    # Filling table
                    p3d_table['error_P3D'][ik_perp,ik_par] = error_P3D
                    
    if k_binning == True:
        n_kbins = 60
        p3d_table = wavenumber_rebin(p3d_table, n_kbins)

    return p3d_table


def plot_integrand(pcross_table):
    """ This function plots integrand used for P3D computation, by fitting a spline funtion to Pcross """
    
    z = 2.59999
    
    # Computing cosmo used for Conversions
    Omega_m=0.3153
    Omega_k=0.
    h = 0.7
    lambda_lya = 1215.67 # Angstrom
    Cosmo = constants.Cosmo(Omega_m, Omega_k, H0=100*h)
    rcomov = Cosmo.get_r_comov
    distang = Cosmo.get_dist_m
    hubble = Cosmo.get_hubble

    # Conversion from degree to Mpc
    deg_to_Mpc = distang(z) * np.pi / 180
    
    # Choice of k_parallel and k_perpandicular
    k_parallel_min = np.min(pcross_table['k_parallel'][0])
    k_parallel_max = np.max(pcross_table['k_parallel'][0])
    k_parallel = np.linspace(k_parallel_min, k_parallel_max, 20)

    
    k_perpandicular_min = 0.17890103
    k_perpandicular_max = 37.64856358
    k_perpandicular = np.linspace(k_perpandicular_min, k_perpandicular_max, 12)
    
    # Knowing that mean_ang_separation is always saved in degrees it must be also converted to Mpc/h
    angular_separation_array = pcross_table['mean_ang_separation'] * deg_to_Mpc * h # [Mpc/h]
    
    # Initializing P3D table
    p3d_table = Table()
    p3d_table['k_perpandicular'] = np.array(k_perpandicular)
    p3d_table['k_parallel'] = np.zeros((len(k_perpandicular), len(k_parallel)))
    p3d_table['P3D'] = np.zeros((len(k_perpandicular), len(k_parallel)))
    p3d_table['error_P3D'] = np.zeros((len(k_perpandicular), len(k_parallel)))

    print("Computing P3D from Pcross")
    for ik_par, k_par in enumerate(k_parallel):  # k_par in [h/Mpc]
        
        fig, ([ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9], [ax10, ax11, ax12]) = plt.subplots(4, 3, sharex=True, sharey=True)
        fig.text(0.5, 0.04, '$\\theta$ [Mpc/h]', ha='center')
        ax_list = [[ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9], [ax10, ax11, ax12]]

        for ik_perp, k_perp in enumerate(k_perpandicular):  # k_perp in [h/Mpc]
            
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

