""" This module provides a set of functions to get P3D for a specific (k,mu) 
    
    using either Mcdonald 2001 model or Arinyo-i-prats 2015 model """

import numpy as np
import os, sys
import astropy.io.fits
from astropy.table import Table
from classy import Class
import scipy
from scipy.integrate import quad
import matplotlib.pyplot as plt

sys.path.insert(0, os.environ['HOME']+'/Software/LyaP3D')
from tools import SPEED_LIGHT, LAMBDA_LYA


def init_p_linear(k_max, z, input_params=None):
    """ Runs CLASS for a given k_max
    
    Arguments:
    ----------
    k_max: Float, units: [h/Mpc]
    Max wavenumber we want to reach.
    
    input_params: Dictionary, Default to None which corresponds to using the params hardcoded in the function
    Cosmological parameters used for p_linear computation. Must check class documentation for params specification.
    
    Return:
    -------
    p_linear: 2D array
    First array is the p_linear in [(Mpc/h)^3], second array is the k_array in [h/Mpc].
    """

    if input_params is not None:
        params = input_params
        h = params['h']
    else:
        # Specifying the cosmological parameters
        h = 0.7 #H0/100
        Omega_b = 0.02237/(h**2)
        Omega_m =  0.3153
        Omega_cdm = 0.1198/(h**2)
        A_s = np.exp(3.043)*(1e-10)
        n_s =  0.9652

        # Create a params dictionary
        params = {
                     'output':'mPk',
                     'non linear':'halofit', #emulation of the non linear ps shape (not needed here bcz the desired output is p_linear)
                     'Omega_b':Omega_b,
                     'Omega_m':Omega_m,
                     'h':h,
                     'A_s':A_s,
                     'n_s':n_s,
                     'P_k_max_h/Mpc':k_max * 2, # Only here k_max must be multiplied by 2
                     'z_max_pk':10. # Default value is 10
        }

    # Computation
    cosmo = Class()
    cosmo.set(params)
    print(cosmo.pars)
    cosmo.compute()
    k_array = np.logspace(-5, np.log10(k_max), num=1000) # h Mpc^-1
    print(np.max(k_array))
    p_linear = np.array([cosmo.pk_lin(ki*h, z) for ki in k_array])

    p_k_linear = [k_array, p_linear*h**3]

    return p_k_linear


def p3d_truth_polar(k, mu, p_k_linear, b_delta_squared=None, beta=None, kv=None, q1=None, b_v=None, model='model1'):
    """ Computes the deviation from linear theory due to non-linear evolution and P3D truth computation in polar coordinates

    Arguments:
    ----------
    k: ndarray of floats, or float [h/Mpc]
    Wavenumber value at which we want p3d to be computed, k = np.sqrt((kx**2)+(ky**2)+(kz**2))

    mu: ndarray of floats, or float
    mu value at which we want p3d to be computed, mu = kz/k
    
    p_k_linear: 2D array of floats (p_linear,k) in [(Mpc/h)^3], [h/Mpc] respectively
    Output of init_p_linear, for certain cosmological parameters and k_max = max(k)

    model: String - Default: 'model1'
    Choice of the fitting model we want to use for p3d computation
    2 possible options:
        'model1': According to Mcdonald 2001
        'model2':According to Arinyo-i-prats 2015
        
    Return:
    -------
    p3d_truth: ndarray of floats, or float [(Mpc/h)^3]
    Truth p3d at specific (k,mu) values (polar coordinates)
    """

    k = np.abs(k)
    mu = np.abs(mu)
    
    # Checking for k and mu shapes
    if np.shape(k) != np.shape(mu):
        raise ValueError("k and mu must have the same shape.")

    # p_linear interpolation on the k array given here
    if np.shape(k) == ():
        k = np.array(k)

    if np.max(k) > np.max(p_k_linear[0]):
        print("Warning, p_k_linear will be extrapolated")

    p_linear = np.interp(k, p_k_linear[0], p_k_linear[1])

    # P3D computation
    if model=='model1':
        
        # Isotropic increase in power due to non-linear growth
        k_nl = 6.77 # h/Mpc
        alpha_nl = 0.550 

        # Jeans smoothing
        k_p = 15.0 # h/Mpc
        alpha_p = 2.12

        # LOS broadening
        k_v0 = 0.819 # h/Mpc
        k_prime_v = 0.917 # h/Mpc
        alpha_prime_v = 0.528

        k_v = k_v0 * (1 + (k / k_prime_v))**alpha_prime_v
        k_parallel = np.abs(mu * k)

        alpha_v = 1.5

        # Fitting model computation
        D = np.exp(((k / k_nl)**alpha_nl) - ((k / k_p)**alpha_p) - ((k_parallel / k_v)**alpha_v))
        del(k_v)
        del(k_parallel)
       
        # p3d_truth computation
        b_delta_squared = 0.0173
        beta = 1.58

        p3d_truth = b_delta_squared * ((1 + (beta * mu**2))**2) * p_linear * D
        del(D)
        
    elif model=='model2':
        
        # Isotropic increase in power due to non-linear growth (NL enhancement of PS)
        if q1 == None:
            q1 = 0.666

        q2 = 0

        # LOS broadening
        if kv == None:
            kv = 0.935003735664152

        a_v = 0.561
        kv_to_av = kv**a_v # [h/Mpc]^av
        if b_v is None:
            b_v = 1.58

        # Jeans smoothing
        k_p = 13.5 # h/Mpc
        a_p = 2 # fixed for this model

        # Fitting model computation
        delta_squared = (k**3) * p_linear / (2 * np.pi**2)
        D = np.exp(((q1 * delta_squared) + (q2 * delta_squared**2)) * 
                   (1 - ((k**a_v) / kv_to_av) * mu**b_v) - (k / k_p)**a_p)
        del(delta_squared)
        
        # p3d_truth compuatation:
        
        if b_delta_squared == None:
            # b_delta_squared = (0.5574 * np.log(0.8185))**2
            b_delta_squared = 0.012462846812427325
            
        if beta == None: 
            beta = 1.385

        p3d_truth = b_delta_squared * ((1 + (beta * mu**2))**2) * p_linear * D
        del(D)
    
    if np.any(p3d_truth<0):
        print("Warning, negative P3D")
    
    return p3d_truth


def p3d_truth_cartesian(k_par, k_perp, p_k_linear, b_delta_squared=None, beta=None, kv=None, q1=None, b_v=None, model='model1'):
    """ Computes P3D truth in cartesian coordinates

    Arguments:
    ----------
    k_par: Float or array of floats, units [h/Mpc]
    k_par array, (k_z)

    k_perp: Float or array of floats, units [h/Mpc]
    k_perp array, defined as sqrt(k_x^2 + k_y^2)
    if k_perp and k_par are ndarrays, they must have the same shape
    
    p_k_linear: 2D array of floats (p_linear,k) in [(Mpc/h)^3], [h/Mpc] respectively
    Output of init_p_linear, for certain cosmological parameters and k_max = max(k)

    model: String - Default: 'model1'
    Choice of the fitting model we want to use for p3d computation
    2 possible options:
        'model1': According to Mcdonald 2001
        'model2':According to Arinyo-i-prats 2015
        
    Return:
    -------
    p3d_truth: ndarray of floats, or float [(Mpc/h)^3] (same dimensions as k_par and k_perp)
    Truth p3d at specific (k_par,k_perp) values (cartesian coordinates)
    """

    if np.shape(k_par) != () and np.shape(k_perp) != ():
        if np.shape(k_par) != np.shape(k_perp):
            raise ValueError("Unsupported shapes of k_par and k_perp.")
    
    # k and mu computation for k_par and k_perp
    k = np.sqrt(k_par**2 + k_perp**2)

    mu = np.ones(k.shape)
    if hasattr(k_par,'__len__') is False:
        mu[(k>0)] = k_par / k[(k>0)] # because k_par can be either an array or a float, if float k_par[(k>0)] doesn't work
    else:
        mu[(k>0)] = k_par[(k>0)] / k[(k>0)]

    # p3d_truth computation
    p3d_truth = p3d_truth_polar(k, mu, p_k_linear, b_delta_squared, beta, kv, q1, b_v, model)
    
    return p3d_truth


def compute_pcross_truth(k_par, k_max, ang_sep, p_k_linear, model='model1'):
    """ Computes p_cross_truth from p3d_truth (computed in p3d_truth_polar) by integrating over k_perp for one (k_par,ang_sep) bin
    
    Arguments:
    ----------
    k_par: Float, or ndarray of floats, units: [h/Mpc]
    k_parallel value (equivalently: k_z)
    
    k_max: Float, units: [h/Mpc]
    Max wavenumber we want to reach
    
    ang_sep: Float, or ndarray of floats, units: [Mpc/h]
    Angular separation at which we want to compute the integral, it must have same units as k
    
    p_k_linear: 2D array of floats (p_linear,k) in [(Mpc/h)^3], [h/Mpc] respectively
    Output of init_p_linear, for certain cosmological parameters and k_max = max(k)
    
    model: String - Default: 'model1'
    Choice of the fitting model we want to use for p3d computation
    2 possible options:
        'model1': According to Mcdonald 2001
        'model2':According to Arinyo-i-prats 2015

    Return:
    -------
    p_cross_truth: p_cross(ang_sep,k_par), units [h/Mpc]
    """

    # Must be done if either of k_par and ang_sep are floats and not arrays
    if hasattr(k_par,'__len__') is False:
        k_par = np.array([k_par])
    if hasattr(ang_sep,'__len__') is False:
        ang_sep = np.array([ang_sep])

    # Defining limits of k_perp over which we want to compute the integral
    k_perp_max = np.sqrt(k_max**2 - np.max(k_par)**2)
    k_perp_min = -3
    n_k_perp = 10000
    k_perp = np.logspace(k_perp_min, np.log10(k_perp_max), num=n_k_perp)

    # Computing p3d truth for (k_par, k_perp)
    k_par_grid, k_perp_grid = np.meshgrid(k_par, k_perp)
    p3d_truth_grid = p3d_truth_cartesian(k_par_grid, k_perp_grid, p_k_linear, model)

    p_cross_truth = np.zeros((len(ang_sep), len(k_par)))
    for itheta, theta in enumerate(ang_sep):
        
        # Getting integrand
        integrand_grid = k_perp_grid * scipy.special.j0(k_perp_grid * theta) * p3d_truth_grid / (2 * np.pi)
        
        # Computing p_cross_truth
        integral = np.trapz(integrand_grid, k_perp, axis=0)  # integrate on k_perp
        p_cross_truth[itheta,:] = integral
        
    p_cross_truth = np.squeeze(p_cross_truth)
    
    # Must be done if both k_par and ang_sep are floats and not arrays
    if p_cross_truth.shape == ():
        p_cross_truth = float(p_cross_truth)
        
    return p_cross_truth


def run_compute_pcross_truth(k_parallel, k_max, ang_sep_bin_centers_to_use, units = 'Mpc/h', model='model1'):
    """ Runs compute_pcross_truth for different (k_par,ang_sep) bins 
        and returns a table of pcross_truth(k_parallel) for different angular separation bins
    
    Arguments:
    ----------
    k_parallel: Array of floats, units: [h/Mpc]
    k_parallel array, (k_z)
    
    k_max: Float, units: [1/Mpc]
    Max wavenumber we want to reach with p_k_linear
    
    ang_sep_bin_centers_to_use: Array of floats, units: [degree]
    Angular separations at which we want to compute the integral
    
    model: String - Default: 'model1'
    Choice of the fitting model we want to use for p3d computation
    2 possible options:
        'model1': According to Mcdonald 2001
        'model2':According to Arinyo-i-prats 2015
        
    units: String, Options: 'Mpc/h', 'Angstrom', 'km/s', Default is Mpc/h
    Units in which to compute the truth power spectrum
    
    Return:
    -------
    p_cross_truth_table: Table, p_cross(k_parallel) per angular separation bin, units [h/Mpc]
    Each row corresponds to the truth cross power spectrum in one angular spearation bin
    """
    
    # Initializing table (column names to be changed)
    p_cross_truth_table = Table()
    p_cross_truth_table['ang_sep_bin_centers_used'] = np.array(ang_sep_bin_centers_to_use) # saved in degree
    
    if k_parallel.shape == ():
        p_cross_truth_table['k_parallel'] = np.zeros(len(ang_sep_bin_centers_to_use))
        p_cross_truth_table['p_cross_truth'] = np.zeros(len(ang_sep_bin_centers_to_use))
    else:
        p_cross_truth_table['k_parallel'] = np.zeros((len(ang_sep_bin_centers_to_use), len(k_parallel)))
        p_cross_truth_table['p_cross_truth'] = np.zeros((len(ang_sep_bin_centers_to_use), len(k_parallel)))

    # Computing cosmo used for conversions
    Omega_m=0.3153
    Omega_k=0.
    h = 0.7
    Cosmo = constants.Cosmo(Omega_m, Omega_k, H0=100*h)
    rcomov = Cosmo.get_r_comov
    distang = Cosmo.get_dist_m
    hubble = Cosmo.get_hubble
    
    # Conversion from degree to Mpc
    deg_to_Mpc = distang(2.5) * np.pi / 180 # zbin = 2.5
    ang_sep_bin_centers_to_use_Mpc = ang_sep_bin_centers_to_use * deg_to_Mpc * h # ang sep in Mpc/h not only Mpc
    
    # Computing p_k_linear used for p3d_truth computation below
    p_k_linear = init_p_linear(k_max*h) # Here k must be in Mpc^-1 and returned p_linear is in [Mpc/h]^3 !!!
    # p_k_linear is a 2darray that must be given as it is to p3d_truth_polar function
    
    # Computing p_cross_truth
    print('Computing truth cross power spectrum')
    p_cross_truth = compute_pcross_truth(k_parallel, k_max, ang_sep_bin_centers_to_use_Mpc, p_k_linear, model)
    
    # Filling p_cross_truth_table
    for iang_sep, ang_sep in enumerate(ang_sep_bin_centers_to_use_Mpc): # ang_sep_bin_means same as defined above
        p_cross_truth_table['k_parallel'][iang_sep] = k_parallel
        p_cross_truth_table['p_cross_truth'][iang_sep] = p_cross_truth[iang_sep,:]
            
    print('Truth power spectrum in [Mpc/h] units computation done')
    
    if units != 'Mpc/h':
        print('Converting to Angstrom units')
        conversion_factor = (hubble(2.5) * LAMBDA_LYA) / SPEED_LIGHT # from Angstrom^-1 to Mpc^-1
        p_cross_truth_table['k_parallel'] /= conversion_factor
        p_cross_truth_table['p_cross_truth'] *= conversion_factor
        p_cross_truth_table['k_parallel'] *= h # Angstrom^-1
        p_cross_truth_table['p_cross_truth'] /= h # [Angstrom]
        
        if units == 'km/s':
            print('Converting to km/s units')
            conversion_factor_2 = (1 + 2.5) * LAMBDA_LYA / SPEED_LIGHT # from Angstrom^-1 to [km/s]^-1
            p_cross_truth_table['p_cross_truth'] /= conversion_factor_2 # km/s
            p_cross_truth_table['k_parallel'] *= conversion_factor_2 # k_parallel in [km/s]^-1

    return p_cross_truth_table

