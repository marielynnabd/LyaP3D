""" This module provides a set of functions in order to generate a box of GRF in real space and return a table of LOS (a mock)
    More importantly, it is used to draw_los from Nyx_simulation_boxes
"""

import numpy as np
import sys, os
import copy
from astropy.table import Table
import fitsio
import scipy
from astropy.cosmology import FlatLambdaCDM

from truth_p3d_computation import init_p_linear, p3d_truth_polar
from tools import SPEED_LIGHT, LAMBDA_LYA


def generate_box(Nx, Ny, Nz, pixel_size, model='model1'):
    """ Gerenate GRF box in real space
    PS: This function was tested and used to create grf mocks but we stopped this analysis and moved to creating mocks using Nyx simulation boxes, with the functions below

    Arguments:
    ----------
    Nx, Ny, Nz: Floats
    Box size
    
    pixel_size: Float
    Cell's size in [Mpc/h]
    
    model: String - Default: 'model1'
    Choice of the fitting model we want t o use for p3d computation
    2 possible options:
        'model1': According to Mcdonald 2001
        'model2':According to Arinyo-i-prats 2015
    
    Return:
    -------
    grf_box: 3D matrix of size [Nx, Ny, Nz]
    box of GRF in real space
    """
    
    h = 0.7 #H0/100
    
    # Initializing box in real space
    box = np.zeros((Nx, Ny, Nz), dtype = np.float32)
    # Generating GRF in real space
    box[:,:,:] = np.float32(np.random.normal(size=(Nx, Ny, Nz)))
    
    # FFT(box) to get a box of GRF in Fourier space
    boxk = np.fft.fftn(box)
    del(box)
    
    # Getting kz_array by FFT
    kz_array = np.fft.fftfreq(Nz, pixel_size) # [h/Mpc]
    kx_array = np.fft.fftfreq(Nx, pixel_size)
    ky_array = np.fft.fftfreq(Ny, pixel_size)
    
    # Creating meshgrids
    Kx, Ky, Kz = np.meshgrid(kx_array, ky_array, kz_array, indexing='ij')
    
    # K Computation
    K = np.sqrt((Kx**2)+(Ky**2)+(Kz**2))
    del(Kx)
    del(Ky)
    
    # Mu Computation
    Mu = np.zeros(K.shape)
    Mu[(K>0)] = Kz[(K>0)] / K[(K>0)] # compute mu when k is diff than zero, otherwise mu is 0 and box will be zero to
    # we do that to avoid nan in our code
    del(Kz)
    
    # Linear power spectrum computation
    p_linear = init_p_linear(2 * np.max(K) * h) # function takes k in Mpc^-1 but retrurns k and plin in h/Mpc and (Mpc/h)^3
    
    # P3D computation
    p3d = p3d_truth_polar(K, Mu, p_linear, model)
    
    # Box parametrization (*sqrt(P3D/cell_volume))
    boxk *= np.sqrt(p3d)
    boxk /= np.sqrt(pixel_size**3)
    
    # Inverse FFT to get box of GRF in real space
    grf_box = np.fft.ifftn(boxk)
    if np.nanmean(np.abs(grf_box.imag/grf_box.real)) > 1.e-8:
        print("Warning, grf_box has significant Imaginary part.")
    
    grf_box = grf_box.real
    
    return grf_box


def run_mock_generation(output_file, Nx, Ny, Nz, pixel_size, los_number, overwrite=True, model='model1'):
    """ Function that generates GRF box and then draws LOS using the generate_box and draw_los functions
    PS: This function was tested and used to create grf mocks but we stopped this analysis and moved to creating mocks using Nyx simulation boxes, with the functions below

    Arguments:
    ----------
    output_file: str
    Outputfile name 'outputfile_name.fits.gz'
    
    Nx, Ny, Nz: int
    Box size, default is 768
    
    pixel_size: float
    Cell's size in [Mpc/h], default 0.1
    
    overwrite: bool
    Overwrite output
    
    model: String - Default: 'model1'
    Choice of the fitting model we want to use for p3d computation
    2 possible options:
        'model1': According to Mcdonald 2001
        'model2':According to Arinyo-i-prats 2015
    
    Return:
    -------
    all_los_table: Table, one column per LOS
    Table of GRF drawn along randomlxy chosen axes
    """

    print('Generating mock with model '+str(model)+' used')
    
    if os.path.exists(output_file) and not overwrite:
        raise RuntimeError('Output file already exists: ' + output_file)
    
    grf_box = generate_box(Nx, Ny, Nz, pixel_size, model)
    print(grf_box)
    los_table = draw_los(grf_box, los_number, pixel_size)
    print(los_table['delta_los'])
    
    all_los_table = fitsio.FITS(output_file, 'rw', clobber=True)
    all_los_table.write(los_table.as_array())
    all_los_table.close()


def rescale_tau(tau_grid, mean_flux_goal, return_scaling_factor=False, verbose=True):
    """ This function rescales the tau grid to a goal value

    Arguments:
    ----------
    tau_grid: 3D matrix of size [Nx, Ny, Nz]
    Grid of optical depth tau

    mean_flux_foal: Float
    The desired value of mean flux after rescaling. It must be between 0 and 1

    return_scaling_factor: Boolean
    If true, the scaling_factor will also be returned in output

    Return:
    -------
    tau_rescaled: 3D matrix of size [Nx, Ny, Nz]
    Grid of rescaled optical depth
    """

    def _residuals(param, f, x, y):
        return y - f(param, x)

    def _chi2(param, f, x, y, error=None, verbose=False):
        if error is None:
            tmp = y - f(param, x)
            if verbose:
                print(param, y, tmp)
            return tmp
        else:
            tmp = (y - f(param, x)) / np.sqrt(error)
            if verbose:
                print(param, y, tmp)
            return tmp

    def fit(f, x, y, p0, error=None):
        if error is None:
            args = (f, x, y)
            # param = scipy.optimize.leastsq(_chi2, p0, args(f, x, y))
            param, _ = scipy.optimize.leastsq(_chi2, p0, args=args)
            res = _residuals(param, f, x, y)
        else:
            args = (f, x, y, error)
            # param = scipy.optimize.leastsq(_chi2, p0, args(f, x, y, error))
            param, _ = scipy.optimize.leastsq(_chi2, p0, args=args)
            res = _residuals(param, f, x, y)
        return param, res

    tau_goal = -np.log(mean_flux_goal)
    f = lambda param, x: -1.0 * np.log(np.mean(np.exp(-1.0 * param * x)))
    param, _ = fit(f, tau_grid, tau_goal, 0)
    scaling_factor = param

    tau_rescaled = tau_grid * scaling_factor

    if verbose:
        print('Rescale flux:')
        print('  Original tau_array had mean flux =', np.mean(np.exp(-tau_grid)))
        print('  Goal mean flux =', mean_flux_goal)
        print('  Rescaled tau has mean flux = ', np.mean(np.exp(-tau_rescaled)))

    if return_scaling_factor:
        return tau_rescaled, scaling_factor
    else:
        return tau_rescaled


def preprocess_simulation_tau_grid(tau_grid, out_type, tau_rescaling=False, mean_flux_goal=None):
    """ This function: 
            - First, takes the 3D tau grid of a hydrodynamical simulation (e.g. Nyx in our case) and rescales it to a goal value if 'rescale_tau' in args
            - It comuptes the transmission/F = exp(-tau) (/tau_rescaled)
            - If out_type is 'transmissions' it returns F
            - Or if out_type is 'deltas' it returns delta = F / mean(F) - 1
        PS: The rescaling done here is done for the same mean_flux_goal as function of redshift

    Arguments:
    ----------
    tau_grid: 3D matrix of size [Nx, Ny, Nz]
    Grid of optical depth tau
    
    out_type: String
    The type of the output, options: 'transmissions', 'deltas'
    
    rescale_tau: Boolean, default: False
    If rescale_tau, the tau grid will be rescaled

    mean_flux_foal: Float, default: None
    The desired value of mean flux after rescaling. It must be between 0 and 1

    Return:
    -------
    tau_rescaled: 3D matrix of size [Nx, Ny, Nz]
    Grid of rescaled optical depth
    """

    if tau_rescaling:
        if mean_flux_goal is None:
            raise ValueError("mean_flux_foal cannot be None when rescale_tau is True")
        else:
            rescaled_tau_grid = rescale_tau(tau_grid, mean_flux_goal)
            del(tau_grid)
        transmissions_grid = np.exp(-1.0 * rescaled_tau_grid)
    else:
        transmissions_grid = np.exp(-1.0 * tau_grid)

    if out_type == 'transmissions':
        return transmissions_grid
    elif out_type == 'deltas':
        return (transmissions_grid / transmissions_grid.mean()) - 1


def draw_los(box, box_type, los_number, pixel_size, z_box, ra_start=0, dec_start=0, noise=0, for_qq=False, n_replic=1, tau_rescaling_with_redshift=False, rescaling_factor_list=None, output_file_name=None):
    """ Draw LOS from a simulation_Transmissions/simulation_Deltas/GRF box in real space and converts their cartesian coordinates to sky coordinates (ra,dec) in degree.
    PS: this code draws LOS from the provided box without changing the type, i.e. if the box is a Deltas box, delta_los will be stored in all_los_table output,
    and if the box is a Transmissions box, transmission_los will be stored in all_los_table output.
    
    Arguments:
    ----------
    box: 3D matrix of size [Nx, Ny, Nz]
    Box could be a GRF box (deltas), Transmissions of a simulation box, or Deltas of a simulation box = F/mean(F) - 1 (usually in real space).
    
    box_type: string
    Type of the box, could be: 'transmissions', 'deltas' (Deltas accounts for both GRF boxes or Deltas from simulations).
    
    los_number: float
    Number of LOS we want to draw.
    
    pixel_size: float
    Cell's size in [Mpc/h].
    
    z_box: float
    Redshift of box.
    
    ra_start: float, default: 0
    Starting point of the ra coordinate used for tiling over DESI footprint.

    dec_start: float, default: 0
    Starting point of the dec coordinate used for tiling over DESI footprint.

    noise: float
    Add white gaussian fluctuations to the deltas: noise = sigma(delta_los) per Angstrom.

    for_qq: boolean
    If for qq the box will have a non linear redshift and wavelength distribution.

    n_replic: float, default: 1
    Number of replications of box we want in order to extend our LOS.

    tau_rescaling_with_redshift: boolean, default: false
    Rescale tau as function of redshift.

    rescaling_factor_list: list
    Rescaling factor used to rescale tau of the drawn los as function of z, it is computed using the rescale_tau function
    First column is flux_goal_array at which the rescaling factor was computed
    Second column is rescaling factor
    PS: This rescaling factor is computed from a specific simulation box, i.e. the rescaling factor list must be the one corresponding to the box used to draw los here.

    output_file_name: string, default: None
    If provided, it should include the path to outdir and file name in fits.gz format, and the mock will be written to file.
    Otherwise, it will not be written.

    Return:
    -------
    all_los_table: Table, one column per LOS
    Table of LOS drawn along randomly chosen axes
    """

    # Arrays of x, y and z coordinates
    Nx = len(box[0])
    Ny = len(box[1])
    Nz = len(box[2])
    Nx_array = np.arange(0, Nx, 1)
    Ny_array = np.arange(0, Ny, 1)
    Nz_array = np.arange(0, Nz, 1)

    if n_replic != 1:
        Nz_array_for_replic = np.arange(0, n_replic * Nz, 1)

    ## TODO: cosmo pars should be args
    # Computing cosmo
    Omega_m = 0.3153
    h = 0.7
    cosmo = FlatLambdaCDM(H0=100*h, Om0=Omega_m)

    # Defining redshift array
    if for_qq: # Here redshift array will be defined non-linearly (equivalently for wavelength)
        redshift_array = np.zeros((n_replic * Nz))
        redshift_array[0] = z_box - 0.1
        for i in range(1, len(redshift_array)):
            redshift_array[i] = redshift_array[i-1] + (cosmo.H(redshift_array[i-1]).value * (pixel_size / h) / (SPEED_LIGHT))
    else: # Here redshift will be linear (equivalently for wavelength)
        if n_replic != 1: # In the replicate case, must use Nz_array_for_replic
            redshift_array = z_box + (cosmo.H(z_box).value * (Nz_array_for_replic * pixel_size / h) / (SPEED_LIGHT)) # there must be a  / factor 0.7
        else: # This is the usual case with no replic and not for qq case
            redshift_array = z_box + (cosmo.H(z_box).value * (Nz_array * pixel_size / h) / (SPEED_LIGHT)) # there must be a  / factor 0.7

    # Preparing for tau rescaling
    if tau_rescaling_with_redshift and box_type=='transmissions':
        flux_goal_los = np.exp(-25e-4 * (1 + redshift_array)**3.7)
        rescaling_factor_los = np.interp(flux_goal_los, rescaling_factor_list[:,0], rescaling_factor_list[:,1])
    elif tau_rescaling_with_redshift and box_type=='deltas':
        print('Warning, input box type is not transmissions, therefore no rescaling could be applied')

    # Initializing table of lines-of-sight
    couples_list = []
    all_los_table = Table()
    all_los_table['ra'] = np.zeros(los_number)
    all_los_table['dec'] = np.zeros(los_number)
    all_los_table['redshift'] = np.zeros((los_number, n_replic * Nz))
    all_los_table['x'] = np.zeros(los_number)
    all_los_table['y'] = np.zeros(los_number)
    all_los_table['z'] = np.zeros((los_number, n_replic * Nz))

    if box_type == 'transmissions': # In this case we would like to save both transmission_los and delta_los
        all_los_table['transmission_los'] = np.zeros((los_number, n_replic * Nz))
        all_los_table['delta_los'] = np.zeros((los_number, n_replic * Nz))
    elif box_type == 'deltas': # In this case the output wil only be delta_los since from a deltas box we can't go back to compute transmission_los
        all_los_table['delta_los'] = np.zeros((los_number, n_replic * Nz))
    all_los_table['wavelength'] = np.zeros((los_number, n_replic * Nz))

    # Choosing random float values of x and y, interpolating on grid and drawing LOS[x, y, :]
    
    ## Defining interpolation functions on box
    interp_function_box = scipy.interpolate.RegularGridInterpolator((Nx_array, Ny_array, Nz_array), box)
    if box_type == 'transmissions':
        mean_box_flux = box.mean()
        delta_box = (box / mean_box_flux) - 1
        interp_function_delta_box = scipy.interpolate.RegularGridInterpolator((Nx_array, Ny_array, Nz_array), delta_box)

    ## Drawing LOS and saving in table
    j = 0
    while j<los_number:
        X = np.random.uniform(1, Nx-1)
        Y = np.random.uniform(1, Ny-1)
        couple = (X, Y)
        
        if couples_list.count(couple)==0:
            X_array = np.ones(Nx) * X
            Y_array = np.ones(Ny) * Y
            
            # transmission_los and/ delta_los computation and point_positions
            point_positions = np.transpose(np.array([X_array, Y_array, Nz_array])) # Z_array = Nz_array, all the z axis is used always
            los_at_point_positions = interp_function_box(point_positions) # = delta_los if deltas box and = transmission_los if transmissions box
            if n_replic != 1:
                los_at_point_positions = np.tile(los_at_point_positions, n_replic)
            if box_type == 'transmissions':
                delta_los_at_point_positions = interp_function_delta_box(point_positions)
                if n_replic != 1:
                    delta_los_at_point_positions = np.tile(delta_los_at_point_positions, n_replic)

            # Rescaling tau as function of redshift, in this case both transmission_los and delta_los will be recomputed
            # otherwise, only those from interpolation are saved in table
            if tau_rescaling_with_redshift and box_type == 'transmissions':
                tau_los = -np.log(los_at_point_positions)
                rescaled_tau_los = tau_los * rescaling_factor_los
                los_at_point_positions = np.exp(-rescaled_tau_los) # los_at_point_positions is a T array here
                # Computing delta_los_at_point_positions from rescaled los_at_point_positions
                delta_los_at_point_positions = (los_at_point_positions / flux_goal_los) - 1

            # Conversion factor from Mpc to degree
            deg_to_Mpc = cosmo.comoving_distance(z_box).value * np.pi / 180
            
            # LOS coordinates computation (x,y) and (ra,dec)
            ## Coordinates that fit in box or replicated box
            x_coord = (X * pixel_size)
            y_coord = (Y * pixel_size)

            ## Corresponding ra and dec coordinates with or without shifting over footprint
            ra = (x_coord) / (deg_to_Mpc * h) + ra_start
            dec = (y_coord) / (deg_to_Mpc * h) + dec_start

            ## x and y coordinates after having shifted ra, dec, = x_coord and y_coord if ra_start and dec_start = 0
            x_coord_shifted = ra * deg_to_Mpc * h
            y_coord_shifted = dec * deg_to_Mpc * h

            # Filling table
            all_los_table['ra'][j] = ra # degree
            all_los_table['dec'][j] = dec # degree
            all_los_table['redshift'][j,:] = redshift_array # redshift
            all_los_table['x'][j] = x_coord_shifted # Mpc/h
            all_los_table['y'][j] = y_coord_shifted # Mpc/h

            if n_replic != 1:
                all_los_table['z'][j,:] = Nz_array_for_replic * pixel_size # Mpc/h
            else:
                all_los_table['z'][j,:] = Nz_array * pixel_size # Mpc/h

            if box_type == 'transmissions':
                all_los_table['transmission_los'][j,:] = los_at_point_positions
                all_los_table['delta_los'][j,:] = delta_los_at_point_positions
            elif box_type == 'deltas':
                all_los_table['delta_los'][j,:] = los_at_point_positions
            all_los_table['wavelength'][j,:] = (1 + redshift_array) * LAMBDA_LYA
            
            couples_list.append(couple)     
            j += 1
            
        else:
            print('couple already exists')

    if noise>0:
        pixel_size_angstrom = (pixel_size / h) * LAMBDA_LYA * cosmo.H(z).value / SPEED_LIGHT
        noise_per_pixel = noise * np.sqrt(1 / pixel_size_angstrom)  # sigma(delta_F) ~ 1/sqrt(pixel size)
        all_los_table['delta_los'] += np.random.normal(scale=noise_per_pixel, size=(los_number, Nz))

    if output_file_name is not None:
        Nyx_mock = fitsio.FITS(output_file_name, 'rw', clobber=True)
        Nyx_mock.write(all_los_table.as_array())
        Nyx_mock.close()

    return all_los_table

