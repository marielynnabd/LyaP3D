""" This module provides a set of functions in order to generate a box of GRF in real space and return a table of LOS """

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


def draw_los(grf_box, los_number, pixel_size, z_box=2.6, noise=0):
    """ Draw LOS from a box of GRF in real space and converts their cartesian coordinates to sky coordinates (ra,dec) in degree
    
    Arguments:
    ----------
    grf_box: 3D matrix of size [Nx, Ny, Nz]
    Box of GRF in real space
    
    los_number: float
    Number of LOS we want to draw
    
    pixel_size: float
    Cell's size in [Mpc/h]
    
    noise: float
    Add white gaussian fluctuations to the deltas: noise = sigma(delta_los) per Angstrom

    Return:
    -------
    all_los_table: Table, one column per LOS
    Table of GRF drawn along randomlxy chosen axes
    """

    # Arrays of x, y and z coordinates
    Nx = len(grf_box[0])
    Ny = len(grf_box[1])
    Nz = len(grf_box[2])
    Nx_array = np.arange(0,Nx,1)
    Ny_array = np.arange(0,Ny,1)
    Nz_array = np.arange(0,Nz,1)

    ## TODO: cosmo pars should be args
    # Computing cosmo used for cartesian to sky coordinates conversion
    Omega_m = 0.3153
    h = 0.7
    cosmo = FlatLambdaCDM(H0=100*h, Om0=Omega_m)

    # Initializing table
    couples_list = []
    all_los_table = Table()
    all_los_table['ra'] = np.zeros(los_number)
    all_los_table['dec'] = np.zeros(los_number)
    all_los_table['redshift'] = np.zeros((los_number, Nz))
    all_los_table['x'] = np.zeros(los_number)
    all_los_table['y'] = np.zeros(los_number)
    all_los_table['z'] = np.zeros((los_number, Nz))
    all_los_table['delta_los'] = np.zeros((los_number, Nz))
    all_los_table['wavelength'] = np.zeros((los_number, Nz))
    
    # Choosing random float values of x and y, interpolating on grid and drawing LOS[x, y, :]
    
    ## Defining interpolation functions on box
    interp_function_delta = scipy.interpolate.RegularGridInterpolator((Nx_array, Ny_array, Nz_array), grf_box)

    ## Drawing LOS and save in table
    j = 0
    while j<los_number:
        # table_test = Table()
        X = np.random.uniform(1, Nx-1)
        Y = np.random.uniform(1, Ny-1)
        couple = (X, Y)
        
        if couples_list.count(couple)==0:
            X_array = np.ones(Nx)*X
            Y_array = np.ones(Ny)*Y
            
            point_positions = np.transpose(np.array([X_array, Y_array, Nz_array])) # Z_array = Nz_array, all the z axis is used always
            delta_los = interp_function_delta(point_positions) 
            
            # Conversion factor from Mpc to degree
            deg_to_Mpc = cosmo.comoving_distance(z_box).value * np.pi / 180
            
            x_coord = X * pixel_size
            y_coord = Y * pixel_size
            
            ra = (X * pixel_size) / (deg_to_Mpc * h)
            dec = (Y * pixel_size) / (deg_to_Mpc * h)
            z = z_box + (cosmo.H(z_box).value * (Nz_array * pixel_size / h) / (SPEED_LIGHT)) # there must be a  / factor 0.7
            
            all_los_table['ra'][j] = ra # degree
            all_los_table['dec'][j] = dec # degree
            all_los_table['redshift'][j,:] = z # redshift
            all_los_table['x'][j] = x_coord # Mpc/h
            all_los_table['y'][j] = y_coord # Mpc/h
            all_los_table['z'][j,:] = Nz_array * pixel_size # Mpc/h
            all_los_table['delta_los'][j,:] = delta_los
            # all_los_table['wavelength [Angstrom]'][j,:] = (1 + z) * lambda_lya
            all_los_table['wavelength'][j,:] = (1 + z) * LAMBDA_LYA
            
            couples_list.append(couple)     
            j += 1
            
        else:
            print('couple already exists')

        if noise>0:
            pixel_size_angstrom = (h * pixel_size) * LAMBDA_LYA * cosmo.H(z).value / SPEED_LIGHT
            noise_per_pixel = noise * np.sqrt(1/pixel_size_angstrom)  # sigma(delta_F) ~ 1/sqrt(pixel size)
            all_los_table['delta_los'] += np.random.normal(scale=noise_per_pixel, size=(los_number, Nz))

    return all_los_table


def run_mock_generation(output_file, Nx, Ny, Nz, pixel_size, los_number, overwrite=True, model='model1'):
    """ Function that generates GRF box and then draws LOS using the above functions
    
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

    