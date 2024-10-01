""" Functions that read Nyx mocks containing LOS transmissions, and adapt them to the input format that QQ accepts """

import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.table import Table, vstack
import fitsio
import sys, os, glob
import multiprocessing
from multiprocessing import Pool
import yaml

sys.path.insert(0, os.environ['HOME']+'/Software/LyaP3D')
from tools import LAMBDA_LYA

from astropy.cosmology import FlatLambdaCDM


def add_missing_args_to_Nyxmock(Nyx_mock_file, replicated_box=False, recompute_radec=False, output_file_name=None):
    """ This function adds the missing arguments 'z_qso', 'qso_id', 'hpix' to Nyxmock, so that it contains the arguments required for QQ
    It is the adapted using the following function
    
    Arguments:
    ----------
    Nyx_mock_file: String
    The mock file containing a fits table with only 1 HDU, where each row corresponds to a QSO, obtained using draw_los in mock_generation (transmissions).

    replicated_box: Same description as in lits_of_allowed_qso function.

    recompute_radec: Bool, default: False
    If True, ra dec coordinates will be recomputed using the assigned z_qso.

    output_file_name: string, default: None
    If provided, it should include the path to outdir and file name in fits.gz format, and the mock will be written to file.
    Otherwise, it will not be written.

    Return:
    -------
    Nyx_mock: Fits table
    It will contain [qso_id, z_qso, ra, dec, hpix, wavelength, transmission_los], same as Nyx_mock in input but with added args.
    """
    
    Nyx_mock = Table.read(Nyx_mock_file)
    lambda_min_mock = np.min(Nyx_mock['wavelength'][0])
    lambda_max_mock = np.max(Nyx_mock['wavelength'][0])
    
    # Computing allowed z_qso
    allowed_z_qso, tid_qso = list_of_allowed_qso(lambda_min_mock, lambda_max_mock, replicated_box)
    
    # Adding z_qso and qso_id
    # TODO: For later, we might want to choose a certain probability of z_qso
    random_index = np.random.choice(len(allowed_z_qso), size=len(Nyx_mock), replace=False)
    Nyx_mock['z_qso'] = allowed_z_qso[random_index]
    Nyx_mock['qso_id'] = tid_qso[random_index]

    # Patching T=1 when > Lya emission peak, just for_qq
    z_qso = Nyx_mock['z_qso'][:, np.newaxis]
    wavelength = Nyx_mock['wavelength']
    Nyx_mock['transmission_los_for_qq'] = Nyx_mock['transmission_los']
    Nyx_mock['transmission_los_for_qq'][wavelength > LAMBDA_LYA * (z_qso + 1)] = 1.0

    # Recomputing ra dec using the z_qso
    if recompute_radec:
        # Computing cosmo
        Omega_m = 0.3153
        h = 0.7
        cosmo = FlatLambdaCDM(H0=100*h, Om0=Omega_m)
        deg_to_Mpc = cosmo.comoving_distance(Nyx_mock['z_qso']).value * np.pi / 180
        Nyx_mock['new_ra'] = Nyx_mock['x'] / (deg_to_Mpc * h)
        Nyx_mock['new_dec'] = Nyx_mock['y'] / (deg_to_Mpc * h)

    # Adding hpix coordinate
    nside = 16
    nest = True
    try:
        Nyx_mock['hpix'] = hp.ang2pix(nside, Nyx_mock['new_ra'], Nyx_mock['new_dec'], nest, lonlat=True)
    except:
        Nyx_mock['hpix'] = hp.ang2pix(nside, Nyx_mock['ra'], Nyx_mock['dec'], nest, lonlat=True)

    if output_file_name is not None:
        Nyx_mock_to_write = fitsio.FITS(output_file_name, 'rw', clobber=True)
        header = [{'name':"FILENAME", 'value': output_file_name}]
        Nyx_mock_to_write.write(Nyx_mock.as_array(), header=header)
        Nyx_mock_to_write.close()

    return Nyx_mock


def list_of_allowed_qso(lambda_min, lambda_max, replicated_box=False):
    """ This function returns the list of allowed z_qso from QSO cat of DESI IRON
    - in a way that their forest covers the forest range of my Nyx box if the LOS is not obtained with a replicated box
    - or just such that they are inside the range of my LOS if replicated_box

    This list will be different as we change the bins

    Arguments:
    ----------
    lambda_min: Float
    Minimum wavelength of the forest in Nyx box
    
    lambda_max: Float
    Maximum wavelength of the forest in Nyx box

    replicated_box: Boolean, default: False
    If replicated box, all qso inside box will be selected

    Return:
    -------
    z_qso_list: Array of floats
    List of allowed z_qso
    
    qso_tid_list: Array of floats
    List of correspodning TARGETIDs. PS: This is just to assign a TARGETID to our QSOs, but doesn't effectively change anything in the computations
    """
    
    # Loading DESI IRON QSO catalog
    qso_cat = Table.read('/global/homes/m/mabdulka/P3D/DESI_IRON_analysis/catalog_iron_v0_qso_target_nobal_BI.fits.gz')

    if replicated_box: # Selecting allowed QSOs with their Lya emission inside replicated box range
        qso_lya_emission = (1 + qso_cat['Z']) * LAMBDA_LYA
        select_qso = (qso_lya_emission > lambda_min) & (qso_lya_emission < lambda_max)
    else: # Selecting allowed QSOs with forests convering the full Nyx box range
        lambda_lyaforest_min = 1060 # Value in rest frame
        lambda_lyaforest_max = 1200 # Value in rest frame
        qso_lambda_lyaforest_min = (1 + qso_cat['Z']) * lambda_lyaforest_min # Observed wavelengths
        qso_lambda_lyaforest_max = (1 + qso_cat['Z']) * lambda_lyaforest_max # Observed wavelengths
        select_qso = (qso_lambda_lyaforest_min < lambda_min) & (qso_lambda_lyaforest_max > lambda_max)

    if np.sum(select_qso)==0:
        print('no qso verifying the condition')
    
    # Allowed z_qso and corresponding tid
    z_qso_list = np.array(qso_cat['Z'][select_qso])
    qso_tid_list = np.array(qso_cat['TARGETID'][select_qso])

    return z_qso_list, qso_tid_list


def filter_mock_per_hpix(mock_file_name, hpix_value):
    """ This funtion reads a mock and gives as output a filtered mock that only contains LOS inside a certain healpix pixel of value hpix_value

    Arguments:
    ----------
    mock_table: String
    Mock file

    hpix_value: Float
    Value of the healpix pixel

    Return:
    -------
    filtered_mock_per_hpix: Table
    Filtered mock table
    """

    mock_table = Table.read(mock_file_name)

    filtered_mock_per_hpix = Table()

    if np.any(mock_table['hpix'] == hpix_value):
        select_pix = (mock_table['hpix'] == hpix_value)
        filtered_mock_per_hpix = vstack([filtered_mock_per_hpix, mock_table[select_pix]])

    return filtered_mock_per_hpix


def adapt_Nyxmock_to_QQ_input(pixels_dict_file_name, outdir, healpix_nside, healpix_nest, use_multiprocessing=False):
    """ This function first reads a mock or several mocks of LOS transmissions created from a Nyx simulation
    using the draw_los function in mock_generation.py and adapts it to the input format accepted by Quickquasars (i.e. fits table to fits image)
    It treats separately the LOS by healpix_pixel subsets and creates separate output transmission files for each healpix_pixel (as required by QQ)
    In the end, it returns 1 transmission file per hpix, combining LOS from several mocks if several mocks were given in the input
    
    PS: The important HDUs to have in the input files given to QQ (i.e. the output file of this function) 
    are the METADATA, WAVELENGTH and F_LYA (or TRANSMISSION) (According to the QQ paper arXiv:2401.00303)
        - METADATA HDU should contain ['RA', 'DEC', 'Z', 'Z_noRSD', 'MOCKID'], with HPXNSIDE, HPXPIXEL, HPXNEST is its header
        - WAVELENGTH HDU should contain a single observed wavelength array
        - F_LYA HDU should contain a matrix of size (N_LOS, len(WAVELENGTH)) where each row corresponds to a LOS Flux

    If there are DLAs, there must be a DLA HDU as well (To be added later, check QQ paper appendix B for reference)

    Arguments:
    ----------
    pixels_dict_file_name: Yaml file name
    File containing all the pixels and for each pixel the corresponding list of Nyx mocks full file names (dir+name) covering the pixel.
    PS: The fits table per mock contains only 1 HDU, where each row corresponds to a QSO
    It must contain [qso_id, z_qso, ra, dec, hpix, wavelength, transmission_los]

    outdir: String
    Directory to store outputs

    healpix_nside: Float
    nside used in the healpy conversion from ra,dec to hpix

    healpix_nest: Boolean
    healpix scheme (usually we use TRUE for nested scheme)

    use_mutliprocessing: Boolean, default: False
    If use_multiprocessing, fir each pixel, the reading of mocks is parallelized. PS: This must not be used if only one mock file is given in input,
    and it is strongly recommended to use it if many mock files are being processed.

    Return:
    -------
    output_fits_image: Fits file
    Fits file with several HDUs: METADATA, WAVELENGTH and F_LYA
    The function might have several outputs depending on the different hpix in the input mock(s), where each of the outputs is written in outdir
    """

    # Loading yaml file
    with open(pixels_dict_file_name, 'r') as file:
        pixels_dict_data = yaml.safe_load(file)

    # Looping over the pixels, one output file will be written at the end of each loop
    for pix in pixels_dict_data:
        if use_multiprocessing: # Check if this part could be improved
            ncpu = multiprocessing.cpu_count()
            with Pool(ncpu) as pool:
                output_filter_mock_per_hpix = pool.starmap(
                    filter_mock_per_hpix,
                    [[mock_file_name, pix] for mock_file_name in pixels_dict_data[str(pix)]]
                )
            pix_mock = vstack([output_filter_mock_per_hpix[i] for i in range(len(output_filter_mock_per_hpix))])
        else:
            pix_mock = Table()
            for mock_file_name in pixels_dict_data[str(pix)]:
                mock_part_in_pix = filter_mock_per_hpix(mock_file_name, pix)
                pix_mock = vstack([pix_mock, mock_part_in_pix])

        pix_N_los = np.sum((pix_mock['hpix'] == pix))
        print('Number of LOS in pixel '+str(pix)+' is:', pix_N_los)

        # Preparing outfiles
        fname = outdir+'/transmission-{}-{}.fits.gz'.format(healpix_nside, pix)
        print('LOS in this pixel will be stored in:', fname)
        output_fits_image = fitsio.FITS(fname, 'rw', clobber=True)

        # METADATA from mock
        try:
            RA = np.array(pix_mock['new_ra'])
            DEC = np.array(pix_mock['new_dec'])
        except:
            RA = np.array(pix_mock['ra'])
            DEC = np.array(pix_mock['dec'])
        Z = np.array(pix_mock['z_qso'])
        Z_noRSD = np.array(pix_mock['z_qso'])
        MOCKID = np.array(pix_mock['qso_id'])
        meta_data_table = [np.float64(RA), np.float64(DEC), np.float64(Z), np.float64(Z_noRSD), MOCKID]
        meta_data_names = ['RA', 'DEC', 'Z', 'Z_noRSD', 'MOCKID']

        # WAVELENGTH from mock
        WAVELENGTH = np.array(pix_mock['wavelength'][0]) # Since all my wavelength arrays are the same

        # F_LYA from mock
        F_LYA = np.array(pix_mock['transmission_los_for_qq'])

        # HEADER
        header_for_all = [{'name':"LYA", 'value': LAMBDA_LYA, 'comment':"LYA wavelength"},
                            {'name':"HPXNSIDE", 'value': healpix_nside, 'comment':"healpix nside parameter"}, 
                            {'name':"HPXPIXEL", 'value': pix, 'comment':"healpix pixel"}, 
                            {'name':"HPXNEST", 'value': healpix_nest, 'comment':"healpix scheme"}]

        # Write to output_fits_image
        output_fits_image.write(meta_data_table, names=meta_data_names, header=header_for_all, extname='METADATA') # METADATA HDU
        output_fits_image.write(np.float32(WAVELENGTH), extname='WAVELENGTH', header=header_for_all) # WAVELENGTH HDU
        output_fits_image.write(np.float32(F_LYA), extname='F_LYA', header=header_for_all) # F_LYA HDU
        output_fits_image.close()


def create_mock_chunk(full_mock_table, z_min, z_max):
    """ This function creates a chunk of the full mock within one redshift bin """

    # Definig mask
    mask = (full_mock_table[0]['redshift'] > z_min) & (full_mock_table[0]['redshift'] < z_max)

    # Creating new_mock chunk
    new_mock = Table()
    new_mock['ra'] = full_mock_table['ra']
    new_mock['dec'] = full_mock_table['dec']
    new_mock['redshift'] = np.zeros((len(full_mock_table), np.sum(mask)))
    new_mock['x'] = full_mock_table['x']
    new_mock['y'] = full_mock_table['y']
    new_mock['z'] = np.zeros((len(full_mock_table), np.sum(mask)))
    new_mock['transmission_los'] = np.zeros((len(full_mock_table), np.sum(mask)))
    new_mock['delta_los'] = np.zeros((len(full_mock_table), np.sum(mask)))
    new_mock['wavelength'] = np.zeros((len(full_mock_table), np.sum(mask)))
    new_mock['z_qso'] = full_mock_table['z_qso']
    new_mock['qso_id'] = full_mock_table['qso_id']
    new_mock['new_ra'] = full_mock_table['new_ra']
    new_mock['new_dec'] = full_mock_table['new_dec']
    new_mock['hpix'] = full_mock_table['hpix']

    for i in range(len(new_mock)):
        for key in ['redshift', 'z', 'transmission_los', 'delta_los', 'wavelength']:
            new_mock[key][i] = full_mock_table[key][i][mask]

    return new_mock

