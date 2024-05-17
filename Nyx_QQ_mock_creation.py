""" Functions that read Nyx mocks containing LOS transmissions, and adapt them to the input format that QQ accepts """

import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.table import Table
import fitsio
import fitsio
import sys, os
sys.path.insert(0, os.environ['HOME']+'/Software/LyaP3D')
from tools import LAMBDA_LYA


def add_missing_args_to_Nyxmock(Nyx_mock_file, )


def adapt_Nyxmock_to_QQ_input(Nyx_mock_file, outdir, healpix_nside, healpix_nest):
    """ This function first reads a mock of LOS transmissions created from a Nyx simulation 
    using the draw_los function in mock_generation.py and adapts it to the input format accepted by Quickquasars (i.e. fits table to fits image)
    It treats separately the LOS by healpix_pixel subsets and creates separate output transmission files for each healpix_pixel (as required by QQ)
    
    PS: The important HDUs to have in the input files given to QQ (i.e. the output file of this function) 
    are the METADATA, WAVELENGTH and F_LYA (or TRANSMISSION) (According to the QQ paper arXiv:2401.00303)
        - METADATA HDU should contain ['RA', 'DEC', 'Z', 'Z_noRSD', 'MOCKID'], with HPXNSIDE, HPXPIXEL, HPXNEST is its header
        - WAVELENGTH HDU should contain a single observed wavelength array
        - F_LYA HDU should contain a matrix of size (N_LOS, len(WAVELENGTH)) where each row corresponds to a LOS Flux

    If there are DLAs, there must be a DLA HDU as well (To be added later, check QQ paper appendix B for reference)

    Arguments:
    ----------
    Nyx_mock_file: String
    The mock file containing a fits table with only 1 HDU, where each row corresponds to a QSO.
    It must contain [qso_id, z_qso, ra, dec, hpix, wavelength, transmission_los]
    
    outdir: String
    Directory to store outputs
    
    healpix_nside: Float
    nside used in the healpy conversion from ra,dec to hpix
    
    healpix_nest: Boolean
    healpix scheme (usually we use TRUE for nested scheme)

    Return:
    -------
    output_fits_image: Fits file
    Fits file with several HDUs: METADATA, WAVELENGTH and F_LYA. 
    The function might several outputs depending on the different hpix in the input mock, where each of the outputs is written in outdir
    """

    # Reading input mock
    Nyx_mock = Table.read(Nyx_mock_file)
    
    # Checking for different hpix
    all_pixels = set(Nyx_mock['hpix'])
    print('This mock contains LOS in the following pixels:', all_pixels)
    
    # Looping over the pixels, one output file will be written at the end of each loop
    for ipix, pix in enumerate(all_pixels):

        # Preparing outfiles
        select_pix = (Nyx_mock['hpix'] == pix)
        print('Number of LOS in pixel '+str(pix)+' is:', np.sum(select_pix))
        # fname = outdir+'/{}/{}/transmission-{}-{}.fits.gz'.format(pix//100, pix, healpix_nside, pix)
        fname = outdir+'/transmission-{}-{}.fits.gz'.format(healpix_nside, pix)
        print('LOS in this pixel will be stored in:', fname)
        output_fits_image = fitsio.FITS(fname, 'rw', clobber=True)

        # Reading the part of the table that only corresponds to pix
        mock_in_pix = Nyx_mock[select_pix]

        # METADATA from mock
        RA = np.array(mock_in_pix['ra'])
        DEC = np.array(mock_in_pix['dec'])
        Z = np.array(mock_in_pix['z_qso'])
        Z_noRSD = np.array(mock_in_pix['z_qso'])
        MOCKID = np.array(mock_in_pix['qso_id'])
        meta_data_table = [np.float32(RA), np.float32(DEC), np.float32(Z), np.float32(Z_noRSD), MOCKID]
        meta_data_names = ['RA', 'DEC', 'Z', 'Z_noRSD', 'MOCKID']

        # WAVELENGTH from mock
        WAVELENGTH = np.array(mock_in_pix['wavelength'][0]) # Since all my wavelength arrays are the same

        # F_LYA from mock
        F_LYA = np.array(mock_in_pix['transmission_los'])

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

