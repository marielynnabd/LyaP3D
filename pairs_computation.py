""" This module provides a set of functions to get pairs of LOS from a data set, having a certain angular separation """

import numpy as np
import sys, os
import glob
from astropy.table import Table, vstack
import multiprocessing
from multiprocessing import Pool


def get_pairs_single_los(i_los, all_los_table, ang_sep_max, radec_names=['ra', 'dec'], preselect_angsep=False, select_halfplate=None):
    """ - This function gets possible pairs for single los:
            - Takes a Mock or Real data LOS table
            - Returns a table of possible pairs with a max angular separation
    
    Arguments:
    ----------
    i_los: Integer
    Index of LOS with which we want to pair other LOS
    
    all_los_table: Table
    Table of LOS we want to use in the analysis: - It accepts mock LOS or real data
                                                 - It accepts Tables of LOS having the full LOS info including DELTAS 
                                                 (Mocks or real data files produced by eBOSS_dr16_analysis/DESI_Y1_analysis), 
                                                 OR just LOS info files that do not include the DELTAS 
                                                 (for example those produced by LOS_info_DESI_Y1_analysis/LOS_info_QQ_Y1_analysis, 
                                                 noting that these codes were made just to simplify and accelerate pairs counting, 
                                                 however, to do a full Pcross/P3D measurement, files with full info are required)
                                                 PS: Nyx mocks already have the good format to be read by this code, and they include all info including DELTAS

    ang_sep_max: Float, units = degree or Mpc/h
    Maximum angular separation possible between two LOS so that they form a pair
    
    radec_names: List of str, Default: ['ra', 'dec']
    ra dec keys in mocks or data table
    Options: - ['ra', 'dec']: my mocks
             - ['RA', 'DEC']: eBOSS data
             - ['TARGET_RA', 'TARGET_DEC']: IRON
             - ['x', 'y']
    
    preselect_angsep: Bool
    first crude preselection of angular pairs, to gain CPU. Use with a lot of caution

    select_halfplate: string or None
    Option specific to the case of SDSS data (could easily be extended to the case of DESI)
    Options: - 'same': keep only pairs with same PLATE and FIBERID the the same group (0-499), (500-1000)
             - 'different': opposite of 'same'

    Return:
    -------
    los_pairs_table: Table
    Each row corresponds to the indices of the pixels forming the pair, and the angular separation between them
    """

    ra = radec_names[0]
    dec = radec_names[1]
    
    # Initializing los_pairs_table
    single_los_pairs_table = Table()
    
    if radec_names == ['x', 'y']:
        cos_los = 1
    else:
        cos_los = np.cos(all_los_table[dec][i_los] * np.pi / 180)

    los_number = len(all_los_table)
    dalpha = all_los_table[ra][i_los+1:los_number] - all_los_table[ra][i_los]
    ddelta = all_los_table[dec][i_los+1:los_number] - all_los_table[dec][i_los]
    i_neighbors = np.arange(i_los+1, los_number)

    # Selecting a radius of 2 degrees/200 Mpc around the pixel (First selection)
    if preselect_angsep:
        if radec_names == ['x', 'y']:
            ang_sep_max_rough = 200
        else:
            ang_sep_max_rough = 2
        if ang_sep_max_rough < 2*ang_sep_max:
            raise ValueError('ang_sep_max is certainly too large.')
        mask1 = (dalpha < ang_sep_max_rough) & (ddelta < ang_sep_max_rough)
        dalpha, ddelta, i_neighbors = dalpha[mask1], ddelta[mask1], i_neighbors[mask1]

    ang_sep_local = (dalpha * cos_los)**2 + ddelta**2
    ang_sep_local = np.sqrt(ang_sep_local)

    # Selecting pairs with angular separation < max angular separation we want (Second selection)
    mask2 = (ang_sep_local <= ang_sep_max)
    i_neighbors = i_neighbors[mask2]
    ang_sep_local = ang_sep_local[mask2]
    
    # (SDSS) half-plate pair selection
    if select_halfplate is not None:
        plate_1 = all_los_table['PLATE'][i_los]
        fibergroup_1 = 1 if all_los_table['FIBERID'][i_los]<500 else 2
        plates = all_los_table['PLATE'][i_neighbors]
        fibergroups = np.ones(len(plates), dtype='>i4')
        fibers = all_los_table['FIBERID'][i_neighbors]
        m = (fibers>500)  # fibers are in the range [1 - 1000]
        fibergroups[m] = 2
        if select_halfplate=='same':
            mask3 = (plates==plate_1)&(fibergroups==fibergroup_1)
        elif select_halfplate=='different':
            mask3 = (plates!=plate_1)|(fibergroups!=fibergroup_1)
        else:
            raise ValueError('Wrong select_halfplate argument')
        i_neighbors = i_neighbors[mask3]
        ang_sep_local = ang_sep_local[mask3]

    # Filling table
    single_los_pairs_table['index_los1'] = np.ones(len(i_neighbors), dtype='int') * i_los
    single_los_pairs_table['index_los2'] = i_neighbors
    single_los_pairs_table['ang_separation'] = ang_sep_local
    
    return single_los_pairs_table


def compute_pairs(all_los_table, ang_sep_max, radec_names=['ra', 'dec'], ncpu='all', outputfile=None, preselect_angsep=False, select_halfplate=None):

    # Getting los_pairs_table from mock 
    if radec_names == ['x', 'y']:
        print('Getting pairs with angular separation < '+str(ang_sep_max)+' Mpc/h')
    else:
        print('Getting pairs with angular separation < '+str(ang_sep_max)+' degrees')

    if ncpu=='all':
        ncpu = multiprocessing.cpu_count()
    print("Number of cpus:", multiprocessing.cpu_count())

    with Pool(ncpu) as pool:
        output_get_pairs_single_los = pool.starmap(
            get_pairs_single_los, [[i, all_los_table, ang_sep_max, radec_names, preselect_angsep, select_halfplate] for i in range(len(all_los_table))])
    output_get_pairs_single_los = [x for x in output_get_pairs_single_los if x is not None] # For sanity check
    all_los_pairs_table = vstack([output_get_pairs_single_los[i] for i in range(len(output_get_pairs_single_los))])

    if outputfile is not None:
        all_los_pairs_table.write(outputfile)
    
    return all_los_pairs_table

