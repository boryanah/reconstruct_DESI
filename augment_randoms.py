import os
import gc
from pathlib import Path

import argparse
import numpy as np
import numba
from astropy.table import Table
import fitsio

DEFAULTS = {}
DEFAULTS['gal_fn'] = "dr9_lrg_pzbins.fits"
DEFAULTS['rand_fn'] = "randoms-1-0.fits"
DEFAULTS['min_nobs'] = 2
DEFAULTS['max_ebv'] = 0.15
DEFAULTS['max_stardens'] = 0

"""
python augment_randoms.py --rand_fn main_randoms-1-0.fits --gal_fn dr9_lrg_pzbins.fits --apply_lrg_mask --remove_islands --pz_bin 1

python augment_randoms.py --rand_fn main_randoms-1-0.fits --gal_fn dr9_lrg_pzbins.fits --apply_lrg_mask --remove_islands

python augment_randoms.py --rand_fn extended_randoms-1-0.fits --gal_fn dr9_extended_lrg_pzbins.fits --apply_lrg_mask --remove_islands
"""

@numba.njit(parallel=True, fastmath=True)
def interpolate_weights(Z, z_means, weights):
    assert len(z_means) == 4 # monotonically increasing
    numba.get_num_threads()
    W = np.zeros(len(Z), dtype=np.float32)
    for i in numba.prange(len(Z)):
        # identify which bin Z belongs to
        if Z[i] < z_means[1]:
            dw = weights[i, 1] - weights[i, 0]
            dz = z_means[1] - z_means[0]
            W[i] = weights[i, 0] + dw/dz * (Z[i] - z_means[0])
        elif Z[i] < z_means[2]:
            dw = weights[i, 2] - weights[i, 1]
            dz = z_means[2] - z_means[1]
            W[i] = weights[i, 1] + dw/dz * (Z[i] - z_means[1])
        else:
            dw = weights[i, 3] - weights[i, 2]
            dz = z_means[3] - z_means[2]
            W[i] = weights[i, 2] + dw/dz * (Z[i] - z_means[2])
    return W 

def main(rand_fn, gal_fn, min_nobs, max_ebv, max_stardens, remove_islands=False, apply_lrg_mask=False, pz_bin=None):
    # attach this to the end of the file
    extra_str = "_z"
    if pz_bin is not None:
        extra_str += f"_bin_{pz_bin:d}"
    
    # mask strings
    isle_str = "_remov_isle" if remove_islands else ""
    lrg_mask_str = "_lrg_mask" if apply_lrg_mask else ""
    
    # selection settings
    mask_str = f"{isle_str}_nobs{min_nobs:d}_ebv{max_ebv:.2f}_stardens{max_stardens:d}{lrg_mask_str}"
    gal_mask_fn = f"{gal_fn.split('.fits')[0]}{mask_str}.npz"
    rand_mask_fn = f"{rand_fn.split('.fits')[0]}{mask_str}.npz"
    save_fn = f"{rand_mask_fn.split('.npz')[0]}{extra_str}.npz"
    print("saving", save_fn)
    
    # randoms and galaxy directories
    rand_dir = Path("/global/cfs/cdirs/desi/users/boryanah/reconstruction_DESI/randoms/")
    gal_dir = Path("/global/cfs/cdirs/desi/users/boryanah/reconstruction_DESI/galaxies/")

    # load randoms
    data_rand = np.load(rand_dir / rand_mask_fn)
    data_gal = np.load(gal_dir / gal_mask_fn)

    if pz_bin is None:
        # generate the redshifts for all photo-z bins
        Z = np.random.choice(data_gal['Z_PHOT_MEDIAN'], size=len(data_rand['RA']), replace=True)
        Z = Z.astype(np.float32)
        del data_gal; gc.collect()

        # calculate weights
        weights = np.zeros((len(Z), 4), dtype=np.float32)
        weights[:, 0] = data_rand['weight_bin_1']
        weights[:, 1] = data_rand['weight_bin_2']
        weights[:, 2] = data_rand['weight_bin_3']
        weights[:, 3] = data_rand['weight_bin_4']
        del data_rand; gc.collect()
        z_means = np.array([0.47499338, 0.6342638, 0.7941149, 0.9208743], dtype=np.float32)
        W = interpolate_weights(Z, z_means, weights)
    else:
        # generate the redshifts for this photo-z bin
        Z = np.random.choice(data_gal['Z_PHOT_MEDIAN'][data_gal['pz_bin'] == pz_bin], size=len(data_rand['RA']), replace=True)
        Z = Z.astype(np.float32)
        del data_gal; gc.collect()
        del data_rand; gc.collect() # weights are in the main file so no need to copy information (think about the environment)
        
    # TODO: could change the way Z is generated, though it seems to be fine
    
    # save final file
    if pz_bin is None:
        np.savez(rand_dir / save_fn, Z=Z, weights=W)
    else:
        np.savez(rand_dir / save_fn, Z=Z)

#>>> cat_rand.keys()
#['RELEASE', 'BRICKID', 'BRICKNAME', 'BRICK_OBJID', 'RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'APFLUX_G', 'APFLUX_R', 'APFLUX_Z', 'APFLUX_IVAR_G', 'APFLUX_IVAR_R', 'APFLUX_IVAR_Z', 'MASKBITS', 'WISEMASK_W1', 'WISEMASK_W2', 'EBV', 'PHOTSYS', 'HPXPIXEL', 'TARGETID', 'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'SUBPRIORITY', 'OBSCONDITIONS', 'PRIORITY_INIT', 'NUMOBS_INIT', 'SCND_TARGET', 'NUMOBS_MORE', 'NUMOBS', 'Z', 'ZWARN', 'TARGET_STATE', 'TIMESTAMP', 'VERSION', 'PRIORITY']

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--rand_fn', help='Randoms file name', type=str, default=DEFAULTS['rand_fn'])
    parser.add_argument('--gal_fn', help='Galaxy file name', type=str, default=DEFAULTS['gal_fn'])
    parser.add_argument('--min_nobs', help='Minimum number of observations per GRZ', type=int, default=DEFAULTS['min_nobs'])
    parser.add_argument('--max_stardens', help='Maximum stellar density', type=int, default=DEFAULTS['max_stardens'])
    parser.add_argument('--max_ebv', help='Maximum EBV value', type=float, default=DEFAULTS['max_ebv'])
    parser.add_argument('--pz_bin', help='Select a single photo-z bin', type=int, default=None)
    parser.add_argument('--apply_lrg_mask', help='Want to apply LRG mask', action='store_true')
    parser.add_argument('--remove_islands', help='Want to remove islands', action='store_true')
    args = vars(parser.parse_args())
    main(**args)
