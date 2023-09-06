import os
import gc
from pathlib import Path

import argparse
import numpy as np
import yaml
from astropy.table import Table, hstack
import fitsio

DEFAULTS = {}
DEFAULTS['fn'] = "dr9_lrg_pzbins.fits"
DEFAULTS['min_nobs'] = 2
DEFAULTS['max_ebv'] = 0.15
DEFAULTS['max_stardens'] = 0 # try to find

"""
python clean_catalog.py --fn main_randoms-1-0.fits --apply_lrg_mask --remove_islands --want_weights

python clean_catalog.py --fn extended_randoms-1-0.fits --apply_lrg_mask --remove_islands --want_weights

python clean_catalog.py --fn dr9_lrg_pzbins.fits --apply_lrg_mask --remove_islands --want_weights

python clean_catalog.py --fn dr9_extended_lrg_pzbins.fits --apply_lrg_mask --remove_islands --want_weights
"""

def main(fn, min_nobs, max_ebv, max_stardens, remove_islands=False, apply_lrg_mask=False, want_weights=False):
    # are you cleaning galaxies or randoms
    if "randoms" in fn:
        want_randoms = True
    else:
        want_randoms = False

    # mask strings
    isle_str = "_remov_isle" if remove_islands else ""
    lrg_mask_str = "_lrg_mask" if apply_lrg_mask else ""
    
    # selection settings
    mask_str = f"{isle_str}_nobs{min_nobs:d}_ebv{max_ebv:.2f}_stardens{max_stardens:d}{lrg_mask_str}"
    save_fn = f"{fn.split('.fits')[0]}{mask_str}.npz"
    print("saving", save_fn)
    
    # load and save directories
    if want_randoms:
        load_dir = Path("/global/cfs/cdirs/desi/public/ets/target/catalogs/dr9/0.49.0/randoms/resolve")
        save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/reconstruction_DESI/randoms/")
        read_fn = "_".join(fn.split("_")[1:])
    else:
        load_dir = Path("/global/cfs/cdirs/desi/users/rongpu/lrg_xcorr/data_products/catalogs/")
        save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/reconstruction_DESI/galaxies/")
        read_fn = fn
        
    # load catalog
    cat = Table(fitsio.read(load_dir / read_fn))
    os.makedirs(save_dir, exist_ok=True)

    # find the mean
    if not want_randoms:
        for bin_index in range(1, 5):  # 4 bins
            choice = cat['pz_bin'] == bin_index
            print("bin, number of objs, median, mean", bin_index, np.sum(choice), np.median(cat['Z_PHOT_MEDIAN'][choice]), np.mean(cat['Z_PHOT_MEDIAN'][choice]))
        print("number of objs, median, mean", len(cat), np.median(cat['Z_PHOT_MEDIAN']), np.mean(cat['Z_PHOT_MEDIAN']))
              
    # initialize mask
    mask = np.ones(len(cat['DEC']), dtype=bool)
    
    if remove_islands:
        # Remove "islands" in the NGC
        mask &= ~((cat['DEC'] < -10.5) & (cat['RA'] > 120) & (cat['RA'] < 260))
        print('Remove islands', np.sum(mask), np.sum(~mask), np.sum(mask)/len(mask))
    
    if min_nobs > 0:
        # NOBS cut
        if want_randoms:
            mask &= (cat['NOBS_G'] >= min_nobs) & (cat['NOBS_R'] >= min_nobs) & (cat['NOBS_Z'] >= min_nobs)
        else:
            mask &= (cat['PIXEL_NOBS_G'] >= min_nobs) & (cat['PIXEL_NOBS_R'] >= min_nobs) & (cat['PIXEL_NOBS_Z'] >= min_nobs)
        print('NOBS', np.sum(mask), np.sum(~mask), np.sum(mask)/len(mask))

    if apply_lrg_mask:
        # Apply LRG mask
        if want_randoms:
            mask_dir = Path("/global/cfs/cdirs/desi/users/rongpu/lrg_xcorr/data_products/catalogs/lrgmask_v1.1/")
            mask &= Table(fitsio.read(mask_dir / (read_fn.split('.fits')[0]+"-lrgmask_v1.1.fits.gz")))['lrg_mask'] == 0
        else:
            mask &= cat['lrg_mask'] == 0
        print('LRG mask', np.sum(mask), np.sum(~mask), np.sum(mask)/len(mask))

    if max_ebv < 1.: # not sure if that's the maximum
        # EBV cut
        mask &= cat['EBV'] < max_ebv
        print('EBV', np.sum(mask), np.sum(~mask), np.sum(mask)/len(mask))

    if max_stardens > 0:
        # STARDENS cut ASK RONGPU!
        stardens = np.load(misc_dir / 'pixweight-dr7.1-0.22.0_stardens_64_ring.npy')  # Stellar density map
        stardens_nside = 64
        mask = stardens >= max_stardens
        bad_hp_idx = np.arange(len(stardens))[mask]
        cat_hp_idx = hp.pixelfunc.ang2pix(stardens_nside, cat['RA'], cat['DEC'], lonlat=True, nest=False)
        mask &= ~np.in1d(cat_hp_idx, bad_hp_idx)
        print('STARDENS', np.sum(~mask), np.sum(mask), np.sum(mask)/len(mask))

    if want_weights:
        # tuks randoms
        weights_dir = Path("/global/cfs/cdirs/desi/users/rongpu/lrg_xcorr/data_products/imaging_weights/")
        if "extended" in fn:
            if max_ebv < 1.:
                weights_path = weights_dir / 'extended_lrg_linear_coeffs_pz.yaml'
            else:
                weights_path = weights_dir / 'extended_lrg_linear_coeffs_pz_no_ebv.yaml'
        else:
            if max_ebv < 1.:
                weights_path = weights_dir / 'main_lrg_linear_coeffs_pz.yaml'
            else:
                weights_path = weights_dir / 'main_lrg_linear_coeffs_pz_no_ebv.yaml'

        # add the depth
        if not want_randoms:
            more_dir = Path("/global/cfs/cdirs/desi/users/rongpu/lrg_xcorr/data_products/catalogs/more/")
            more_2 = Table(fitsio.read(more_dir / (fn.split('_pzbins')[0]+'_more_2.fits'),
                                       columns=['GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']))
            cat = hstack([cat, more_2], join_type='exact')
            del more_2; gc.collect()

        # Convert depths to units of magnitude
        cat['galdepth_gmag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['GALDEPTH_G'])))-9) - 3.214*cat['EBV']
        cat['galdepth_rmag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['GALDEPTH_R'])))-9) - 2.165*cat['EBV']
        cat['galdepth_zmag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['GALDEPTH_Z'])))-9) - 1.211*cat['EBV']

        # initializes the weights
        if want_randoms:
            for bin_index in range(1, 5):  # 4 bins
                cat[f'weight_bin_{bin_index:d}'] = 0.
        else:
            cat['weight'] = 0.

        # NGC and SGC
        for field in ['north', 'south']:
            if field == 'south':
                photsys = 'S'
                sign = -1.
            elif field == 'north':
                photsys = 'N'
                sign = 1.
                
            # Load weights
            with open(weights_path, "r") as f:
                linear_coeffs = yaml.safe_load(f)

            for bin_index in range(1, 5):  # 4 bins

                # select this GC
                mask_bin = cat['PHOTSYS'] == photsys

                # select this bin
                if not want_randoms:
                    mask_bin &= cat['pz_bin'] == bin_index 
                cat1 = cat[mask_bin].copy()

                # read in all the coefficients
                xnames_fit = list(linear_coeffs['south_bin_1'].keys())
                xnames_fit.remove('intercept')

                # Assign zero weights to objects with invalid imaging properties
                # (their fraction should be negligibly small)
                mask_bad = np.full(len(cat1), False)
                for col in xnames_fit:
                    mask_bad |= ~np.isfinite(cat1[col])
                if np.sum(mask_bad) != 0:
                    print(f'{np.sum(mask_bad):d} invalid objects')

                # initialize weights
                weights = np.zeros(len(cat1))
                bin_str = f'{field}_bin_{bin_index:d}'

                # create array of coefficients, with the first coefficient being the intercept
                coeffs = np.array([linear_coeffs[bin_str]['intercept']]+[linear_coeffs[bin_str][xname] for xname in xnames_fit])
                data = np.column_stack([cat1[~mask_bad][xname] for xname in xnames_fit])

                # create 2-D array of imaging properties, with the first columns being unity
                data1 = np.insert(data, 0, 1., axis=1)

                # wt = coeff0 + coeff1 * rand['EBV'] + coeff2 * rand['PSFSIZE_G'] + ...
                weights[~mask_bad] = 1./np.dot(coeffs, data1.T)  # 1/predicted_density as weights for objects

                # save the weights
                if want_randoms:
                    cat[f'weight_bin_{bin_index:d}'][mask_bin] = sign*weights
                else:
                    cat['weight'][mask_bin] = sign*weights

    # apply all selections
    cat = cat[mask]
                
    # save final file tuks when no weights
    if want_randoms:
        if want_weights:
            np.savez(save_dir / save_fn, RA=cat['RA'], DEC=cat['DEC'], weight_bin_1=cat['weight_bin_1'], weight_bin_2=cat['weight_bin_2'], weight_bin_3=cat['weight_bin_3'], weight_bin_4=cat['weight_bin_4'])
        else:
            np.savez(save_dir / save_fn, RA=cat['RA'], DEC=cat['DEC'])
    else:
        if want_weights:
            np.savez(save_dir / save_fn, RA=cat['RA'], DEC=cat['DEC'], Z_PHOT_MEDIAN=cat['Z_PHOT_MEDIAN'], weight=cat['weight'], pz_bin=cat['pz_bin'])
        else:
            np.savez(save_dir / save_fn, RA=cat['RA'], DEC=cat['DEC'], Z_PHOT_MEDIAN=cat['Z_PHOT_MEDIAN'], pz_bin=cat['pz_bin'])

#>>> cat_rand.keys()
#['RELEASE', 'BRICKID', 'BRICKNAME', 'BRICK_OBJID', 'RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'APFLUX_G', 'APFLUX_R', 'APFLUX_Z', 'APFLUX_IVAR_G', 'APFLUX_IVAR_R', 'APFLUX_IVAR_Z', 'MASKBITS', 'WISEMASK_W1', 'WISEMASK_W2', 'EBV', 'PHOTSYS', 'HPXPIXEL', 'TARGETID', 'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'SUBPRIORITY', 'OBSCONDITIONS', 'PRIORITY_INIT', 'NUMOBS_INIT', 'SCND_TARGET', 'NUMOBS_MORE', 'NUMOBS', 'Z', 'ZWARN', 'TARGET_STATE', 'TIMESTAMP', 'VERSION', 'PRIORITY']

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--fn', help='File name', type=str, default=DEFAULTS['fn'])
    parser.add_argument('--min_nobs', help='Minimum number of observations per GRZ', type=int, default=DEFAULTS['min_nobs'])
    parser.add_argument('--max_stardens', help='Maximum stellar density', type=int, default=DEFAULTS['max_stardens'])
    parser.add_argument('--max_ebv', help='Maximum EBV value', type=float, default=DEFAULTS['max_ebv'])
    parser.add_argument('--apply_lrg_mask', help='Want to apply LRG mask', action='store_true')
    parser.add_argument('--remove_islands', help='Want to remove islands', action='store_true')
    parser.add_argument('--want_weights', help='Want to apply weights', action='store_true')
    args = vars(parser.parse_args())
    main(**args)
