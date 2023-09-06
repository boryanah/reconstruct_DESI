from pathlib import Path
import os
import gc

import numpy as np
import argparse
    
from pyrecon import  utils, IterativeFFTParticleReconstruction, MultiGridReconstruction, IterativeFFTReconstruction
from cosmoprimo.fiducial import Planck2018FullFlatLCDM, AbacusSummit, DESI, TabulatedDESI

DEFAULTS = {}
DEFAULTS['gal_fn'] = "dr9_lrg_pzbins.fits"
DEFAULTS['rand_fn'] = "randoms-1-0.fits"
DEFAULTS['min_nobs'] = 2
DEFAULTS['max_ebv'] = 0.15
DEFAULTS['max_stardens'] = 0
DEFAULTS['nmesh'] = 1024
DEFAULTS['sr'] = 12.5 # Mpc/h
DEFAULTS['rectype'] = "MG"
DEFAULTS['convention'] = "recsym"

"""
Usage:
python reconstruct_data.py --rand_fn main_randoms-1-0.fits --gal_fn dr9_lrg_pzbins.fits --apply_lrg_mask --remove_islands --nmesh 1024 --sr 12.5 --rectype MG --convention recsym --want_gal_weights --pz_bin 1

python reconstruct_data.py --rand_fn main_randoms-1-0-3.fits --gal_fn dr9_lrg_pzbins.fits --apply_lrg_mask --remove_islands --nmesh 2048 --sr 12.5 --rectype MG --convention recsym --want_gal_weights

python reconstruct_data.py --rand_fn main_randoms-1-0.fits --gal_fn dr9_lrg_pzbins.fits --apply_lrg_mask --remove_islands --nmesh 2048 --sr 12.5 --rectype MG --convention recsym --want_rand_weights --pz_bin 1
"""

def main(gal_fn, rand_fn, min_nobs, max_ebv, max_stardens, nmesh, sr, rectype, convention, want_rand_weights=False, want_gal_weights=False, remove_islands=False, apply_lrg_mask=False, pz_bin=False):
    #want_footprint = None
    want_footprint = "S" # negative weights
    #want_footprint = "N" # positive weights
    #want_footprint = "DES"
    foot_str = ""

    if want_footprint == "DES":
        import healpy as hp
        nside = 256
        des_footprint = hp.fitsfunc.read_map(f"/global/cfs/cdirs/desi/users/rongpu/lrg_xcorr/data_products/misc/hp_in_des_{nside}_ring.fits.gz")        
        des_hp_idx = np.arange(len(des_footprint))[des_footprint]
        foot_str = "_des"
        del des_footprint; gc.collect()
    elif want_footprint == "N":
        foot_str = "_north"
    elif want_footprint == "S":
        import healpy as hp
        nside = 256
        des_footprint = hp.fitsfunc.read_map(f"/global/cfs/cdirs/desi/users/rongpu/lrg_xcorr/data_products/misc/hp_in_des_{nside}_ring.fits.gz")        
        des_hp_idx = np.arange(len(des_footprint))[des_footprint]
        del des_footprint; gc.collect()
        foot_str = "_south"
        
    # make sure weights either applied to the gals or the rands
    assert (want_rand_weights and want_gal_weights) == False, "Can't apply weights to both galaxies and randoms"
    
    # mask strings
    isle_str = "_remov_isle" if remove_islands else ""
    lrg_mask_str = "_lrg_mask" if apply_lrg_mask else ""

    # selection settings
    mask_str = f"{isle_str}_nobs{min_nobs:d}_ebv{max_ebv:.2f}_stardens{max_stardens:d}{lrg_mask_str}"
    gal_mask_fn = f"{gal_fn.split('.fits')[0]}{mask_str}.npz"
    rand_mask_fn = f"{rand_fn.split('.fits')[0]}{mask_str}.npz"
    extra_str = "_z"
    if pz_bin is not None:
        pz_str = f"_bin_{pz_bin:d}"
    else:
        pz_str = ""
    extra_str += pz_str
    rand_mask_z_fn = f"{rand_mask_fn.split('.npz')[0]}{extra_str}.npz"

    #for rand_mask_fn in rand_mask_fns
    #rand_fn = main_randoms-1-0.fits
    splits = rand_fn.split('-')
    hyphens = splits[1:]
    assert len(hyphens) >= 2 and len(hyphens) <= 3
    if len(hyphens) == 2: # single file
        rand_mask_fns = [f"{rand_fn.split('.fits')[0]}{mask_str}.npz"]
        rand_mask_z_fns = [f"{rand_mask_fn.split('.npz')[0]}{extra_str}.npz"]
    else:
        rand_mask_fns = []
        rand_mask_z_fns = []
        start = int(hyphens[1])
        final = int(hyphens[2].split('.fits')[0])
        for i in range(start, final+1):
            rand_mask_fn = f"{splits[0]}-{hyphens[0]}-{i:d}{mask_str}.npz"
            rand_mask_z_fn = f"{rand_mask_fn.split('.npz')[0]}{extra_str}.npz"
            rand_mask_fns.append(rand_mask_fn)
            rand_mask_z_fns.append(rand_mask_z_fn)

    # select mean redshift
    if "lrg" in gal_fn:
        z_means = np.array([0.47499338, 0.6342638, 0.7941149, 0.9208743], dtype=np.float32)
        mean_z = 0.72 #0.5
        if pz_bin is not None:
            mean_z = z_means[pz_bin-1]
        bias = 2.2

    # directory where the reconstructed mock catalogs are saved
    recon_dir = Path("/global/cfs/cdirs/desi/users/boryanah/reconstruction_DESI/recon/")
    rand_dir = Path("/global/cfs/cdirs/desi/users/boryanah/reconstruction_DESI/randoms/")
    gal_dir = Path("/global/cfs/cdirs/desi/users/boryanah/reconstruction_DESI/galaxies/")
    os.makedirs(recon_dir, exist_ok=True)
    
    # file to save to
    save_fn = f"displacements_{gal_fn.split('.fits')[0]}_{rand_fn.split('.fits')[0]}{mask_str}_R{sr:.2f}_nmesh{nmesh:d}_{convention}_{rectype}{pz_str}{foot_str}.npz"
    
    # 32, 128 physical cpu per node for cori, perlmutter (hyperthreading doubles)
    ncpu = 256
    
    # reconstruction parameters
    if rectype == "IFT":
        recfunc = IterativeFFTReconstruction
    elif rectype == "IFTP":
        recfunc = IterativeFFTParticleReconstruction
    elif rectype == "MG":
        recfunc = MultiGridReconstruction
    
    # simulation parameters
    cosmo = DESI() # AbacusSummit
    ff = cosmo.growth_factor(mean_z)
    H_z = cosmo.hubble_function(mean_z)
    los = 'local'

    # load galaxies
    data_gal = np.load(gal_dir / gal_mask_fn)
    RA = data_gal['RA']
    DEC = data_gal['DEC']
    Z = data_gal['Z_PHOT_MEDIAN']
    if want_gal_weights:
        W = data_gal['weight']        
    else:
        W = None

    # select only galaxies in this photo-z bin
    if pz_bin is not None:
        choice = data_gal['pz_bin'] == pz_bin
        RA = RA[choice]
        DEC = DEC[choice]
        Z = Z[choice]
        if want_gal_weights:
            W = W[choice]
        del choice
    del data_gal; gc.collect()

    # select objects in the footprint
    if want_footprint == "DES":
        cat_hp_idx = hp.pixelfunc.ang2pix(nside, RA, DEC, lonlat=True, nest=False)
        mask = np.in1d(cat_hp_idx, des_hp_idx)
        RA, DEC, Z = RA[mask], DEC[mask], Z[mask]
        if want_gal_weights:
            W = W[mask]
    elif want_footprint == "N":
        mask = W > 0.
        RA, DEC, Z = RA[mask], DEC[mask], Z[mask]
        if want_gal_weights:
            W = W[mask]
    elif want_footprint == "S":
        mask = W < 0.
        cat_hp_idx = hp.pixelfunc.ang2pix(nside, RA, DEC, lonlat=True, nest=False)
        mask &= ~np.in1d(cat_hp_idx, des_hp_idx)
        RA, DEC, Z = RA[mask], DEC[mask], Z[mask]
        if want_gal_weights:
            W = W[mask]

    if want_gal_weights:
        W = np.abs(W) # sign tells you if SGC (-) or NGC (+)
            
    # transform into Cartesian coordinates
    Position = utils.sky_to_cartesian(cosmo.comoving_radial_distance(Z), RA, DEC)
    print("number of galaxies", Position.shape[0])
    if want_footprint is not None:
        del RA, DEC
    else:
        del RA, DEC, Z
    gc.collect()

    RandomPositionFinal = np.empty((0, 3))
    RAND_W_FINAL = np.empty(0)
    for i in range(len(rand_mask_fns)):
        # load randoms
        rand_mask_fn = rand_mask_fns[i]
        rand_mask_z_fn = rand_mask_z_fns[i]        
        data_rand = np.load(rand_dir / rand_mask_fn)
        data_rand_z = np.load(rand_dir / rand_mask_z_fn)            
        RAND_RA = data_rand['RA']
        RAND_DEC = data_rand['DEC']
        RAND_Z = data_rand_z['Z']
        if want_rand_weights or want_footprint == "N" or want_footprint == "S":
            if pz_bin is None:
                RAND_W = data_rand_z['weights']
            else:
                RAND_W = data_rand[f'weight_bin_{pz_bin:d}']
            mask = np.isclose(RAND_W, 0.)
            RAND_W[~mask] = 1./RAND_W[~mask]
            del mask
            gc.collect()
        del data_rand, data_rand_z; gc.collect()

        # select objects in the footprint
        if want_footprint == "DES":
            cat_hp_idx = hp.pixelfunc.ang2pix(nside, RAND_RA, RAND_DEC, lonlat=True, nest=False)
            mask = np.in1d(cat_hp_idx, des_hp_idx)
            print("percentage", np.sum(mask)/len(mask))
            RAND_RA, RAND_DEC = RAND_RA[mask], RAND_DEC[mask]
            RAND_Z = np.random.choice(Z, size=len(RAND_RA), replace=True)
            del Z; gc.collect()
            if want_rand_weights:
                RAND_W = RAND_W[mask]
            del mask; gc.collect()
        elif want_footprint == "N":
            mask = RAND_W > 0.
            print("percentage", np.sum(mask)/len(mask))
            RAND_RA, RAND_DEC = RAND_RA[mask], RAND_DEC[mask]
            RAND_Z = np.random.choice(Z, size=len(RAND_RA), replace=True)
            del Z; gc.collect()
            if want_rand_weights:
                RAND_W = RAND_W[mask]
            del mask; gc.collect()
        elif want_footprint == "S":
            mask = RAND_W < 0.
            cat_hp_idx = hp.pixelfunc.ang2pix(nside, RAND_RA, RAND_DEC, lonlat=True, nest=False)
            mask &= ~np.in1d(cat_hp_idx, des_hp_idx)
            print("percentage", np.sum(mask)/len(mask))
            RAND_RA, RAND_DEC = RAND_RA[mask], RAND_DEC[mask]
            RAND_Z = np.random.choice(Z, size=len(RAND_RA), replace=True)
            del Z; gc.collect()
            if want_rand_weights:
                RAND_W = RAND_W[mask]
            del mask; gc.collect()
        
            
        if want_rand_weights:
            RAND_W = np.abs(RAND_W) # sign tells you if SGC (-) or NGC (+)
        
        # transform into cartesian coordinates
        RandomPosition = utils.sky_to_cartesian(cosmo.comoving_radial_distance(RAND_Z), RAND_RA, RAND_DEC)
        del RAND_RA, RAND_DEC, RAND_Z
        gc.collect()

        # coadd positions and weights, if requested
        RandomPositionFinal = np.vstack((RandomPositionFinal, RandomPosition))
        del RandomPosition
        gc.collect()
        if want_rand_weights:
            RAND_W_FINAL = np.hstack((RAND_W_FINAL, RAND_W))
            del RAND_W
            gc.collect()
        else:
            RAND_W_FINAL = None
        print("number of randoms loaded so far", RandomPositionFinal.shape[0])
        
    # run reconstruction
    print('Recon First tracer')
    recon_tracer = recfunc(f=ff, bias=bias, nmesh=nmesh, los=los, positions=Position, # used only to define box size if not provided
                           nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=False)
    print('grid set up', flush=True)
    recon_tracer.assign_data(Position, weights=W)
    print('data assigned', flush=True)
    recon_tracer.assign_randoms(RandomPositionFinal, weights=RAND_W_FINAL)
    print('randoms assigned', flush=True)
    recon_tracer.set_density_contrast(smoothing_radius=sr)
    print('density constrast calculated, now doing recon', flush=True)
    recon_tracer.run()
    print('recon has been run', flush=True)

    # read the displacements in real and redshift space (rsd cause real data has RSD obviously)
    if rectype == "IFTP":
        #displacements_rsd = recon_tracer.read_shifts('data', field='disp+rsd')
        displacements_rsd_nof = recon_tracer.read_shifts('data', field='disp')
    else:
        #displacements_rsd = recon_tracer.read_shifts(Position, field='disp+rsd')
        displacements_rsd_nof = recon_tracer.read_shifts(Position, field='disp')

    #random_displacements_rsd = recon_tracer.read_shifts(RandomPosition, field='disp+rsd')
    #random_displacements_rsd_nof = recon_tracer.read_shifts(RandomPosition, field='disp')
    
    # save the displacements
    np.savez(recon_dir / save_fn, mean_z=mean_z, growth_factor=ff, Hubble_z=H_z, displacements=displacements_rsd_nof)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--gal_fn', help='Galaxy file name', type=str, default=DEFAULTS['gal_fn'])
    parser.add_argument('--rand_fn', help='Randoms file name', type=str, default=DEFAULTS['rand_fn'])
    parser.add_argument('--want_rand_weights', help='Want to apply weights to the randoms', action='store_true')
    parser.add_argument('--want_gal_weights', help='Want to apply weights to the galaxies', action='store_true')
    parser.add_argument('--pz_bin', help='Select a single photo-z bin', type=int, default=None)
    parser.add_argument('--min_nobs', help='Minimum number of observations per GRZ', type=int, default=DEFAULTS['min_nobs'])
    parser.add_argument('--max_stardens', help='Maximum stellar density', type=int, default=DEFAULTS['max_stardens'])
    parser.add_argument('--max_ebv', help='Maximum EBV value', type=float, default=DEFAULTS['max_ebv'])
    parser.add_argument('--apply_lrg_mask', help='Want to apply LRG mask', action='store_true')
    parser.add_argument('--remove_islands', help='Want to remove islands', action='store_true')
    parser.add_argument('--nmesh', help='Number of cells per dimension for reconstruction', type=int, default=DEFAULTS['nmesh'])
    parser.add_argument('--sr', help='Smoothing radius', type=float, default=DEFAULTS['sr'])
    parser.add_argument('--rectype', help='Reconstruction type', default=DEFAULTS['rectype'], choices=["IFT", "MG", "IFTP"])
    parser.add_argument('--convention', help='Reconstruction convention', default=DEFAULTS['convention'], choices=["recsym", "reciso"])
    args = vars(parser.parse_args())
    main(**args)
