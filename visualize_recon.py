from pathlib import Path
import os
import gc

import numpy as np
import argparse
from astropy.table import Table
import fitsio
import healpy as hp
    
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
python visualize_recon.py --gal_fn dr9_lrg_pzbins.fits --rand_fn main_randoms-1-0.fits --apply_lrg_mask --remove_islands --nmesh 1024 --sr 12.5 --rectype MG --convention recsym --pz_bin 1

python visualize_recon.py --gal_fn dr9_lrg_pzbins.fits --rand_fn main_randoms-1-0.fits --apply_lrg_mask --remove_islands --nmesh 2048 --sr 12.5 --rectype MG --convention recsym --pz_bin 1

python visualize_recon.py --gal_fn dr9_lrg_pzbins.fits --rand_fn main_randoms-1-0-3.fits --apply_lrg_mask --remove_islands --nmesh 2048 --sr 12.5 --rectype MG --convention recsym
"""

def calc_velocity(Psi, unit_los, a, f, H, h, want_rsd=False):
    """
    Internal function for calculating the ZA velocity given the displacement.
    """
    # construct the perpendicular unit_los vectors
    unit_perp1 = np.vstack((unit_los[:, 2], unit_los[:, 2], -unit_los[:, 1] -unit_los[:, 0])).T
    unit_perp2 = np.vstack((unit_los[:, 1]*unit_perp1[:, 2] - unit_los[:, 2]*unit_perp1[:, 1],
                            -(unit_los[:, 0]*unit_perp1[:, 2] - unit_los[:, 2]*unit_perp1[:, 0]),
                            unit_los[:, 0]*unit_perp1[:, 1] - unit_los[:, 1]*unit_perp1[:, 0])).T
    unit_perp1 /= np.linalg.norm(unit_perp1, axis=1)[:, None]
    unit_perp2 /= np.linalg.norm(unit_perp2, axis=1)[:, None]

    # calculate the velocity
    Velocity = Psi*a*f*H # km/s/h
    Velocity /= h # km/s
    Velocity_r = np.sum(Velocity*unit_los, axis=1)    
    Velocity_p1 = np.sum(Velocity*unit_perp1, axis=1)
    Velocity_p2 = np.sum(Velocity*unit_perp2, axis=1)
    if want_rsd:
        Velocity_r /= (1+f) # real space

    Velocity = np.vstack((Velocity_p1, Velocity_p2, Velocity_r)).T
    return Velocity

def main(gal_fn, rand_fn, min_nobs, max_ebv, max_stardens, nmesh, sr, rectype, convention, remove_islands=False, apply_lrg_mask=False, pz_bin=False):

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
        
    # mask strings
    isle_str = "_remov_isle" if remove_islands else ""
    lrg_mask_str = "_lrg_mask" if apply_lrg_mask else ""
    if pz_bin is not None:
        pz_str = f"_bin_{pz_bin:d}"
    else:
        pz_str = ""

    # selection settings
    mask_str = f"{isle_str}_nobs{min_nobs:d}_ebv{max_ebv:.2f}_stardens{max_stardens:d}{lrg_mask_str}"
    gal_mask_fn = f"{gal_fn.split('.fits')[0]}{mask_str}.npz"
    extra_str = "_z"

    # directory where the reconstructed mock catalogs are saved
    recon_dir = Path("/global/cfs/cdirs/desi/users/boryanah/reconstruction_DESI/recon/")
    gal_dir = Path("/global/cfs/cdirs/desi/users/boryanah/reconstruction_DESI/galaxies/")
    
    # file to save to
    save_fn = f"displacements_{gal_fn.split('.fits')[0]}_{rand_fn.split('.fits')[0]}{mask_str}_R{sr:.2f}_nmesh{nmesh:d}_{convention}_{rectype}{pz_str}{foot_str}.npz"

    # load galaxies
    data_gal = np.load(gal_dir / gal_mask_fn)
    RA = data_gal['RA']
    DEC = data_gal['DEC']
    Z = data_gal['Z_PHOT_MEDIAN']
    if want_footprint == "N" or want_footprint == "S":
        W = data_gal['weight']
    
    # select only galaxies in this photo-z bin
    if pz_bin is not None:
        choice = data_gal['pz_bin'] == pz_bin
        RA = RA[choice]
        DEC = DEC[choice]
        Z = Z[choice]
        if want_footprint == "N" or want_footprint == "S":
            W = W[choice]
        del choice
    
    # select objects in the footprint
    if want_footprint == "DES":
        cat_hp_idx = hp.pixelfunc.ang2pix(nside, RA, DEC, lonlat=True, nest=False)
        mask = np.in1d(cat_hp_idx, des_hp_idx)
        RA, DEC, Z = RA[mask], DEC[mask], Z[mask]
    elif want_footprint == "N":
        mask = W > 0.
        RA, DEC, Z = RA[mask], DEC[mask], Z[mask]
    elif want_footprint == "S":
        mask = W < 0.
        cat_hp_idx = hp.pixelfunc.ang2pix(nside, RA, DEC, lonlat=True, nest=False)
        mask &= ~np.in1d(cat_hp_idx, des_hp_idx)
        RA, DEC, Z = RA[mask], DEC[mask], Z[mask]
    del data_gal; gc.collect()
        
    # transform into Cartesian coordinates
    cosmo = DESI()
    Position = utils.sky_to_cartesian(cosmo.comoving_radial_distance(Z), RA, DEC)
    del RA, DEC, Z
    gc.collect()
    print("Position", Position[:10], Position[:, 0].min(), Position[:, 0].max(), Position[:, 1].min(), Position[:, 1].max(), Position[:, 2].min(), Position[:, 2].max())

    # get unit vector
    unit_los = Position/np.linalg.norm(Position, axis=1)[:, None]
    print("unit_los", unit_los[:10])

    # load displacements
    data_recon = np.load(recon_dir / save_fn)
    mean_z = data_recon['mean_z']
    growth_factor = data_recon['growth_factor']
    Hubble_z = data_recon['Hubble_z']
    displacements = data_recon['displacements']

    # calculate reconstructed velocities
    h = cosmo.hubble_function(0.)/100.
    Velocity = calc_velocity(displacements, unit_los, 1./(1.+mean_z), growth_factor, Hubble_z, h, want_rsd=False)
    Velocity_para = Velocity[:, 2]
    print("Velocity_para", Velocity_para.min(), Velocity_para.max(), np.mean(Velocity_para[Velocity_para > 0.]), np.mean(Velocity_para[Velocity_para < 0.]), np.sum(Velocity_para > 0.)/len(Velocity_para))
    
    # save into a csv format
    #data = np.vstack((Position.T, Velocity_perp, Velocity_para)).T
    data = np.vstack((Position.T, Velocity_para)).T
    data = data.astype(np.float32)
    std = np.std(Velocity_para)
    #std = np.std(Velocity_perp)
    max_std = 3.*std
    data[Velocity_para > max_std, 3] = max_std
    data[Velocity_para < -max_std, 3] = -max_std
    print("min max vel after", np.max(data[:, 3]), np.min(data[:, 3]))
    print("before downsampling", data.shape)
    if pz_bin is not None:
        data = data[::40]
    else:
        data = data[::400]
    print("after downsampling", data.shape)
    np.savetxt(f'data/reconstructed_velocity{pz_str}{foot_str}.csv', data, delimiter=',')
 
    
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--gal_fn', help='Galaxy file name', type=str, default=DEFAULTS['gal_fn'])
    parser.add_argument('--rand_fn', help='Randoms file name', type=str, default=DEFAULTS['rand_fn'])
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
