# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:17:35 2020

@author: Yoshi
"""
import numpy as np
import os, sys
sys.path.append('..//new_general_DR_solver//')

from growth_rates_from_RBSP         import extract_species_arrays
from dispersion_solver_multispecies import create_species_array
import omura_play as op


## For resonance velocity :: Do the advanced solver to calculate (w - Omega) / k
## Since I have both w and k (can also get phase velocity)
## Group velocity could come from a finite difference

## Limitations :: Omura scripts don't do hot heavy ion species, only hot
##                protons, which are sent separate from the Species array

## Note: Really ought to low-pass the background field and re-run the linear
##       theory analysis. It's hard to tell what was changed by the ULF field
##       and what was changed by the EMIC wave. At the very least, it should 
##       be LP'd to prevent aliasing in what is effectively a decimation.

if __name__ == '__main__':
    qp     = 1.602e-19
    qe     =-1.602e-19
    mp     = 1.673e-27
    me     = 9.110e-31
    e0     = 8.854e-12
    mu0    = 4e-7*np.pi
    RE     = 6.371e6
    c      = 3e8
    kB     = 1.380649e-23
    B_surf = 3.12e-5
     
    rbsp_path  = 'G://DATA//RBSP//'
    time_start = np.datetime64('2013-07-25T21:20:00')
    time_end   = np.datetime64('2013-07-25T22:00:00')
    probe      = 'a'
    pad        = 0
    
    ccomp = [70, 20, 10]
    
    times, B0, name, mass, charge, density, tper, ani, cold_dens = \
        extract_species_arrays(rbsp_path, time_start, time_end, probe, pad, ccomp, return_raw_ne=True)

    for ii in range(times.shape[0]):
        pass
# =============================================================================
#     Th_para  = (mp * (6e5)**2 / kB) / 11603.
#     Th_perp  = (mp * (8e5)**2 / kB) / 11603.
#     Ah       = Th_perp / Th_para - 1
#     apar_h   = np.sqrt(2.0 * qp * Th_para  / mp)
#     
#     # Parameters in SI units (Note: Added the hot bit here. Is it going to break anything?) nh = 7.2
#     _name    = np.array(['H'    , 'He'  , 'O'  , 'Hot H' ])
#     _mass    = np.array([1.0    , 4.0   , 16.0 , 1.0     ]) * mp
#     _charge  = np.array([1.0    , 1.0   , 1.0  , 1.0     ]) * qp
#     _density = np.array([144.0  , 17.0  , 17.0 , 7.2     ]) * 1e6
#     _tpar    = np.array([0.0    , 0.0   , 0.0  ,  Th_para])
#     _ani     = np.array([0.0    , 0.0   , 0.0  ,  Ah     ])
#     _tper    = (_ani + 1) * _tpar
#     
#     #_Species, _PP = create_species_array(_B0, _name, _mass, _charge, _density, _tper, _ani)
# =============================================================================
