# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:39:35 2021

@author: Yoshi
"""
import os, warnings, pdb
import numpy as np
import numba as nb
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import spacepy as sp

kB = 1.381e-23
qp = 1.602e-19
mp = 1.673e-27
c  = 3e8

def boundary_idx64(time_arr, start, end):
    '''Returns index values corresponding to the locations of the start/end times in a numpy time array, if specified times are np.datetime64'''
    idx1 = np.where(abs(time_arr - start) == np.min(abs(time_arr - start)))[0][0] 
    idx2 = np.where(abs(time_arr - end)   == np.min(abs(time_arr - end)))[0][0]
    return (idx1, idx2) 


@nb.njit()
def calc_distro_2D(DENS, VPER, VPAR, VTH_PER, VTH_PAR):
    DIST  = np.zeros((VPER.shape[0], VPAR.shape[0]))
    
    out = DENS / np.sqrt(8 * np.pi**3 * VTH_PER**4 * VTH_PAR**2)
    for aa in nb.prange(VPER.shape[0]):
        for bb in nb.prange(VPAR.shape[0]):
            exp = np.exp(-0.5*VPER[aa]**2/VTH_PER**2
                         -0.5*VPAR[bb]**2/VTH_PAR**2)
            DIST[aa, bb] = out*exp
    return DIST


@nb.njit()
def integrate_distro_2D(DIST, VPERP, VPARA, MASS, Emin, Emax):
    dvperp = VPERP[1] - VPERP[0]
    dvpara = VPARA[1] - VPARA[0]
    
    SUM = 0.0
    for aa in nb.prange(DIST.shape[0]):
        for bb in nb.prange(DIST.shape[1]):
            VEL2 = VPERP[aa]**2 + VPARA[bb]**2
            ENRG = MASS*VEL2/(2*qp)
            
            if VPERP[aa] >= 0.0 and ENRG >= Emin and ENRG <= Emax:
                SUM += 2*np.pi*VPERP[aa]*DIST[aa, bb]*dvpara*dvperp
    return SUM


if __name__ == '__main__':
    # Create Kappa distribution with sample values
    # See how they look in velocity and energy space
    # And compare them against the equivalent Maxwellian distribution

    