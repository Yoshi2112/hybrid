# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:56:09 2019

@author: Yoshi
"""
import numpy as np
import matplotlib.pyplot as plt
import convective_growth_rate as conv

def init_param_arrays():
    global N, ndensc, ndensw, temperp, A, betapar
    N       = 3    
    ndensc  = np.zeros(N)
    ndensw  = np.zeros(N)
    temperp = np.zeros(N)
    betapar = np.zeros(N)
    A       = np.zeros(N)
    return

def geomagnetic_magnitude(L_shell, lat=0.):
    '''Returns the magnetic field magnitude (intensity) on the specified L shell at the given MLAT, in nanoTesla.
    
    INPUT:
        L_shell : McIlwain L-parameter defining distance of disired field line at equator, in RE
        lat     : Geomagnetic latitude (MLAT) in degrees. Default value 0.
        
    OUPUT:
        B_tot   : Magnetic field magnitude, in nT
    '''
    B_surf     = 3.12e-5
    r_loc      = L_shell * np.cos(np.pi * lat / 180.) ** 2
    B_tot      = B_surf / (r_loc ** 3) * np.sqrt(1. + 3.*np.sin(np.pi * lat / 180.) ** 2)
    return B_tot * 1e9

def plot_growth_rate(F, GR, ST, ax):
    ax.plot(F, GR)
    for ii in range(ST.shape[0] - 1):
        if ST[ii] == 1:
            ax.axvspan(F[ii], F[ii + 1], color='k')            # PLOT STOP BAND
    return

def set_figure_text(ax):
    font = 'monospace'
    top  = 0.97
    left = 0.70
    
    TPER_kev = temperp / 1e3
    ax.text(left + 0.03, top, 'Cold($cm^{-3}$)' , transform=ax.transAxes, fontsize=10, fontname=font)
    ax.text(left + 0.10, top, 'Hot($cm^{-3}$)'  , transform=ax.transAxes, fontsize=10, fontname=font)
    ax.text(left + 0.16, top, '$A_i$'           , transform=ax.transAxes, fontsize=10, fontname=font)
    ax.text(left + 0.19, top, '$T_\perp (keV)$' , transform=ax.transAxes, fontsize=10, fontname=font)
    
    ax.text(left + 0*0.045, top - 0.02, ' H+:'                      , transform=ax.transAxes, fontsize=10, fontname=font)
    ax.text(left + 1*0.045, top - 0.02, '{:>7.3f}'.format(ndensc[0]), transform=ax.transAxes, fontsize=10, fontname=font)
    ax.text(left + 2*0.045, top - 0.02, '{:>7.3f}'.format(ndensw[0]), transform=ax.transAxes, fontsize=10, fontname=font)
    ax.text(left + 3*0.045, top - 0.02, '{:>7.2f}'.format(A[0]),      transform=ax.transAxes, fontsize=10, fontname=font)
    ax.text(left + 4*0.045, top - 0.02, '{:>7.0f}'.format(TPER_kev[0]),transform=ax.transAxes, fontsize=10, fontname=font)
    
    ax.text(left + 0*0.045, top - 0.04, 'He+:'                      , transform=ax.transAxes, fontsize=10, fontname=font)
    ax.text(left + 1*0.045, top - 0.04, '{:>7.3f}'.format(ndensc[1]), transform=ax.transAxes, fontsize=10, fontname=font)
    ax.text(left + 2*0.045, top - 0.04, '{:>7.3f}'.format(ndensw[1]), transform=ax.transAxes, fontsize=10, fontname=font)
    ax.text(left + 3*0.045, top - 0.04, '{:>7.2f}'.format(A[1]),      transform=ax.transAxes, fontsize=10, fontname=font)
    ax.text(left + 4*0.045, top - 0.04, '{:>7.0f}'.format(TPER_kev[1]),transform=ax.transAxes, fontsize=10, fontname=font)
    
    ax.text(left + 0*0.045, top - 0.06, ' O+:'                      , transform=ax.transAxes, fontsize=10, fontname=font)
    ax.text(left + 1*0.045, top - 0.06, '{:>7.3f}'.format(ndensc[2]), transform=ax.transAxes, fontsize=10, fontname=font)
    ax.text(left + 2*0.045, top - 0.06, '{:>7.3f}'.format(ndensw[2]), transform=ax.transAxes, fontsize=10, fontname=font)
    ax.text(left + 3*0.045, top - 0.06, '{:>7.2f}'.format(A[2]),      transform=ax.transAxes, fontsize=10, fontname=font)
    ax.text(left + 4*0.045, top - 0.06, '{:>7.0f}'.format(TPER_kev[2]),transform=ax.transAxes, fontsize=10, fontname=font)
    return




if __name__ == '__main__':
    init_param_arrays()
    plt.ioff()
    fig = plt.figure(figsize=(18, 10))
    ax  = fig.add_subplot(111)
    
    #for A_var in [0.75, 1.0, 1.5]:
    field      = 487.5           # Magnetic field flux in nT
    
    ndensc[0]  = 10.            # Density of cold species (number/cc) FRASER CHECK VALUES
    ndensc[1]  = 0.
    ndensc[2]  = 0.
    
    ndensw[0]  = 5.0             # Density of warm species (number/cc)
    ndensw[1]  = 5.00
    ndensw[2]  = 5.0
    
    temperp[0] = 50000.          # Perpendicular temperature (ev)
    temperp[1] = 10000.
    temperp[2] = 10000.
    
    betapar[0] = 0.34223326
    betapar[1] = 0.00111841
    betapar[2] = 0.00290786
    
    A[0] = 1.                    # Temperature anisotropy
    A[1] = 1.
    A[2] = 1.

    FREQ, GR_RATE, STFLAG = conv.calculate_growth_rate(field, ndensc, ndensw, A, temperp=temperp, norm_freq=1)
    plot_growth_rate(FREQ, GR_RATE, STFLAG, ax)
    
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Growth Rate ($\omega / V_g 10^{-7} cm^{-1}$)', rotation=90, fontsize=14)
    set_figure_text(ax)
    ax.set_xlim(0, 0.5)
    plt.show()
        