# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:56:09 2019

@author: Yoshi
"""
import sys
data_scripts_dir = 'F://Google Drive//Uni//PhD 2017//Data//Scripts'
sys.path.append(data_scripts_dir)

import numpy                     as np
import matplotlib.pyplot         as plt
import matplotlib.gridspec       as gs
import convective_growth_rate    as conv

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


def set_figure_text(ax1):
    font    = 'monospace'
    top     = 1.07
    left    = 0.78
    h_space = 0.04
    
    ax1.text(0.00, top - 0.02, '$B_0 = $variable',         transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(0.00, top - 0.05, '$n_0 = $%.1f$cm^{-3}$' % n0, transform=ax1.transAxes, fontsize=10, fontname=font)
    
    ax1.text(left + 0.06, top,  'Cold'             , transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 0.099, top,  'Warm'             , transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 0.143, top,  '$A_i$'               , transform=ax1.transAxes, fontsize=10, fontname=font)
    
    if beta_flag == 1:
        ax1.text(left + 0.192, top, r'$\beta_{\parallel}$' , transform=ax1.transAxes, fontsize=10, fontname=font)
        ax1.text(left + 4.2*h_space, top - 0.02, '{:>7.2f}'.format(betapar[0]),transform=ax1.transAxes, fontsize=10, fontname=font)
        ax1.text(left + 4.2*h_space, top - 0.04, '{:>7.2f}'.format(betapar[1]),transform=ax1.transAxes, fontsize=10, fontname=font)
        ax1.text(left + 4.2*h_space, top - 0.06, '{:>7.2f}'.format(betapar[2]),transform=ax1.transAxes, fontsize=10, fontname=font)
    else:
        TPER_kev = temperp / 1e3
        ax1.text(left + 0.188, top, r'$T_{\perp} (keV)$' , transform=ax1.transAxes, fontsize=10, fontname=font)
        ax1.text(left + 4.2*h_space, top - 0.02, '{:>7.2f}'.format(TPER_kev[0]),transform=ax1.transAxes, fontsize=10, fontname=font)
        ax1.text(left + 4.2*h_space, top - 0.04, '{:>7.2f}'.format(TPER_kev[1]),transform=ax1.transAxes, fontsize=10, fontname=font)
        ax1.text(left + 4.2*h_space, top - 0.06, '{:>7.2f}'.format(TPER_kev[2]),transform=ax1.transAxes, fontsize=10, fontname=font)

    ax1.text(left + 0.5*h_space, top - 0.02, ' H+:'                      , transform=ax1.transAxes, fontsize=10, fontname=font)
    #ax1.text(left + 1*h_space, top - 0.02, '{:>7.2f}'.format(H_frac[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
    #ax1.text(left + 2*h_space, top - 0.02, '{:>7.2f}'.format(H_frac[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 3*h_space, top - 0.02, '{:>7.2f}'.format(A[0]),      transform=ax1.transAxes, fontsize=10, fontname=font)
    
    ax1.text(left + 0.5*h_space, top - 0.04, 'He+:'                      , transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 1*h_space, top - 0.04, '{:>7.2f}'.format(He_frac[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 2*h_space, top - 0.04, '{:>7.2f}'.format(He_frac[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 3*h_space, top - 0.04, '{:>7.2f}'.format(A[1]),      transform=ax1.transAxes, fontsize=10, fontname=font)
    
    ax1.text(left + 0.5*h_space, top - 0.06, ' O+:'                      , transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 1*h_space, top - 0.06, '{:>7.2f}'.format(O_frac[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 2*h_space, top - 0.06, '{:>7.2f}'.format(O_frac[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 3*h_space, top - 0.06, '{:>7.2f}'.format(A[2]),      transform=ax1.transAxes, fontsize=10, fontname=font)
    return


def plot_CGR_space(plot_data_points=False, show_sample_points=False):
    plt.ioff()
    fig    = plt.figure(1, figsize=(16,9))
    grids  = gs.GridSpec(2, 2, width_ratios=[1, 0.02], height_ratios=[0.05, 1])
    ax1    = fig.add_subplot(grids[1, 0])
    
    im1 = ax1.contourf(ni, B0, growth_max, 400, cmap='nipy_spectral')
    
    ax1.set_xlim(ni_min, ni_max)
    ax1.set_xlabel('Relative hot proton fraction, $n_i / n_0$')
    ax1.set_ylim(B_min, B_max)
    ax1.set_ylabel('Background magnetic field, $B_0$ (nT)')
    ax1.set_title(r'Max. $S(\omega)$ for each $B_0$, $n_i$')
    
    cax1 = fig.add_subplot(grids[1, 1])
    fig.colorbar(im1, cax=cax1, extend='both').set_label(r'Max. CGR ($\omega / V_g 10^{-7} cm^{-1}$)', rotation=90, fontsize=11)
    fig.subplots_adjust(wspace=0.02, hspace=0)
    set_figure_text(ax1)

    if plot_data_points == True:
        n_data = np.array([38, 160, 38, 160, 150, 150, 350, 350])
        B_data = np.array([158, 158, 134, 134, 200, 260, 200, 260])
        ax1.scatter(n_data, B_data, label='Event Data Limits', marker='x', c='cyan')
        #plt.legend()
        
    if show_sample_points == True:
        for ii in range(ni.shape[0]):
            for jj in range(B0.shape[0]):
                ax1.scatter(ni[ii], B0[jj], c='k', marker='o', s=1, alpha=0.5)
    plt.show()
    return



if __name__ == '__main__':
    N         = 3   # Number of species
    beta_flag = 1   # Flag to define if beta or temperp used. Helps with plotting
    
    n_min  = 20.   ; n_max  = 420.    ; dn  = 5.
    B_min  = 100.  ; B_max  = 300.    ; db  = 5.
    ni_min = 0.0   ; ni_max = 1.0     ; dni = 0.01
    
    B0           = np.arange(B_min , B_max  + db , db )
    n0           = 180.#np.arange(n_min , n_max  + dn , dn )
    ni           = np.arange(ni_min, ni_max + dni, dni)
    
    growth_max   = np.zeros((B0.shape[0], ni.shape[0]))
    
    #H_frac  = [0.9, 0.1]
    He_frac = [0.0, 0.0]
    O_frac  = [0.0, 0.0]
    
    for ii in range(B0.shape[0]):
        for jj in range(ni.shape[0]):
            # Magnetic field flux in nT
            field   = B0[ii]           
              
            # Density of cold/warm species (number/cc)
            ndensc     = np.zeros(N)
            ndensc[0]  = (1.0 - ni[jj])*n0                           
            ndensc[1]  = He_frac[0]*n0  
            ndensc[2]  = O_frac[ 0]*n0  
            
            ndensw     = np.zeros(N)
            ndensw[0]  = (ni[jj])*n0 
            ndensw[1]  = He_frac[1]*n0  
            ndensw[2]  = O_frac[ 1]*n0  
                
            # Perpendicular temperature (ev)
            temperp = np.zeros(N)
            temperp[0] = 50000.
            temperp[1] = 0.
            temperp[2] = 0.
            
            #OR
            
            # Warm species parallel beta
            betapar = np.zeros(N)
            betapar[0] = 10.
            betapar[1] = 0.
            betapar[2] = 0.
            
            # Temperature anisotropy
            A    = np.zeros(N)
            A[0] = 2.                                         
            A[1] = 0.
            A[2] = 0.
        
            if beta_flag == 1:
                FREQ, GR_RATE, STFLAG = conv.calculate_growth_rate(field, ndensc, ndensw, A, beta=betapar, norm_freq=0, maxfreq=1.0)
                temperp = None
            else:
                FREQ, GR_RATE, STFLAG = conv.calculate_growth_rate(field, ndensc, ndensw, A, temperp=temperp, norm_freq=0, maxfreq=1.0)
                betapar = None
            
            growth_max[ii, jj] = GR_RATE.max()
            
plot_CGR_space(plot_data_points=False, show_sample_points=True)

# =============================================================================
#     plot_growth_rate(FREQ, GR_RATE, STFLAG, ax1)
#     plt.xlim(0, 0.5)
#     plt.ylim(0, 18)
#     plt.show()
# =============================================================================
# =============================================================================
#     for ax0 in [ax1]:
#         ax0.set_xlabel('Frequency (Hz)', fontsize=14)
#         ax0.set_ylabel('Growth Rate ($\omega / V_g 10^{-7} cm^{-1}$)', rotation=90, fontsize=14)
#         set_figure_text(ax0)
#         ax0.set_xlim(0, FREQ[-1])
# =============================================================================
