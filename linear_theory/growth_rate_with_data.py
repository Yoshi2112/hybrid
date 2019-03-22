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

import rbsp_file_readers as rfr
import analysis_scripts as ascr
import fast_scripts     as fscr
import coordinate_conversions as cc

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


def plot_CGR_space():
    plt.ioff()
    fig    = plt.figure(1, figsize=(16,9))
    grids  = gs.GridSpec(2, 2, width_ratios=[1, 0.02], height_ratios=[0.05, 1])
    ax1    = fig.add_subplot(grids[1, 0])
    
    im1 = ax1.contourf(n0, B0, growth_max, 400, cmap='nipy_spectral')
    ax1.set_xlim(n_min, n_max)
    ax1.set_xlabel('Total ion density, $n_0$ $(cm^{-3})$')
    ax1.set_ylim(B_min, B_max)
    ax1.set_ylabel('Background magnetic field, $B_0$ (nT)')
    
    cax1 = fig.add_subplot(grids[1, 1])
    fig.colorbar(im1, cax=cax1, extend='both').set_label(r'Max. CGR ($\omega / V_g 10^{-7} cm^{-1}$)', rotation=90, fontsize=11)
    fig.subplots_adjust(wspace=0.02, hspace=0)
    #fig.tight_layout()
    plt.show()
    
    font    = 'monospace'
    top     = 1.07
    left    = 0.78
    h_space = 0.04
    
    ax1.text(0.00, top - 0.02, '$B_0 = $variable',         transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(0.00, top - 0.05, '$n_0 = $variable', transform=ax1.transAxes, fontsize=10, fontname=font)
    
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
    ax1.text(left + 1*h_space, top - 0.02, '{:>7.2f}'.format(H_frac[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 2*h_space, top - 0.02, '{:>7.2f}'.format(H_frac[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
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


if __name__ == '__main__':
    ###############################   
    #### LINEAR THEORY SECTION ####
    ###############################
# =============================================================================
#     N         = 3   # Number of species
#     beta_flag = 1   # Flag to define if beta or temperp used. Helps with plotting
#     
#     n_min = 20.   ; n_max = 220.    ; dn = 5.
#     B_min = 100.  ; B_max = 200.    ; db = 5.
#     
#     B0           = np.arange(B_min, B_max + db, db)
#     n0           = np.arange(n_min, n_max + dn, dn)
#     growth_max   = np.zeros((B0.shape[0], n0.shape[0]))
#     
#     H_frac  = [0.9, 0.1]
#     He_frac = [0.0, 0.0]
#     O_frac  = [0.0, 0.0]
#     
#     for ii in range(B0.shape[0]):
#         for jj in range(n0.shape[0]):
#             # Magnetic field flux in nT
#             field   = B0[ii]           
#               
#     
#             # Density of cold/warm species (number/cc)
#             ndensc     = np.zeros(N)
#             ndensc[0]  = H_frac[ 0]*n0[jj]                           
#             ndensc[1]  = He_frac[0]*n0[jj]  
#             ndensc[2]  = O_frac[ 0]*n0[jj]  
#             
#             ndensw     = np.zeros(N)
#             ndensw[0]  = H_frac[ 1]*n0[jj]  
#             ndensw[1]  = He_frac[1]*n0[jj]  
#             ndensw[2]  = O_frac[ 1]*n0[jj]  
#                 
#             # Perpendicular temperature (ev)
#             temperp = np.zeros(N)
#             temperp[0] = 50000.
#             temperp[1] = 0.
#             temperp[2] = 0.
#             
#             #OR
#             
#             # Warm species parallel beta
#             betapar = np.zeros(N)
#             betapar[0] = 10.
#             betapar[1] = 0.
#             betapar[2] = 0.
#             
#             # Temperature anisotropy
#             A    = np.zeros(N)
#             A[0] = 2.                                         
#             A[1] = 0.
#             A[2] = 0.
#         
#             if beta_flag == 1:
#                 FREQ, GR_RATE, STFLAG = conv.calculate_growth_rate(field, ndensc, ndensw, A, beta=betapar, norm_freq=0, maxfreq=1.0)
#                 temperp = None
#             else:
#                 FREQ, GR_RATE, STFLAG = conv.calculate_growth_rate(field, ndensc, ndensw, A, temperp=temperp, norm_freq=0, maxfreq=1.0)
#                 betapar = None
#             
#             growth_max[ii, jj] = GR_RATE.max()
#             
#     plot_CGR_space()
# =============================================================================
   
    ######################   
    #### DATA SECTION ####
    ###################### 
    drive      = 'G:'                                   # Drive containing RBSP path
    rbsp_path  = '%s//DATA//RBSP//'  % drive            # Path to data main RBSP directory

    time_start = np.datetime64('2013-06-30T23:15:00')
    time_end   = np.datetime64('2013-06-30T23:30:00')
    
    probe = 'B'
    pad   = 3600
    fmax  = 1.0
    
    LP_bg   = 1.0
    pc1_res = 50.
    overlap = 0.99 
    dt_64   = 1./64

    hope_params = ['Tperp_Tpar_he_30', 'Tperp_Tpar_o_30', 'Tperp_Tpar_p_30', 'Dens_p_30', 'Dens_he_30', 'Dens_o_30']

    times, mags, R, b_err = rfr.retrieve_RBSP_magnetic_data(rbsp_path, 
                                                    time_start, time_end, probe,
                                                    padding=[pad, pad])
            
    den_times, edens, den_error = rfr.retrieve_RBSP_electron_density_data(rbsp_path, time_start, time_end, probe)
    mags                        = ascr.gain_switch_despike(mags)

# =============================================================================
#     background   = ascr.clw_low_pass(np.copy(mags), LP_bg, dt_64)               # LP to get main field B0 (in mGSE)
#     b_mgse       = mags - background                                            # Detrend B in mGSE
#     fac_mags     = cc.ALL_to_FAC(b_mgse, R, background)                         # Convert magnetic field to Field-Aligned
# 
#     pc1_xpower, pc1_xtimes, pc1_xfreq = fscr.autopower_spectra(times, fac_mags[:, 0], time_start, 
#                                                     time_end, pc1_res, dt_64, overlap)
#                      
#     pc1_ypower, pc1_ytimes, pc1_yfreq = fscr.autopower_spectra(times, fac_mags[:, 1], time_start, 
#                                                     time_end, pc1_res, dt_64, overlap)
#                      
#     pc1_perp_power = np.log10(pc1_xpower + pc1_ypower)
#     
#     fig    = plt.figure(1, figsize=(16,9))                  # Initialize figure
#     grids  = gs.GridSpec(1, 2, width_ratios=[1, 0.01])      # Create gridspace
# 
#     ax1 = fig.add_subplot(grids[0, 0])
#     im1 = ax1.pcolormesh(pc1_xtimes, pc1_xfreq, pc1_perp_power.T, vmin=-7, vmax=1, cmap='nipy_spectral')
#     cax1 = fig.add_subplot(grids[0, 1])
#     fig.colorbar(im1, cax=cax1, extend='both').set_label(r'$log_{10}$ $\left( \frac{nT^2}{Hz} \right)$', rotation=90, fontsize=14)
# =============================================================================

