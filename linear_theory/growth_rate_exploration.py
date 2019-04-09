# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:56:09 2019

@author: Yoshi
"""
import sys
data_scripts_dir = 'F://Google Drive//Uni//PhD 2017//Data//Scripts'
sys.path.append(data_scripts_dir)
import matplotlib.cm as cm
import numpy                        as np
import matplotlib.pyplot            as plt
import matplotlib.gridspec          as gs
from timeit import default_timer    as timer
import os
import pdb

def calculate_growth_rate(field, ndensc, ndensw, ANI, beta, norm_ampl=0, norm_freq=0, NPTS=500, maxfreq=1.0):
    '''
    Calculates the convective growth rate S as per eqn. 6 of Kozyra (1984). Plasma parameters passed as 
    length 3 numpy.ndarrays, one for each H+, He+, O+
    
    INPUT:
        field  -- Magnetic field intensity, nT
        ndensc -- Cold plasma density, /cm3
        ndensw -- Warm plasma density, /cm3
        ANI    -- Temperature anisotropy of each species
        
    OPTIONAL:
        temperp   -- Perpendicular temperature of species warm component, eV
        beta      -- Parallel plasma beta of species warm component
        norm_ampl -- Flag to normalize growth rate to max value (0: No, 1: Yes). Default 0
        norm_ampl -- Flag to normalize frequency to proton cyclotron units. Default 0
        NPTS      -- Number of sample points up to maxfreq. Default 500
        maxfreq   -- Maximum frequency to calculate for in proton cyclotron units. Default 1.0
        
    NOTE: At least one of temperp or beta must be defined.
    '''
    # Perform input checks 
    N   = ndensc.shape[0]
    
    for ii in np.arange(N):
        if ndensw[ii] != 0:
            if beta[ii] == 0:
                raise ValueError('Zero beta not allowed for warm components.')
        
    # CONSTANTS
    PMASS   = 1.673E-27
    MUNOT   = 1.25660E-6
    CHARGE  = 1.602E-19
    
    # OUTPUT PARAMS
    growth = np.zeros(NPTS)                         # Output growth rate variable
    x      = np.zeros(NPTS)                         # Input normalized frequency
    stop   = np.zeros(NPTS)                         # Stop band flag (0, 1)
    
    # Set here since these are constant (and wrong?)
    M    = np.zeros(N)
    M[0] = 1.0 
    M[1] = 4.0  
    M[2] = 16.0 
    
    # LOOP PARAMS
    step  = maxfreq / float(NPTS)
    FIELD = field*1.0E-9                 # convert to Tesla
        
    NCOLD   = ndensc * 1.0E6
    NHOT    = ndensw * 1.0E6
    etac    = NCOLD / NHOT[0]        # Needs a factor of Z ** 2?
    etaw    = NHOT  / NHOT[0]        # Here too
    numer   = M * (etac+etaw)
    
    TPAR    = np.zeros(N)
    for ii in range(N):
        if NHOT[ii] != 0:
            TPAR[ii]    = (FIELD*FIELD/(2.0*MUNOT)) * beta[ii] / NHOT[ii]
        
    alpha = np.sqrt(2.0 * TPAR / PMASS)
    return alpha.max() / 3e8

    for k in np.arange(1, NPTS):
          x[k]   = k*step
          denom  = 1.0 - M*x[k]
          
          sum1  = 0.0
          prod2 = 1.0
          
          for i in np.arange(N):
               prod2   *= denom[i]
               prod     = 1.0
               temp     = denom[i]
               denom[i] = numer[i]
               
               for j in np.arange(N):
                   prod *= denom[j]
               
               sum1    += prod
               denom[i] = temp
          
          sum2 = 0.0
          arg4 = prod2 / sum1
    
          # Check for stop band.
          if (arg4 < 0.0) and (x[k] > 1.0/M[N-1]):
              growth[k] = 0.0
              stop[k]   = 1
          else:
             arg3 = arg4 / (x[k] ** 2)
             
             
             for i in np.arange(N):
                if (NHOT[i] > 1.0E-3):
                     arg1  = np.sqrt(np.pi) * etaw[i] / ((M[i]) ** 2 * alpha[i])       # Outside term
                     arg1 *= ((ANI[i] + 1.0) * (1.0 - M[i]*x[k]) - 1.0)                # Inside square brackets (multiply by outside)
                     arg2  = (-etaw[i] / M[i]) * (M[i]*x[k] - 1.0) ** 2 / beta[i]*arg3
                     
                     sum2 += arg1*np.exp(arg2)
             
             growth[k] = sum2*arg3/2.0
    
    ###########################
    ### NORMALIZE AND CLEAN ###
    ###########################    
    for ii in np.arange(NPTS):
        if (growth[ii] < 0.0):
            growth[ii] = 0.0
          
    if (norm_freq == 0):
        cyclotron  = CHARGE*FIELD/(2.0*np.pi*PMASS)
        x         *= cyclotron
          
    if (norm_ampl == 1):
        growth /= growth.max()
    else:
        growth *= 1e9
    return x, growth, stop


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
    
    ax1.text(0.00, top - 0.02, '$B_0 = $variable'     , transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(0.00, top - 0.05, '$n_0 = $variable' % n0, transform=ax1.transAxes, fontsize=10, fontname=font)
    
    ax1.text(left + 0.06,  top, 'Cold'    , transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 0.099, top, 'Warm'    , transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 0.143, top, '$A_i$'   , transform=ax1.transAxes, fontsize=10, fontname=font)
    
    ax1.text(left + 0.192, top, r'$\beta_{\parallel}$'                    , transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 4.2*h_space, top - 0.02, '{:>7.2f}'.format(betapar[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 4.2*h_space, top - 0.04, '{:>7.2f}'.format(betapar[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 4.2*h_space, top - 0.06, '{:>7.2f}'.format(betapar[2]), transform=ax1.transAxes, fontsize=10, fontname=font)

    ax1.text(left + 0.49*h_space, top - 0.02, ' H+:'                    , transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 1*h_space, top - 0.02, '{:>7.3f}'.format(H_frac[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 2*h_space, top - 0.02, '{:>7.3f}'.format(H_frac[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 3*h_space, top - 0.02, '{:>7.2f}'.format(A[0]),      transform=ax1.transAxes, fontsize=10, fontname=font)
    
    ax1.text(left + 0.49*h_space, top - 0.04, 'He+:'                     , transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 1*h_space, top - 0.04, '{:>7.3f}'.format(He_frac[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 2*h_space, top - 0.04, '{:>7.3f}'.format(He_frac[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 3*h_space, top - 0.04, '{:>7.2f}'.format(A[1])      , transform=ax1.transAxes, fontsize=10, fontname=font)
    
    ax1.text(left + 0.49*h_space, top - 0.06, ' O+:'                    , transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 1*h_space, top - 0.06, '{:>7.3f}'.format(O_frac[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 2*h_space, top - 0.06, '{:>7.3f}'.format(O_frac[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
    ax1.text(left + 3*h_space, top - 0.06, '{:>7.2f}'.format(A[2])     , transform=ax1.transAxes, fontsize=10, fontname=font)
    return


def plot_CGR_space(plot_data_points=True, show_sample_points=False, save=False):
    plt.ioff()
    fig    = plt.figure(1, figsize=(16,9))
    grids  = gs.GridSpec(2, 2, width_ratios=[1, 0.02], height_ratios=[0.05, 1])
    ax1    = fig.add_subplot(grids[1, 0])
    
    im = ax1.contourf(n0, B0, growth_max, np.linspace(0, 8000, 401), cmap=cm.nipy_spectral)
    
    ax1.set_xlim(n_min, n_max)
    ax1.set_ylim(B_min, B_max)
    ax1.set_xlabel('Total Ion Density, $n_0 (cm^{-3})$')
    ax1.set_ylabel('Background magnetic field, $B_0$ (nT)')
    ax1.set_title(r'Max. $S(%.1fHz < f < %.1fHz)$ for each $B_0$ with %d%% Warm Ion Fraction' % (f_min, f_max, hot_ion_frac * 100))
    
    cax1 = fig.add_subplot(grids[1, 1])
    
    fig.colorbar(im, cax=cax1, extend='both').set_label(r'Max. CGR ($\omega / V_g 10^{-7} cm^{-1}$)', rotation=90, fontsize=11)
    fig.subplots_adjust(wspace=0.02, hspace=0)
    set_figure_text(ax1)

    if plot_data_points == True:
        n_data = np.array([38, 160, 38, 160, 150, 150, 350, 350])
        B_data = np.array([158, 158, 134, 134, 200, 260, 200, 260])
        ax1.scatter(n_data, B_data, label='Event Data Limits', marker='x', c='cyan')

    if show_sample_points == True:
        for ii in np.arange(n0.shape[0]):
            for jj in np.arange(B0.shape[0]):
                ax1.scatter(n0[ii], B0[jj], c='k', marker='o', s=1, alpha=0.25)
                
    if save == True:
        series   = '//Frequency Limited Max//He_only//%d_percent_warm//' % (hot_ion_frac * 100.)
        fullpath = save_dir + series + 'linear_He_{:06}.png'.format(count)
        
        if os.path.exists(save_dir + series) == False:
            os.makedirs(save_dir + series)
            
        fig.savefig(fullpath, edgecolor='none')
        plt.close('all')
    else:    
        plt.show()
    return



if __name__ == '__main__':
    save_dir  = 'G://LINEAR_THEORY//'
    
    N         = 3   # Number of species
    
    n_min   = 20.  ; n_max   = 420.   ; dn   = 5.
    B_min   = 100. ; B_max   = 300.   ; db   = 5.
    nhe_min = 0.0  ; nhe_max = 0.99   ; dnhe = 0.0025
    
    
    B0           = np.arange(B_min ,  B_max   + db  , db )
    n0           = np.arange(n_min ,  n_max   + dn  , dn )
    nhe          = np.arange(nhe_min, nhe_max + dnhe, dnhe)

    growth_max   = np.zeros((B0.shape[0], n0.shape[0]))
    
    frequency_limit = True
    f_min           = 0.2
    f_max           = 0.8
    max_alpha = 0.0
    start_time = timer()
    for hot_ion_frac in [0.10]:
        count = 0
        for kk in np.arange(1, nhe.shape[0]):
            print('Plotting for {}% He'.format(nhe[kk]*100.))
            for ii in np.arange(B0.shape[0]):
                for jj in np.arange(n0.shape[0]):
                    # Magnetic field flux in nT
                    field   = B0[ii]     
                    
                    n_cold  = n0[jj]*(1 - hot_ion_frac)                 # Split density into cold
                    n_warm  = n0[jj]*hot_ion_frac                       # and warm components
                    
                    # Density of cold/warm species (number/cc)
                    ndensc     = np.zeros(N)
                    ndensc[0]  = (1 - nhe[kk]) * n_cold                          
                    ndensc[1]  = nhe[kk]       * n_cold 
                    ndensc[2]  = 0.0  
                    
                    ndensw     = np.zeros(N)
                    ndensw[0]  = (1 - nhe[kk]) * n_warm 
                    ndensw[1]  = nhe[kk]       * n_warm  
                    ndensw[2]  = 0.0  
        
                    H_frac  = 100. * np.array([ndensc[0], ndensw[0]]) / n0[jj]
                    He_frac = 100. * np.array([ndensc[1], ndensw[1]]) / n0[jj]
                    O_frac  = 100. * np.array([ndensc[2], ndensw[2]]) / n0[jj]
                    
                    # Warm species parallel beta
                    betapar = np.zeros(N)
                    betapar[0] = 10.
                    betapar[1] = 10.
                    betapar[2] = 0.
                    
                    # Temperature anisotropy
                    A    = np.zeros(N)
                    A[0] = 2.                                         
                    A[1] = 2.
                    A[2] = 0.
                    
                    this_alpha = calculate_growth_rate(field, ndensc, ndensw, A, beta=betapar, norm_freq=0, maxfreq=1.0)
                    if this_alpha > max_alpha:
                        max_alpha = this_alpha
# =============================================================================
#                     FREQ, GR_RATE, STFLAG = calculate_growth_rate(field, ndensc, ndensw, A, beta=betapar, norm_freq=0, maxfreq=1.0)
#                     
#                     if frequency_limit == True:
#                         idx_min = np.where(abs(FREQ - f_min) == abs(FREQ - f_min).min())[0][0]
#                         idx_max = np.where(abs(FREQ - f_max) == abs(FREQ - f_max).min())[0][0]
#                         
#                         try:
#                             growth_max[ii, jj] = GR_RATE[GR_RATE != np.inf][idx_min: idx_max].max()
#                         except:
#                             growth_max[ii, jj] = np.inf
#                     else:
#                         growth_max[ii, jj]     = GR_RATE[GR_RATE != np.inf].max()
#     
#             plot_CGR_space(save=True)
#             count += 1
# =============================================================================

    print(max_alpha)