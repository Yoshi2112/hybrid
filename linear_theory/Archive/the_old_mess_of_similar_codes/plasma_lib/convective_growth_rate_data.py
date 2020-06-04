# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:56:09 2019

@author: Yoshi
"""

# This program calculates the growth rate for the cyclotron instability for compositions of
# singly ionized H, He, and O, including both warm and cold species. It also allows variable
# anisotropy ratios, perpendicular temperatures (ev), and magnetic flux densities.
# NOTE: All values of the growth rate are multiplied by 1.0E9,  unless the amplitudes
# are normalized, and then the maximum value is 1.0.
# Original Pascal code by John C. Samson
#
# Converted to IDL : Colin L. Waters
# July, 2009
#
# Converted to Python : Joshua S. Williams
# October, 2018
#
# Minimal changes to make program Python-compatible. Default parameters are those used 
# in Fraser et al. (1989)  Figure 7/Table 1.
#
# Made callable as a function: March, 2019
import os
import pdb
import numpy                        as np
import matplotlib.pyplot            as plt
import matplotlib.gridspec          as gs
import extract_parameters_from_data as data


def set_figure_text(ax):
    font    = 'monospace'
    fsize   = 10
    top     = 1.0               # Top of text block
    left    = 1.01               # Left boundary of text block
    
    TPER_kev  = _temp_perp  / 1e3
    TPER_kev2 = _temp_perp2 / 1e3
    
    ax.text(left, top - 0.02, '$B_0 = ${}nT'.format(round(_field, 2)),           transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, top - 0.05, '$n_0 = $%.1f$cm^{-3}$' % n0, transform=ax.transAxes, fontsize=fsize, fontname=font)

    v_space = 0.03              # Vertical spacing between lines
    
    c_top   = top - 0.15         # 'Cold' top
    w_top   = top - 0.35         # 'Warm' top
    h_top   = top - 0.55         # 'Hot' (Warmer) top
    
    
    # Cold Table
    ax.text(left, c_top + 1*v_space, 'Cold Population', transform=ax.transAxes, fontsize=fsize+2, fontname=font, fontweight='bold')
    ax.text(left + 0.05, c_top, r'  $n_c (cm^{-3})$', transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, c_top - 1*v_space, ' H+:   {:>7.2f} '.format(round(_ndensc[0], 2)) , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, c_top - 2*v_space, 'He+:   {:>7.2f} '.format(round(_ndensc[1], 2))    , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, c_top - 3*v_space, ' O+:   {:>7.2f} '.format(round(_ndensc[2], 2))    , transform=ax.transAxes, fontsize=fsize, fontname=font)

    
    # Warm Table
    ax.text(left, w_top + 1*v_space, 'Warm Population (HOPE)', transform=ax.transAxes, fontsize=fsize+2, fontname=font, fontweight='bold')
    ax.text(left + 0.05, w_top, r'  $n_i (cm^{-3})$  $T_{\perp} (keV)$   $A_i$ ', transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, w_top - 1*v_space, ' H+:   {:>5.2f}   {:>5}   {:>5}'.format(round(_ndensw[0], 2), round(TPER_kev[0], 2), round(_A[0], 3)) , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, w_top - 2*v_space, 'He+:   {:>5.2f}   {:>5}   {:>5}'.format(round(_ndensw[1], 2), round(TPER_kev[1], 2), round(_A[1], 3))    , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, w_top - 3*v_space, ' O+:   {:>5.2f}   {:>5}   {:>5}'.format(round(_ndensw[2], 2), round(TPER_kev[2], 2), round(_A[2], 3))    , transform=ax.transAxes, fontsize=fsize, fontname=font)


    # Hot Table
    ax.text(left, h_top + 1*v_space, 'Warm Population (SPICE)', transform=ax.transAxes, fontsize=fsize+2, fontname=font, fontweight='bold')
    ax.text(left + 0.05, h_top, r'  $n_i (cm^{-3})$  $T_{\perp} (keV)$   $A_i$ ', transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, h_top - 1*v_space, ' H+:   {:>5.2f}   {:>5}   {:>5}'.format(round(_ndensw2[0], 2), round(TPER_kev2[0], 2), round(_A2[0], 3)) , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, h_top - 2*v_space, 'He+:   {:>5.2f}   {:>5}   {:>5}'.format(round(_ndensw2[1], 2), round(TPER_kev2[1], 2), round(_A2[1], 3)) , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, h_top - 3*v_space, ' O+:   {:>5.2f}   {:>5}   {:>5}'.format(round(_ndensw2[2], 2), round(TPER_kev2[2], 2), round(_A2[2], 3)) , transform=ax.transAxes, fontsize=fsize, fontname=font)
    return

def calculate_growth_rate(field, ndensc, ndensw, ANI, temperp=None, beta=None, norm_ampl=0, norm_freq=0, NPTS=1000, maxfreq=1.0):
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
        norm_freq -- Flag to normalize frequency to proton cyclotron units. Default 0
        NPTS      -- Number of sample points up to maxfreq. Default 500
        maxfreq   -- Maximum frequency to calculate for in proton cyclotron units. Default 1.0
        
    NOTE: At least one of temperp or beta must be defined. Still seems to give inf values for values of 
    the cyclotron frequency, even when those densities are zero.
    '''
    # Perform input checks 
    N   = ndensc.shape[0]
    
    if temperp is None and beta is None:
        raise ValueError('Either beta or tempperp must be defined.')
    elif temperp is not None and beta is not None:
        pdb.set_trace()
        print('Both temperp and beta arrays defined, defaulting to temperp array values...')
        beta = None
        
    # CONSTANTS
    PMASS   = 1.673E-27
    MUNOT   = 1.25660E-6
    EVJOULE = 6.242E18
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
    etac    = NCOLD / NHOT[0]       
    etaw    = NHOT  / NHOT[0] 
    numer   = M * (etac+etaw)
    
    if beta is None:
        TPERP   = temperp / EVJOULE
        TPAR    = TPERP / (1.0 + ANI)
        bet     = NHOT*TPAR / (FIELD*FIELD/(2.0*MUNOT))
    else:
        bet     = beta
        TPAR    = np.zeros(N)
        for jj in range(N):
            if NHOT[jj] != 0:
                TPAR[jj]    = (FIELD*FIELD/(2.0*MUNOT)) * bet[jj] / NHOT[jj]
        
    alpha = np.sqrt(2.0 * TPAR /  (M * PMASS))      # M added in denominator

    for k in range(1, NPTS):
          x[k]   = k*step
          denom  = 1.0 - M*x[k]
          
          sum1  = 0.0
          prod2 = 1.0
          
          for i in range(N):
               prod2   *= denom[i]
               prod     = 1.0
               temp     = denom[i]
               denom[i] = numer[i]
               
               for j in range(N):
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
             
             
             for i in range(N):
                if (NHOT[i] > 1.0E-3):
                     arg1  = np.sqrt(np.pi) * etaw[i] / ((M[i]) ** 2 * alpha[i])       # Outside term
                     arg1 *= ((ANI[i] + 1.0) * (1.0 - M[i]*x[k]) - 1.0)                # Inside square brackets (multiply by outside)
                     arg2  = (-etaw[i] / M[i]) * (M[i]*x[k] - 1.0) ** 2 / bet[i]*arg3
                     
                     sum2 += arg1*np.exp(arg2)
             
    
             growth[k] = sum2*arg3/2.0
    
    ###########################
    ### NORMALIZE AND CLEAN ###
    ###########################    
    for jj in range(NPTS):
        if (growth[jj] < 0.0):
            growth[jj] = 0.0
            
        if np.isnan(growth[jj]) == True:
            growth[jj] = np.inf
          
    if (norm_freq == 0):
        cyclotron  = CHARGE*FIELD/(2.0*np.pi*PMASS)
        x         *= cyclotron
          
    if (norm_ampl == 1):
        growth /= growth.max()
    else:
        growth *= 1e9
    return x, growth, stop



if __name__ == '__main__':
    _NPTS     = 1000
    output    = 'save'
    overwrite = False
    figtext   = True
    
    PMASS   = 1.673E-27
    MUNOT   = 1.25660E-6
    EVJOULE = 6.242E18
    CHARGE  = 1.602E-19
    
    time_start = np.datetime64('2013-07-25T21:00:00')
    time_end   = np.datetime64('2013-07-25T22:00:00')
    probe      = 'a'
    pad        = 0
    
    date_string = time_start.astype(object).strftime('%Y%m%d')
    save_string = time_start.astype(object).strftime('%Y%m%d_%H%M')
    save_dir    = 'G://EVENTS//event_{}//LINEAR_THEORY_CGR//'.format(date_string)
    data_path   = save_dir + 'cgr_history_{}'.format(save_string)
    
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
        
    times, mag, cold_dens, hope_dens, hope_tp, hope_A, spice_dens, spice_tp, spice_A =\
    data.load_and_interpolate_plasma_params(time_start, time_end, probe, pad)
    
    all_cgr = np.zeros((times.shape[0], _NPTS, 3, 2), dtype=np.float64)
    
    for ii in range(times.shape[0]):
        print('Calculating CGRs for {}'.format(times[ii]))
        _ndensc     = cold_dens[:, ii]
        _ndensw     = hope_dens[:, ii]
        _temp_perp  = hope_tp[:, ii]
        _A          = hope_A[:, ii]
        
        _ndensw2    = spice_dens[:, ii]
        _temp_perp2 = spice_tp[:, ii]
        _A2         = spice_A[:, ii]
        
        _field      = mag[ii]
        n0          = _ndensc.sum() + _ndensw.sum() + _ndensw2.sum()
        
        fr1, gr1, st1 = calculate_growth_rate(mag[ii], cold_dens[:, ii], hope_dens[:, ii], hope_A[:, ii], 
                            temperp=hope_tp[:, ii], norm_freq=0, maxfreq=1.0, NPTS=_NPTS)
        
        fr2, gr2, st2 = calculate_growth_rate(mag[ii], cold_dens[:, ii], spice_dens[:, ii], spice_A[:, ii], 
                            temperp=spice_tp[:, ii], norm_freq=0, maxfreq=1.0, NPTS=_NPTS)
        
        # Store HOPE growth rate
        all_cgr[ii, :, 0, 0] = fr1
        all_cgr[ii, :, 1, 0] = gr1
        all_cgr[ii, :, 2, 0] = st1
        
        # Store SPICE growth rate
        all_cgr[ii, :, 0, 1] = fr2
        all_cgr[ii, :, 1, 1] = gr2
        all_cgr[ii, :, 2, 1] = st2
        
        ############################
        ## DO THE PLOTTING THINGS ##
        ############################
        plt.ioff()
        fig     = plt.figure(figsize=(16, 10))
        grid    = gs.GridSpec(1, 1)
        
        ax1    = fig.add_subplot(grid[0, 0])
        
        fig.text(0.67, 0.971, '{}'.format(times[ii]))
        
        ax1.plot(fr1, gr1, c='blue', label='HOPE (<50keV)')
        ax1.plot(fr2, gr2, c='red',  label='SPICE (<1MeV)')
        
        for kk in range(st1.shape[0] - 1):
            if st1[kk] == 1:
                ax1.axvspan(fr1[kk], fr1[kk + 1], color='grey', alpha=0.1)            # PLOT STOP BAND
        
        for kk in range(st1.shape[0] - 1):
            if st2[kk] == 1:
                ax1.axvspan(fr2[kk], fr2[kk + 1], color='grey', alpha=0.1)            # PLOT STOP BAND
        
        if ax1.get_ylim()[1] > 10:
            y_max = None
        else:
            y_max = 10
        
        ax1.set_ylim(0, y_max)
        ax1.set_xlim(0, fr1[-1])
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel(r'CGR $(cm^{-1})$')

        ax1.set_title('Convective Growth Rate: Single Hot Population')
        ax1.legend(loc=1)
        
        if figtext == True:
            set_figure_text(ax1)
            
        fig.tight_layout()
        fig.subplots_adjust(right=0.80)
            
        if output == 'show':
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()
        elif output == 'save':
            figsave_path = save_dir + 'linearcgr_{}_{:04}.png'.format(save_string, ii)
            print('Saving {}'.format(figsave_path))
            fig.savefig(figsave_path)
            plt.close('all')
            
    print('Saving history...')
    np.savez(data_path, all_cgr=all_cgr)