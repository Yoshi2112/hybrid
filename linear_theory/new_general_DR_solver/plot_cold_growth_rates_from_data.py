# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:30:20 2020

@author: Yoshi

Note: This script just copies the functions related to calculating the cold
dispersion/growth rates since the 'omura play' source script isn't a final
product
"""
import warnings, pdb, sys
import numpy as np
import matplotlib.pyplot as plt
from   scipy.special     import wofz
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append('F://Google Drive//Uni//PhD 2017//Data//Scripts//')
import rbsp_fields_loader as rfl
import rbsp_file_readers  as rfr 
import analysis_scripts   as ascr
import fast_scripts       as fscr

from growth_rates_from_RBSP         import extract_species_arrays
from dispersion_solver_multispecies import create_species_array

c  = 3e8
qp = 1.602e-19
mp = 1.673e-27

def Z(arg):
    '''
    Return Plasma Dispersion Function : Normalized Fadeeva function
    Plasma dispersion function related to Fadeeva function
    (Summers & Thorne, 1993) by i*sqrt(pi) factor.
    '''
    return 1j*np.sqrt(np.pi)*wofz(arg)


def nearest_index(items, pivot):
    closest_val = min(items, key=lambda x: abs(x - pivot))
    for ii in range(len(items)):
        if items[ii] == closest_val:
            return ii
    sys.exit('Error: Unable to find index')
        

def get_k_cold(w, Species):
    '''
    Calculate the k of a specific angular frequency w in a cold
    multicomponent plasma. Assumes a cold plasma (i.e. negates 
    thermal effects). Hot species cast to cold by including their
    plasma frequencies.
    
    This will give the cold plasma dispersion relation for the Species array
    specified, since the CPDR is surjective in w (i.e. only one possible k for each w)
    
    Omura et al. (2010)
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cold_sum = 0.0
        for ii in range(Species.shape[0]):
            cold_sum += Species[ii]['plasma_freq_sq'] / (w * (w - Species[ii]['gyrofreq']))
    
        k = np.sqrt(1 - cold_sum) * w / c
    return k


def warm_plasma_dispersion_relation(wr, k, Species):
    '''    
    Function used in scipy.fsolve minimizer to find roots of dispersion relation.
    Iterates over each k to find values of w that minimize to D(wr, k) = 0
    '''
    components = 0.0
    for ii in range(Species.shape[0]):
        sp = Species[ii]
        if sp['tper'] == 0:
            components += sp['plasma_freq_sq'] * wr / (sp['gyrofreq'] - wr)
        else:
            pdisp_arg   = (wr - sp['gyrofreq']) / (sp['vth_par']*k)
            pdisp_func  = Z(pdisp_arg)*sp['gyrofreq'] / (sp['vth_par']*k)
            brackets    = (sp['anisotropy'] + 1) * (wr - sp['gyrofreq'])/sp['gyrofreq'] + 1
            Is          = brackets * pdisp_func + sp['anisotropy']
            components += sp['plasma_freq_sq'] * Is
    return (wr ** 2) - (c * k) ** 2 + components


def solve_w_from_k(k, Species):
    '''
    This function solves the warm plasma dispersion relation for real frequency
    
    '''
    
    return


def linear_growth_rates(w, Species):
    '''
    Calculates the temporal and convective linear growth rates for a plasma
    composition contained in Species for each frequency w. Assumes a cold
    dispersion relation is valid for k but uses a warm approximation in the
    solution for D(w, k).
    
    Equations adapted from Chen et al. (2011) (or 2013?)
    
    To do:
        --- Need to actually solve w for each k just like with the Wang (2016)
            code. Actually just wr for D(wr, k) = 0 and then equations as before.
        --- Feed in k-series rather than w-series. Is this still decent as a 
            cold approximation? Maybe worth checking against paper.
        --- What makes an approximation cold/warm/hot?
              -- Cold negates any thermal stuff
              -- Warm assumes gamma << wr and solves D(wr, k) to then solve for gamma
              -- Hot solves wr + i*gamma in a 2D parameter space with no (?) assumptions
        --- Validate against plots shown in Chen (2013)
    
    Input values in SI?
     -- Frequencies in rad/s
     -- Anisotropies dimensionless
     -- Thermal velocity in m/s ?? 
    '''
    # Get k for each frequency to evaluate
    # -- Actually, this needs to be swapped around:
    #     1) Define k-space array (not w) in function call
    #     2) Numerically solve wr for each k
    #     3) Use *that* wr in the calculation for Di (and hence gamma, S)
    k  = get_k_cold(w, Species)
    
    # Calculate Dr/k_para
    w_der_sum = 0.0
    k_der_sum = 0.0
    Di        = 0.0
    for ii in range(Species.shape[0]):
        sp = Species[ii]
        
        # If cold
        if sp['tper'] == 0:
            w_der_sum += sp['plasma_freq_sq'] * sp['gyrofreq'] / (w - sp['gyrofreq'])**2
            k_der_sum += 0.0
            Di        += 0.0
        
        # If hot
        else:
            zs           = (w - sp['gyrofreq']) / (sp['vth_par']*k)
            Yz           = np.real(Z(zs))
            dYz          = -2*(1 + zs*Yz)
            A_bit        = (sp['anisotropy'] + 1) * w / sp['gyrofreq']
            
            # Calculate frequency derivative of Dr (sums bit)
            w_der_outsd  = sp['plasma_freq_sq']*sp['gyrofreq'] / (w*k*sp['vth_par'])
            w_der_first  = A_bit * Yz
            w_der_second = (A_bit - sp['anisotropy']) * w * dYz / (k * sp['vth_par']) 
            w_der_sum   += w_der_outsd * (w_der_first + w_der_second)
    
            # Calculate Di (sums bit)
            Di_bracket = 1 + (sp['anisotropy'] + 1) * (w - sp['gyrofreq']) / sp['gyrofreq']
            Di_after   = sp['gyrofreq'] / (k * sp['vth_par']) * np.sqrt(np.pi) * np.exp(- zs ** 2)
            Di        += sp['plasma_freq_sq'] * Di_bracket * Di_after
    
            # Calculate wavenumber derivative of Dr (sums bit)
            k_der_outsd  = sp['plasma_freq_sq']*sp['gyrofreq'] / (w*k*k*sp['vth_par'])
            k_der_first  = A_bit - sp['anisotropy']
            k_der_second = Yz + zs * dYz
            k_der_sum   += k_der_outsd * k_der_first * k_der_second
    
    # Get and return ratio
    Dr_wder = 2*w + w_der_sum
    Dr_kder = -2*k*c**2 - k_der_sum

    temporal_growth_rate   = - Di / Dr_wder
    group_velocity         = - Dr_kder / Dr_wder
    convective_growth_rate = - temporal_growth_rate / np.abs(group_velocity)
    return temporal_growth_rate, convective_growth_rate


def plot_growth_rates_2D(rbsp_path, time_start, time_end, probe, pad, norm=None, norm_B0=200.):
    '''
    Main function that downloads the data, queries the growth rates, and 
    plots a 2D colorplot of the temporal and convective growth rates. Might
    also include plugins to other data such as the Pc1 power to more directly
    look at the change. Also include growth rates?
    
    To Do: Add a normalize flag? At each point, this will normalize f by the 
    proton cylcotron frequency (which is varying in time). This may give slightly
    different results for when there are large perturbations in the background
    field due to the presence of HM?
    
    Also option to normalize based on set B rather than B0 at each time (equivalent
    to absolute measure)
    
    norm: None       (no normalization applied)
          'local'    (normalized based on B0 at each time)
          'absolute' (normalized based on B0 of norm_B0, default 200nT)
    
    How to set growth rate min/max?
    '''
    # Frequencies over which to solve (determines frequency cadence)
    Nf    = 1000
    f_max = 1.2
    f_min = 0.0
    freqs = np.linspace(f_max, f_min, Nf)
    w     = 2 * np.pi * freqs
    
    # Create species array for each time (determines time cadence)    
    times, B0, name, mass, charge, density, tper, ani, cold_dens = \
    extract_species_arrays(time_start, time_end, probe, pad,
                           rbsp_path='G://DATA//RBSP//', 
                           cmp=[70, 20, 10],
                           return_raw_ne=True,
                           HM_filter_mhz=50,
                           nsec=None)
    
    # Initialize empty arrays for GR returns
    Nt  = times.shape[0]
    TGR = np.zeros((Nt, Nf), dtype=np.float64)
    CGR = np.zeros((Nt, Nf), dtype=np.float64)
    
    # Calculate growth rate, normalize if flagged
    try:
        pcyc_abs = qp * norm_B0 * 1e-9 / mp
    except:
        print('Something up with norm_B0 flag = {}. Defaulting to 200nT'.format(norm_B0))
        pcyc_abs = qp * 200e-9 / mp
        
    for ii in range(times.shape[0]):
        Species, PP = create_species_array(B0[ii], name, mass, charge, density[:, ii],
                                            tper[:, ii], ani[:, ii])
        
        TGR[ii], CGR[ii] = linear_growth_rates(w, Species)
        
        # Normalize
        if norm is not None:
            if norm == 'local':
                pcyc_local = qp * PP['B0'] / mp
                TGR[ii]   /= pcyc_local
                CGR[ii]   /= pcyc_local
            else:
                if norm != 'absolute': 
                    print('Unknown normalization flag {}. Defaulting to absolute.'.format(norm))
                TGR[ii] /= pcyc_abs
                CGR[ii] /= pcyc_abs
                
    if True:
        # Set growth rate plot limits based on results
        TGR_min  = 0.0
        #TGR_mean = TGR[np.isnan(TGR) == False].mean()
        TGR_max  = TGR[np.isnan(TGR) == False].max()
        
        CGR_min  = 0.0
        #CGR_mean = CGR[np.isnan(CGR) == False].mean()
        CGR_max  = CGR[np.isnan(CGR) == False].max()
        
        if norm is not None:
            TGR_max = 1.0
            CGR_max = None
        
        # Plot the things
        plt.ioff()
        fig, axes = plt.subplots(2)
        
        im1 = axes[0].pcolormesh(times, freqs, TGR.T, vmin=TGR_min, vmax=TGR_max)
        axes[0].set_title('Temporal Growth Rate')
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].set_xlim(time_start, time_end)
        axes[0].set_ylim(f_min, f_max)
        
        divider1= make_axes_locatable(axes[0])
        cax1    = divider1.append_axes("right", size="2%", pad=0.5)
        fig.colorbar(im1, cax=cax1, label='$\gamma$', orientation='vertical', extend='both')
    
        im2 = axes[1].pcolormesh(times, freqs, CGR.T, vmin=CGR_min, vmax=CGR_max)
        axes[1].set_title('Convective Growth Rate')
        axes[1].set_xlim(time_start, time_end)
        axes[1].set_ylim(f_min, f_max)
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].set_xlabel('Times (UT)')
        
        divider2= make_axes_locatable(axes[1])
        cax2    = divider2.append_axes("right", size="2%", pad=0.5)
        fig.colorbar(im2, cax=cax2, label='S', orientation='vertical', extend='both')
        
        plt.show()
    
    return


def plot_max_GR_timeseries(rbsp_path, time_start, time_end, probe, pad, norm=None, norm_B0=200.):
    '''
    Note: Because this calculates in frequency space (using cold k) the 'max/min' values
    in the returned arrays are nan's because they are in a stop band.
    '''
    # Frequencies over which to solve (determines frequency cadence)
    Nf    = 1000
    f_max = 1.2
    f_min = 0.0
    freqs = np.linspace(f_max, f_min, Nf)
    w     = 2 * np.pi * freqs
    
    # Create species array for each time (determines time cadence)    
    times, B0, name, mass, charge, density, tper, ani, cold_dens = \
    extract_species_arrays(time_start, time_end, probe, pad,
                           rbsp_path='G://DATA//RBSP//', 
                           cmp=[70, 20, 10],
                           return_raw_ne=True,
                           HM_filter_mhz=50,
                           nsec=None)
        
    # Initialize empty arrays for GR returns
    Nt      = times.shape[0]
    max_TGR = np.zeros((Nt, 3), dtype=np.float64)
    max_CGR = np.zeros((Nt, 3), dtype=np.float64)
        
    for ii in range(times.shape[0]):
        Species, PP = create_species_array(B0[ii], name, mass, charge, density[:, ii],
                                            tper[:, ii], ani[:, ii])
        
        pcyc   = qp * B0[ii] / (2 * np.pi * mp)
        H_idx  = nearest_index(freqs, pcyc)
        He_idx = nearest_index(freqs, 0.25*pcyc)
        O_idx  = nearest_index(freqs, 0.0625*pcyc)
        
        TGR, CGR = linear_growth_rates(w, Species)
        
        # Mask nan's
        TGR[np.isnan(TGR) == True] = 0.0
        CGR[np.isnan(CGR) == True] = 0.0
        
        try:
            # H-band max growth
            if TGR[H_idx:He_idx].shape[0] == 0:
                max_TGR[ii, 0] = np.nan
                max_CGR[ii, 0] = np.nan
            else:
                max_TGR[ii, 0] = TGR[H_idx:He_idx].max()
                max_CGR[ii, 0] = CGR[H_idx:He_idx].max()
                
            # He-band max growth
            if TGR[He_idx: O_idx].shape[0] == 0:
                max_TGR[ii, 1] = np.nan
                max_CGR[ii, 1] = np.nan
            else:
                max_TGR[ii, 1] = TGR[He_idx: O_idx].max()
                max_CGR[ii, 1] = CGR[He_idx: O_idx].max()
                
            # O-band max growth
            if TGR[O_idx:].shape[0] == 0:
                max_TGR[ii, 2] = np.nan
                max_CGR[ii, 2] = np.nan
            else:
                max_TGR[ii, 2] = TGR[O_idx:].max()
                max_CGR[ii, 2] = CGR[O_idx:].max()
        except:
            pdb.set_trace()

    # Just Temporal Growth Rates in single overlaid plot
    if False:
        fig, ax = plt.subplots(sharex=True)
        ax.plot(times, max_TGR[:, 0], c='r')
        ax.plot(times, max_TGR[:, 1], c='b')
        ax.plot(times, max_TGR[:, 2], c='green')
        
    # Pc1 spectra and TGR/CGR (with log and linear) in two figures
    if False:
        ti, fac_mags, fac_elec, dt, e_flag, gyfreqs = rfl.load_both_fields(rbsp_path, time_start, time_end, probe, pad=3600)
        
        pc1_xpower, pc1_xtimes, pc1_xfreq = fscr.autopower_spectra(ti, fac_mags[:, 0], time_start, 
                                                     time_end, dt, overlap=0.99, df=25.0)
    
        pc1_ypower, pc1_ytimes, pc1_yfreq = fscr.autopower_spectra(ti, fac_mags[:, 1], time_start, 
                                                         time_end, dt, overlap=0.99, df=25.0)
        
        pc1_perp_power = np.log10(pc1_xpower[:, :] + pc1_ypower[:, :])
        
        plt.ioff()
        # Temporal Growth Rate
        fig1, axes1 = plt.subplots(3, figsize=(16,10), sharex=True)
        
        axes1[0].pcolormesh(pc1_xtimes, pc1_xfreq, pc1_perp_power.T, vmin=-7, vmax=1, cmap='jet')
        axes1[0].set_ylabel('Frequency (Hz)')
        axes1[0].set_ylim(f_min, f_max)
        axes1[0].set_title('Temporal Growth Rate (s) vs. Time :: {} :: Cold-k Approximation'.format(date_string))
        
        axes1[1].semilogy(times, max_TGR[:, 0], c='r', label='$H^{+}$')
        axes1[1].semilogy(times, max_TGR[:, 1], c='b', label='$He^{+}$')
        axes1[1].semilogy(times, max_TGR[:, 2], c='green', label='$O^{+}$')
        axes1[1].legend()
        axes1[1].set_ylabel('$\log_{10}(\gamma)$', rotation=0)
        
        axes1[2].plot(times, max_TGR[:, 0], c='r', label='$H^{+}$')
        axes1[2].plot(times, max_TGR[:, 1], c='b', label='$He^{+}$')
        axes1[2].plot(times, max_TGR[:, 2], c='green', label='$O^{+}$')
        axes1[2].legend()
        axes1[2].set_ylabel('$\gamma$', rotation=0)
        
        axes1[2].set_xlim(time_start, time_end)
        
        # Convective Growth Rate
        fig2, axes2 = plt.subplots(3, figsize=(16,10), sharex=True)
        
        axes2[0].pcolormesh(pc1_xtimes, pc1_xfreq, pc1_perp_power.T, vmin=-7, vmax=1, cmap='jet')
        axes2[0].set_ylabel('Frequency (Hz)')
        axes2[0].set_ylim(f_min, f_max)
        axes2[0].set_title('Temporal Growth Rate (s) vs. Time :: {} :: Cold-k Approximation'.format(date_string))
        
        axes2[1].semilogy(times, max_CGR[:, 0], c='r', label='$H^{+}$')
        axes2[1].semilogy(times, max_CGR[:, 1], c='b', label='$He^{+}$')
        axes2[1].semilogy(times, max_CGR[:, 2], c='green', label='$O^{+}$')
        axes2[1].legend()
        axes2[1].set_ylabel('$\log_{10}(S)$', rotation=0)
        
        axes2[2].plot(times, max_CGR[:, 0], c='r', label='$H^{+}$')
        axes2[2].plot(times, max_CGR[:, 1], c='b', label='$He^{+}$')
        axes2[2].plot(times, max_CGR[:, 2], c='green', label='$O^{+}$')
        axes2[2].legend()
        axes2[2].set_ylabel('$S$', rotation=0)
        
        axes2[2].set_xlim(time_start, time_end)
        
        plt.show()
        
    
    # Linear TGR and CGR with input parameters for calculation (looking for spikes!)
    if True:
        # TGR
        # B0
        # Cold densities 
        # HOPE densities
        # HOPE temperatures
        # HOPE anisotropies
        # RBSPICE densities
        # RBSPICE temperatures
        # RBSPICE anisotropies
        plt.ioff()
        for max_arr, title in zip([max_TGR, max_CGR], ['TGR', 'CGR']):
            fig, axes = plt.subplots(9, figsize=(8,20), sharex=True)
            
            axes[0].set_title('Max. {} for {} :: Derived from Satellite Data'.format(title, date_string))
            axes[0].plot(times, max_arr[:, 0], c='r')
            axes[0].plot(times, max_arr[:, 1], c='b')
            axes[0].plot(times, max_arr[:, 2], c='green')
            
            axes[1].plot(times, B0*1e9)
            axes[1].set_ylabel('B0\nnT')
            
            axes[2].plot(times, density[0]*1e-6, c='r')
            axes[2].plot(times, density[1]*1e-6, c='b')
            axes[2].plot(times, density[2]*1e-6, c='green')
            axes[2].set_ylabel('Cold $n_i$\n/cc')
            
            axes[3].plot(times, density[3]*1e-6, c='r')
            axes[3].plot(times, density[4]*1e-6, c='b')
            axes[3].plot(times, density[5]*1e-6, c='green')
            axes[3].set_ylabel('HOPE $n_i$\n/cc')
            
            axes[4].plot(times, density[6]*1e-6, c='r')
            axes[4].plot(times, density[7]*1e-6, c='b')
            axes[4].plot(times, density[8]*1e-6, c='green')
            axes[4].set_ylabel('RBSPICE $n_i$\n/cc')
            
            axes[5].plot(times, tper[3]*1e-3, c='r')
            axes[5].plot(times, tper[4]*1e-3, c='b')
            axes[5].plot(times, tper[5]*1e-3, c='green')
            axes[5].set_ylabel('HOPE $T_\perp$\nkeV')
            
            axes[6].plot(times, tper[6]*1e-3, c='r')
            axes[6].plot(times, tper[7]*1e-3, c='b')
            axes[6].plot(times, tper[8]*1e-3, c='green')
            axes[6].set_ylabel('RBSPICE $T_\perp$\nkeV')
            
            axes[7].plot(times, ani[3], c='r')
            axes[7].plot(times, ani[4], c='b')
            axes[7].plot(times, ani[5], c='green')
            axes[7].set_ylabel('HOPE $A_i$')
            
            axes[8].plot(times, ani[6], c='r')
            axes[8].plot(times, ani[7], c='b')
            axes[8].plot(times, ani[8], c='green')
            axes[8].set_ylabel('RBSPICE $A_i$')
            
            fig.subplots_adjust(hspace=0)
            fig.align_ylabels()
            
            for ax in axes:
                ax.set_xlim(time_start, time_end)
                
            if False:
                plt.show()
            else:
                svpath = save_dir + '{}_stackplot.png'.format(title)
                print('Plot saved as {}'.format(svpath))
                fig.savefig(svpath)
                plt.close('all')
        
    return





if __name__ == '__main__':
    # To Do:
    # Peaks to line up
    
    rbsp_path = 'G://DATA//RBSP//'
    save_drive= 'G://'
    
    time_start  = np.datetime64('2013-07-25T21:25:00')
    time_end    = np.datetime64('2013-07-25T21:47:00')
    probe       = 'a'
    pad         = 0
    
    date_string = time_start.astype(object).strftime('%Y%m%d')
    save_string = time_start.astype(object).strftime('%Y%m%d_%H%M_') + time_end.astype(object).strftime('%H%M')
    save_dir    = '{}NEW_LT//EVENT_{}//NEW_FIXED_DISPERSION_RESULTS//COLD_K_TIMESERIES//1D_MAX_GROWTH//'.format(save_drive, date_string)

    plot_max_GR_timeseries(rbsp_path, time_start, time_end, probe, pad, norm=None, norm_B0=200.)
    #plot_growth_rates_2D(rbsp_path, time_start, time_end, probe, pad, norm='absolute')