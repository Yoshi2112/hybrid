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
from   scipy.optimize    import fsolve
from   scipy.special     import wofz
from   mpl_toolkits.axes_grid1 import make_axes_locatable


sys.path.append('F://Google Drive//Uni//PhD 2017//Data//Scripts//')
import rbsp_fields_loader as rfl
import fast_scripts       as fscr

from growth_rates_from_RBSP         import extract_species_arrays
from dispersion_solver_multispecies import create_species_array

c  = 3e8
qp = 1.602e-19
mp = 1.673e-27

def nearest_index(items, pivot):
    closest_val = min(items, key=lambda x: abs(x - pivot))
    for ii in range(len(items)):
        if items[ii] == closest_val:
            return ii
    sys.exit('Error: Unable to find index')
    
    
def Z(arg):
    '''
    Return Plasma Dispersion Function : Normalized Fadeeva function
    Plasma dispersion function related to Fadeeva function
    (Summers & Thorne, 1993) by i*sqrt(pi) factor.
    '''
    return 1j*np.sqrt(np.pi)*wofz(arg)

def Y(arg):
    return np.real(Z(arg))


def hot_dispersion_eqn(w, k, Species):
    '''
    Function used in scipy.fsolve minimizer to find roots of dispersion relation
    for hot plasma approximation.
    Iterates over each k to find values of w that minimize to D(wr, k) = 0
    
    In this case, w is a vector [wr, wi] and fsolve is effectively doing a multivariate
    optimization.
    
    type_out allows purely real or purely imaginary (coefficient only) for root
    finding. Set as anything else for complex output.
    
    FSOLVE OPTIONS :: If bad solution, return np.nan?
    
    Eqns 1, 13 of Chen et al. (2013)
    '''
    wc = w[0] + 1j*w[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hot_sum = 0.0
        for ii in range(Species.shape[0]):
            sp = Species[ii]
            if sp['tper'] == 0:
                hot_sum += sp['plasma_freq_sq'] * wc / (sp['gyrofreq'] - wc)
            else:
                pdisp_arg   = (wc - sp['gyrofreq']) / (sp['vth_par']*k)
                pdisp_func  = Z(pdisp_arg)*sp['gyrofreq'] / (sp['vth_par']*k)
                brackets    = (sp['anisotropy'] + 1) * (wc - sp['gyrofreq'])/sp['gyrofreq'] + 1
                Is          = brackets * pdisp_func + sp['anisotropy']
                hot_sum    += sp['plasma_freq_sq'] * Is

    solution = (wc ** 2) - (c * k) ** 2 + hot_sum
    return np.array([solution.real, solution.imag])


def warm_dispersion_eqn(w, k, Species):
    '''    
    Function used in scipy.fsolve minimizer to find roots of dispersion relation
    for warm plasma approximation.
    Iterates over each k to find values of w that minimize to D(wr, k) = 0
    
    Eqn 14 of Chen et al. (2013)
    '''
    wr = w[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warm_sum = 0.0
        for ii in range(Species.shape[0]):
            sp = Species[ii]
            if sp['tper'] == 0:
                warm_sum   += sp['plasma_freq_sq'] * wr / (sp['gyrofreq'] - wr)
            else:
                pdisp_arg   = (wr - sp['gyrofreq']) / (sp['vth_par']*k)
                numer       = ((sp['anisotropy'] + 1)*wr - sp['anisotropy']*sp['gyrofreq'])
                Is          = sp['anisotropy'] + numer * Y(pdisp_arg) / (sp['vth_par']*k)
                warm_sum   += sp['plasma_freq_sq'] * Is
            
    solution = wr ** 2 - (c * k) ** 2 + warm_sum
    return np.array([solution, 0.0])


def cold_dispersion_eqn(w, k, Species):
    '''
    Function used in scipy.fsolve minimizer to find roots of dispersion relation
    for warm plasma approximation.
    Iterates over each k to find values of w that minimize to D(wr, k) = 0
    
    Eqn 19 of Chen et al. (2013)
    '''
    wr = w[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cold_sum = 0.0
        for ii in range(Species.shape[0]):
            cold_sum += Species[ii]['plasma_freq_sq'] * wr / (Species[ii]['gyrofreq'] - wr)
            
    solution = wr ** 2 - (c * k) ** 2 + cold_sum
    return np.array([solution, 0.0])


def get_warm_growth_rates(wr, k, Species):
    '''
    Calculates the temporal and convective linear growth rates for a plasma
    composition contained in Species for each frequency w. Assumes a cold
    dispersion relation is valid for k but uses a warm approximation in the
    solution for D(w, k).
    
    Equations adapted from Chen et al. (2013)
    '''    
    w_der_sum = 0.0
    k_der_sum = 0.0
    Di        = 0.0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for ii in range(Species.shape[0]):
            sp = Species[ii]
            
            # If cold
            if sp['tper'] == 0:
                w_der_sum += sp['plasma_freq_sq'] * sp['gyrofreq'] / (wr - sp['gyrofreq'])**2
                k_der_sum += 0.0
                Di        += 0.0
            
            # If hot
            else:
                zs           = (wr - sp['gyrofreq']) / (sp['vth_par']*k)
                Yz           = np.real(Z(zs))
                dYz          = -2*(1 + zs*Yz)
                A_bit        = (sp['anisotropy'] + 1) * wr / sp['gyrofreq']
                
                # Calculate frequency derivative of Dr (sums bit)
                w_der_outsd  = sp['plasma_freq_sq']*sp['gyrofreq'] / (wr*k*sp['vth_par'])
                w_der_first  = A_bit * Yz
                w_der_second = (A_bit - sp['anisotropy']) * wr * dYz / (k * sp['vth_par']) 
                w_der_sum   += w_der_outsd * (w_der_first + w_der_second)
        
                # Calculate Di (sums bit)
                Di_bracket = 1 + (sp['anisotropy'] + 1) * (wr - sp['gyrofreq']) / sp['gyrofreq']
                Di_after   = sp['gyrofreq'] / (k * sp['vth_par']) * np.sqrt(np.pi) * np.exp(- zs ** 2)
                Di        += sp['plasma_freq_sq'] * Di_bracket * Di_after
        
                # Calculate wavenumber derivative of Dr (sums bit)
                k_der_outsd  = sp['plasma_freq_sq']*sp['gyrofreq'] / (wr*k*k*sp['vth_par'])
                k_der_first  = A_bit - sp['anisotropy']
                k_der_second = Yz + zs * dYz
                k_der_sum   += k_der_outsd * k_der_first * k_der_second
    
    # Get and return ratio
    Dr_wder = 2*wr + w_der_sum
    Dr_kder = -2*k*c**2 - k_der_sum

    temporal_growth_rate   = - Di / Dr_wder
    group_velocity         = - Dr_kder / Dr_wder
    convective_growth_rate = - temporal_growth_rate / np.abs(group_velocity)
    return temporal_growth_rate, convective_growth_rate


def get_cold_growth_rates(wr, k, Species):
    '''
    Simplified version of the warm growth rate equation.
    '''
    w_der_sum = 0.0
    Di        = 0.0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for ii in range(Species.shape[0]):
            sp         = Species[ii]
            w_der_sum += sp['plasma_freq_sq'] * sp['gyrofreq'] / (wr - sp['gyrofreq'])**2
            
            if sp['vth_par'] != 0.0:
                # Calculate Di (sums bit)
                zs         = (wr - sp['gyrofreq']) / (sp['vth_par']*k)
                Di_bracket = 1 + (sp['anisotropy'] + 1) * (wr - sp['gyrofreq']) / sp['gyrofreq']
                Di_after   = sp['gyrofreq'] / (k * sp['vth_par']) * np.sqrt(np.pi) * np.exp(- zs ** 2)
                Di        += sp['plasma_freq_sq'] * Di_bracket * Di_after
    
    # Get and return ratio
    Dr_wder = 2*wr + w_der_sum
    Dr_kder = -2*k*c**2

    temporal_growth_rate   = - Di / Dr_wder
    group_velocity         = - Dr_kder / Dr_wder
    convective_growth_rate = - temporal_growth_rate / np.abs(group_velocity)
    return temporal_growth_rate, convective_growth_rate


def get_dispersion_relation(Species, k, approx='warm', return_all=False):
    '''
    Given a range of k, returns the real and imaginary parts of the plasma dispersion
    relation specified by the Species present.
    
    Type of dispersion relation controlled by 'approx' kwarg as:
        hot  :: Full dispersion relation for complex w = wr + i*gamma
        warm :: Small growth rate approximation that allows D(wr, k) = 0
        cold :: Dispersion relation used for wr, growth rate calculated as per warm
    '''
    gyfreqs, counts = np.unique(Species['gyrofreq'], return_counts=True)
    
    # Remove electron count, 
    gyfreqs = gyfreqs[1:]
    N_solns = counts.shape[0] - 1
    
    # fsolve arguments
    eps    = 0.01           # Offset used to supply initial guess (since right on w_cyc returns an error)
    tol    = 1e-10          # Absolute solution convergence tolerance in rad/s
    fev    = 1000000        # Maximum number of iterations
    Nk     = k.shape[0]     # Number of wavenumbers to solve for
    
    # Solution and error arrays :: Two-soln array for wr, gamma.
    # Gamma is only solved for in the DR with the hot approx
    # The other two require calls to another function with a wr arg.
    PDR_solns = np.ones((Nk, N_solns, 2)) * eps
    ier       = np.zeros((Nk, N_solns), dtype=int)
    msg       = np.zeros((Nk, N_solns), dtype='<U256')

    # Initial guesses
    for ii in range(1, N_solns):
        PDR_solns[0, ii - 1]  = np.array([[gyfreqs[-ii - 1] * 1.05, 0.0]])

    if approx == 'hot':
        func = hot_dispersion_eqn
    elif approx == 'warm':
        func = warm_dispersion_eqn
    elif approx == 'cold':
        func = cold_dispersion_eqn
    else:
        sys.exit('ABORT :: kwarg approx={} invalid. Must be \'cold\', \'warm\', or \'hot\'.'.format(approx))
    
    
    # Define function to solve for (all have same arguments)
    for jj in range(N_solns):
        for ii in range(1, Nk):
            PDR_solns[ii, jj], infodict, ier[ii, jj], msg[ii, jj] =\
                fsolve(func, x0=PDR_solns[ii - 1, jj], args=(k[ii], Species), xtol=tol, maxfev=fev, full_output=True)

        # Solve for k[0] using initial guess of k[1]
        PDR_solns[0, jj], infodict, ier[0, jj], msg[0, jj] =\
            fsolve(func, x0=PDR_solns[1, jj], args=(k[0], Species), xtol=tol, maxfev=fev, full_output=True)


    # Filter out bad solutions
    # Why only warm approx giving me bad solutions?
    if True:
        N_bad = 0
        for jj in range(N_solns):
            for ii in range(1, Nk):
                if ier[ii, jj] == 5:
                    PDR_solns[ii, jj] = np.nan
                    N_bad += 1
        print('{} solutions filtered for {} approximation.'.format(N_bad, approx))

    # Solve for growth rate/convective growth rate here 
    # (how to do for hot? Maybe just make NoneType)
    # (Would have to calculate Vg from dw/dk, finite difference?)
    if approx == 'hot':
        conv_growth = None
    elif approx == 'warm':
        for jj in range(N_solns):
            PDR_solns[:, jj, 1], conv_growth = get_warm_growth_rates(PDR_solns[:, jj, 0], k, Species)
    elif approx == 'cold':
        for jj in range(N_solns):
            PDR_solns[:, jj, 1], conv_growth = get_cold_growth_rates(PDR_solns[:, jj, 0], k, Species)
    else:
        sys.exit('ABORT :: kwarg approx={} invalid. Must be \'cold\', \'warm\', or \'hot\'.'.format(approx))
    
    if return_all == False:
        return PDR_solns, conv_growth
    else:
        return PDR_solns, conv_growth, ier, msg





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
        
        TGR[ii], CGR[ii] = get_warm_growth_rates(w, Species)
        
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
        
        TGR, CGR = get_warm_growth_rates(w, Species)
        
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

    mp       = 1.673e-27                        # Proton mass (kg)
    qi       = 1.602e-19                        # Elementary charge (C)
    e0       = 8.854e-12                        # Permittivity of free space
    _n0      = 100e6                            # Electron number density in /m3                      
    _B0      = 144e-9                           # Background magnetic field in T
    nhh      = 0.03                             # Fraction hot hydrogen
    nHe      = 0.20                             # Fraction warm helium
    THe      = 100.0                              # Helium temperature (eV) -- Does this have to be altered for 'total temp'?
    
    _name    = np.array(['Hot H'  , 'Cold H'        , 'Cold He'])               # Species label
    _mass    = np.array([1.0      , 1.0             , 4.0      ]) * mp          # Mass   in proton masses (kg)
    _charge  = np.array([1.0      , 1.0             , 1.0      ]) * qi          # Change in elementary units (C)
    _density = np.array([nhh      , 1.0 - nhh - nHe , nHe      ]) * _n0         # Density as a fraction of n0 (/m3)
    _tpar    = np.array([25e3     , 0.0             , THe      ])               # Parallel temperature in eV
    _ani     = np.array([1.0      , 0.0             , 0.0      ])               # Temperature anisotropy
    _tper    = (_ani + 1) * _tpar                                               # Perpendicular temperature in eV
    
    _Spec, _PP = create_species_array(_B0, _name, _mass, _charge, _density, _tper, _ani)

    _kh  = np.sqrt(_n0 * qi ** 2 / (mp * e0))/c
    _Nk  = 1000
    _k   = np.linspace(0.0, 2.0*_kh, _Nk)
    pcyc = qi * _B0 / mp 

    cold_DR, cold_CGR = get_dispersion_relation(_Spec, _k, approx='cold', return_all=False)
    warm_DR, warm_CGR = get_dispersion_relation(_Spec, _k, approx='warm', return_all=False)
    hot_DR ,  hot_CGR = get_dispersion_relation(_Spec, _k, approx='hot' , return_all=False)

    # Plot the things
    k_norm    = _k / _kh
    fig, axes = plt.subplots(2, sharex=True)
    
    for jj in range(hot_DR.shape[1]):
        axes[0].semilogx(k_norm,  hot_DR[:, jj, 0] / pcyc, ls='-' , c='r', lw=0.5)
        axes[0].semilogx(k_norm, warm_DR[:, jj, 0] / pcyc, ls='--', c='b', lw=0.5)
        axes[0].semilogx(k_norm, cold_DR[:, jj, 0] / pcyc, ls='--', c='k', lw=0.5)
        axes[0].set_ylabel('$\omega_r/\Omega_h$', fontsize=16)
        
        axes[1].semilogx(k_norm,  hot_DR[:, jj, 1] / pcyc, ls='-' , c='r', label='Full', lw=0.5)
        axes[1].semilogx(k_norm, warm_DR[:, jj, 1] / pcyc, ls='--', c='b', label='Warm', lw=0.5)
        axes[1].semilogx(k_norm, cold_DR[:, jj, 1] / pcyc, ls='--', c='k', label='Cold', lw=0.5)
        
    axes[1].set_ylabel('$\gamma/\Omega_h$', fontsize=16)
    axes[1].set_xlabel('$k_\parallel/k_h$', fontsize=16)
    
    axes[0].set_ylim(0.0, 0.8)
    axes[0].set_xlim(1e-2, 2)
    axes[1].set_ylim(-0.01, 0.06)
    axes[1].set_xlim(1e-2, 2)
    
    fig.subplots_adjust(hspace=0)


    #plot_max_GR_timeseries(rbsp_path, time_start, time_end, probe, pad, norm=None, norm_B0=200.)
    #plot_growth_rates_2D(rbsp_path, time_start, time_end, probe, pad, norm='absolute')