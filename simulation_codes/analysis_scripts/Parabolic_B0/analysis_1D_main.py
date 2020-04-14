# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:56:34 2016

@author: c3134027
"""
import numpy as np
import numba as nb
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from   matplotlib.gridspec import GridSpec
import os
import pdb

import analysis_backend as bk
import analysis_config  as cf
import dispersions      as disp
import get_growth_rates as ggg

q   = 1.602e-19               # Elementary charge (C)
c   = 3e8                     # Speed of light (m/s)
me  = 9.11e-31                # Mass of electron (kg)
mp  = 1.67e-27                # Mass of proton (kg)
e   = -q                      # Electron charge (C)
mu0 = (4e-7) * np.pi          # Magnetic Permeability of Free Space (SI units)
kB  = 1.38065e-23             # Boltzmann's Constant (J/K)
e0  = 8.854e-12               # Epsilon naught - permittivity of free space

'''
Aim: To populate this script with plotting routines ONLY. Separate out the 
processing/loading/calculation steps into other modules that can be called.    

For parabolic with ABC's - Which cells are 'real'? (Important for k plots)

B field
First real cell: ND (LHS boundary)
Last  real cell: ND + NX (RHS boundary)
Cell range     : ND : ND + NX + 1

E field
First cell: ND
Last  cell: ND+NX-1
Cell range: ND:ND+NX
'''

@nb.njit()
def calc_poynting(bx, by, bz, ex, ey, ez):
    '''
    Calculates point value of Poynting flux using E, B 3-vectors
    '''
    # Poynting Flux: Time, space, component
    S = np.zeros((bx.shape[0], bx.shape[1], 3))
    
    for ii in range(bx.shape[0]):       # For each time
        print('Calculating Poynting flux for time ',ii)
        for jj in range(bx.shape[1]):   # For each point in space
            S[ii, jj, 0] = ey[ii, jj] * bz[ii, jj] - ez[ii, jj] * by[ii, jj]
            S[ii, jj, 1] = ez[ii, jj] * bx[ii, jj] - ex[ii, jj] * bz[ii, jj]
            S[ii, jj, 2] = ex[ii, jj] * by[ii, jj] - ey[ii, jj] * bx[ii, jj]
            
    S *= 1 / mu0
    return  S

def plot_spatial_poynting(saveas='poynting_space_plot', save=False, log=False):
    '''
    Plots poynting flux at each gridpoint point in space
    
    -- Need to interpolate B to cell centers
    -- S = E x H = 1/mu0 E x B
    -- Loop through each cell     : E(t), B(t), S(t)
    -- Maybe also do some sort of S, x plot
    -- And/or do dynamic Poynting spectrum
    '''
    plt.ioff()

    t, bx, by, bz, ex, ey, ez, vex, vey, vez, te, jx, jy, jz, qdens,\
        field_sim_time, damping_array = cf.get_array(get_all=True)

    bx, by, bz = bk.interpolate_B_to_center(bx, by, bz, zero_boundaries=False)
    S          = calc_poynting(bx, by, bz, ex, ey, ez)
    
    ## PLOT IT
    vlim = None
    #if True:
    for ii, comp in zip(range(3), ['x', 'y', 'z']):
        #ii = 2; comp = 'z'
        print('Creating plots for S{}'.format(comp))
        for tmax, lbl in zip([60, None], ['min', 'full']):
            fig, ax = plt.subplots(1, figsize=(15, 10))
            # 
            if log == True:
                im1 = ax.pcolormesh(cf.E_nodes, t, S[:, :, ii],
                     norm=colors.SymLogNorm(linthresh=1e-7, vmin=vlim, vmax=vlim),  
                     cmap='bwr')
                suffix = '_log'
            else:
                im1 = ax.pcolormesh(cf.E_nodes, t, S[:, :, ii], cmap='bwr', vmin=vlim, vmax=vlim)
                suffix = ''
            
            cb  = fig.colorbar(im1)
            cb.set_label('$S_%s$'%comp)
            
            if comp == 'x':
                suff = 'FIELD-ALIGNED DIRECTION'
            else:
                suff = 'TRANSVERSE DIRECTION'
            
            ax.set_title('Power Propagation :: Time vs. Space :: {}'.format(suff), fontsize=14)
            ax.set_ylabel(r't (s)', rotation=0, labelpad=15)
            ax.set_xlabel('x (m)')
            ax.set_ylim(0, tmax)
            
            ax.set_xlim(cf.grid_min, cf.grid_max)
            ax.axvline(cf.xmin, c='w', ls=':', alpha=1.0)
            ax.axvline(cf.xmax, c='w', ls=':', alpha=1.0)
            ax.axvline(cf.grid_mid, c='w', ls=':', alpha=0.75)   
                
            if save == True:
                fullpath = cf.anal_dir + saveas + '_s{}_{}'.format(comp, lbl) + suffix + '.png'
                plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
                plt.close('all')
    return


def plot_tx(component='By', saveas='tx_plot', save=False, log=False):
    plt.ioff()

    t, arr = cf.get_array(component)
    
    if component[0] == 'B':
        arr *= 1e9
        x    = cf.B_nodes
    else:
        arr *= 1e3
        x    = cf.E_nodes
    
    ## PLOT IT
    for tmax, lbl in zip([60, None], ['min', 'full']):
        fig, ax = plt.subplots(1, figsize=(15, 10))
        
        if log == True:
            im1 = ax.pcolormesh(x, t, abs(arr),
                           norm=colors.LogNorm(vmin=1e-3, vmax=None), cmap='nipy_spectral')
            suffix = '_log'
        else:
            im1 = ax.pcolormesh(x, t, arr, cmap='nipy_spectral', vmin=0, vmax=100)      # Remove f[0] since FFT[0] >> FFT[1, 2, ... , k]
            suffix = ''
        
        cb  = fig.colorbar(im1)
        
        if component[0] == 'B':
            cb.set_label('nT', rotation=0)
        else:
            cb.set_label('mV/m', rotation=0)
    
        ax.set_title('Time-Space ($t-x$) Plot :: {} component'.format(component.upper()), fontsize=14)
        ax.set_ylabel(r't (s)', rotation=0, labelpad=15)
        ax.set_xlabel('x (m)')
        ax.set_ylim(0, tmax)
        
        ax.set_xlim(cf.grid_min, cf.grid_max)
        ax.axvline(cf.xmin, c='w', ls=':', alpha=1.0)
        ax.axvline(cf.xmax, c='w', ls=':', alpha=1.0)
        ax.axvline(cf.grid_mid, c='w', ls=':', alpha=0.75)   
            
        if save == True:
            fullpath = cf.anal_dir + saveas + '_{}_{}'.format(component.lower(), lbl) + suffix + '.png'
            plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
            print('t-x Plot saved')
            plt.close('all')
    return


def plot_wx(component='By', saveas='wx_plot', linear_overlay=False, save=False, pcyc_mult=None):
    plt.ioff()
    ftime, wx = disp.get_wx(component)
    
    if component[0] == 'B':
        x = cf.B_nodes
    else:
        x = cf.E_nodes
        
    f  = np.fft.rfftfreq(ftime.shape[0], d=cf.dt_field)
    
    ## PLOT IT
    fig = plt.figure(1, figsize=(15, 10))
    ax  = fig.add_subplot(111)

    im1 = ax.pcolormesh(x, f, wx, cmap='nipy_spectral')      # Remove f[0] since FFT[0] >> FFT[1, 2, ... , k]
    fig.colorbar(im1)
    
    lbl  = [r'$\Omega_{H^+}$', r'$\Omega_{He^+}$', r'$\Omega_{O^+}$']
    clr  = ['white', 'yellow', 'red']    
    M    = np.array([1., 4., 16.])
    
    for ii in range(3):
        if cf.species_present[ii] == True:
            cyc    = q * cf.Bc[:, 0] / (2 * np.pi * mp * M[ii])
            ax.plot(cf.B_nodes, cyc, linestyle='--', c=clr[ii], label=lbl[ii])
    
    if linear_overlay == True:
        try:
            freqs, cgr, stop = disp.get_cgr_from_sim()
            max_idx          = np.where(cgr == cgr.max())
            max_lin_freq     = freqs[max_idx]
            plt.axhline(max_lin_freq, c='green', linestyle='--', label='CGR')
        except:
            pass

    ax.set_title('Frequency-Space ($\omega-x$) Plot :: {} component'.format(component.upper()), fontsize=14)
    ax.set_ylabel(r'f (Hz)', rotation=0, labelpad=15)
    ax.set_xlabel('x (m)')
    
    if pcyc_mult is not None:
        ax.set_ylim(0, pcyc_mult*cyc[0])
    else:
        ax.set_ylim(0, None)
      
    ax.set_xlim(cf.grid_min, cf.grid_max)
    ax.axvline(cf.xmin, c='w', ls=':', alpha=1.0)
    ax.axvline(cf.xmax, c='w', ls=':', alpha=1.0)
    ax.axvline(cf.grid_mid, c='w', ls=':', alpha=0.75)   
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('w-x Plot saved')
        plt.close('all')
    return


def plot_kt(component='By', saveas='kt_plot', save=False):
    plt.ioff()
    k, ftime, kt, st, en = disp.get_kt(component)
    
    fig = plt.figure(1, figsize=(15, 10))
    ax  = fig.add_subplot(111)
    
    im1 = ax.pcolormesh(k, ftime, kt, cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k] antialiased=True
    fig.colorbar(im1)
    ax.set_title('Wavenumber-Time ($k-t$) Plot :: {} component'.format(component.upper()), fontsize=14)
    ax.set_ylabel(r'$\Omega_i t$', rotation=0)
    ax.set_xlabel(r'$k (m^{-1}) \times 10^6$')
    #ax.set_ylim(0, 15)
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close('all')
        print('k-t Plot saved')
    return


def plot_wk(component='By', saveas='wk_plot' , dispersion_overlay=False, save=False, pcyc_mult=None):
    plt.ioff()
    
    k, f, wk = disp.get_wk(component)

    xlab = r'$k (\times 10^{-6}m^{-1})$'
    ylab = r'f (Hz)'

    fig = plt.figure(1, figsize=(15, 10))
    ax  = fig.add_subplot(111)
    
    im1 = ax.pcolormesh(1e6*k[1:], f[1:], np.log10(wk[1:, 1:].real), cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k]
    fig.colorbar(im1)
    ax.set_title(r'$\omega/k$ Dispersion Plot :: {} component'.format(component.upper()), fontsize=14)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    
# =============================================================================
#     # Doesn't work because cyclotron frequencies aren't constant in x
#     clr  = ['white', 'yellow', 'red'] 
#     lbl  = [r'$\Omega_{H^+}$', r'$\Omega_{He^+}$', r'$\Omega_{O^+}$']
#     M    = np.array([1., 4., 16.])
#     
#     for ii in range(3):
#         if cf.species_present[ii] == True:
#             cyc  = q * cf.B_eq / (2 * np.pi * mp * M[ii])
#             
#             ax.axhline(cyc, linestyle='--', c=clr[ii], label=lbl[ii])
#     
#     if pcyc_mult is not None:
#         ax.set_ylim(0, pcyc_mult*cyc[0])
#     else:
#         ax.set_ylim(0, None)
#         
#     if dispersion_overlay == True:
#         '''
#         Some weird factor of about 2pi inaccuracy? Is this inherent to the sim? Or a problem
#         with linear theory? Or problem with the analysis?
#         '''
#         try:
#             k_vals, CPDR_solns, warm_solns = disp.get_linear_dispersion_from_sim(k)
#             for ii in range(CPDR_solns.shape[1]):
#                 ax.plot(k_vals, CPDR_solns[:, ii]*2*np.pi,      c='k', linestyle='--', label='CPDR')
#                 ax.plot(k_vals, warm_solns[:, ii].real*2*np.pi, c='k', linestyle='-',  label='WPDR')
#         except:
#             pass
#         
#     ax.legend(loc=2, facecolor='grey')
# =============================================================================
    
    ax.set_ylim(0, None)
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('w-k saved')
        plt.close('all')
    return


def plot_wk_AGU(component='By', saveas='wk_plot' , dispersion_overlay=False, save=False, pcyc_mult=None):
    plt.ioff()
    
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    
    fontsize = 18
    
    k, f, wk = disp.get_wk(component)

    xlab = r'$k (\times 10^{-6}m^{-1})$'
    ylab = r'f (Hz)'

    fig = plt.figure(1, figsize=(15, 10))
    ax  = fig.add_subplot(111)
    
    im1 = ax.pcolormesh(1e6*k[1:], f[1:], np.log10(wk[1:, 1:].real), cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k]
    fig.colorbar(im1).set_label(r'$log_{10}$(Power)', rotation=90, fontsize=fontsize)
    ax.set_title(r'$\omega/k$ Dispersion Plot for {}'.format(component), fontsize=fontsize+4)
    ax.set_ylabel(ylab, fontsize=fontsize)
    ax.set_xlabel(xlab, fontsize=fontsize)
    
    clr  = ['black', 'green', 'red'] 
    lbl  = [r'$f_{H^+}$', r'$f_{He^+}$', r'$f_{O^+}$']
    M    = np.array([1., 4., 16.])
    cyc  = q * cf.B0 / (2 * np.pi * mp * M)
    for ii in range(3):
        if cf.species_present[ii] == True:
            ax.axhline(cyc[ii], linestyle='--', c=clr[ii], label=lbl[ii])
    
    ax.set_xlim(0, None)
    
    if pcyc_mult is not None:
        ax.set_ylim(0, pcyc_mult*cyc[0])
    else:
        ax.set_ylim(0, None)
    
    if dispersion_overlay == True:
        '''
        Some weird factor of about 2pi inaccuracy? Is this inherent to the sim? Or a problem
        with linear theory? Or problem with the analysis?
        '''
        try:
            k_vals, CPDR_solns, warm_solns = disp.get_linear_dispersion_from_sim(k)
            for ii in range(CPDR_solns.shape[1]):
                ax.plot(k_vals, CPDR_solns[:, ii]*2*np.pi,      c='k', linestyle='--', label='CPDR')
                ax.plot(k_vals, warm_solns[:, ii].real*2*np.pi, c='k', linestyle='-',  label='WPDR')
        except:
            pass
        
    ax.legend(loc=2, facecolor='white', prop={'size': fontsize})
        
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('w-k for component {} saved'.format(component.lower()))
        plt.close('all')
    return


def plot_dynamic_spectra(component='By', saveas='power_spectra', save=False, ymax=None, cell=None,
                         overlap=0.99, win_idx=None, slide_idx=None, df=50):
    
    if ymax is None:
        dynspec_folderpath = cf.anal_dir + '//field_dynamic_spectra//{}//'.format(component.upper())
    else:
        dynspec_folderpath = cf.anal_dir + '//field_dynamic_spectra//{}_ymax{}//'.format(component.upper(), ymax)
    
    if os.path.exists(dynspec_folderpath) == False:
        os.makedirs(dynspec_folderpath)
        
    if cell is None:
        cell = cf.NX // 2
        
    plt.ioff()
    
    powers, times, freqs = disp.autopower_spectra(component=component, overlap=overlap, win_idx=win_idx,
                                                  slide_idx=slide_idx, df=df, cell_idx=cell)
    
    fig = plt.figure(1, figsize=(15, 10))
    ax  = fig.add_subplot(111)

    im1 = ax.pcolormesh(times, freqs, powers.T,
                          cmap='jet')
    
    fig.colorbar(im1)
    ax.set_title(r'{} Dynamic Spectra :: Cell {}'.format(component.upper(), cell), fontsize=14)
    ax.set_ylabel(r'Frequency (Hz)', rotation=90)
    ax.set_xlabel(r'Time (s)')
    ax.set_ylim(0, ymax)
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        
        fig.savefig(dynspec_folderpath + 'dynamic_spectra_{}_{}.png'.format(component.lower(), cell), edgecolor='none')
        
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close('all')
        print('Dynamic spectrum (field {}, cell {}) saved'.format(component.upper(), cell))
    else:
        plt.show()
    return


def plot_energies(normalize=True, save=False):
    mag_energy, electron_energy, particle_energy, total_energy = bk.get_energies()

    ftime_sec  = cf.dt_field    * np.arange(mag_energy.shape[0])
    ptime_sec  = cf.dt_particle * np.arange(particle_energy.shape[0])

    fig     = plt.figure(figsize=(15, 7))
    ax      = plt.subplot2grid((7, 7), (0, 0), colspan=6, rowspan=7)

    if normalize == True:
        ax.plot(ftime_sec, mag_energy      / mag_energy[0],      label = r'$U_B$', c='green')
        ax.plot(ftime_sec, electron_energy / electron_energy[0], label = r'$U_e$', c='orange')
        ax.plot(ptime_sec, total_energy    / total_energy[0],    label = r'$Total$', c='k')
        
        for jj in range(cf.Nj):
            ax.plot(ptime_sec, particle_energy[:, jj, 0] / particle_energy[0, jj, 0],
                     label=r'$K_{E\parallel}$ %s' % cf.species_lbl[jj], c=cf.temp_color[jj], linestyle=':')
            
            ax.plot(ptime_sec, particle_energy[:, jj, 1] / particle_energy[0, jj, 1],
                     label=r'$K_{E\perp}$ %s' % cf.species_lbl[jj], c=cf.temp_color[jj], linestyle='-')
    else:
        ax.plot(ftime_sec, mag_energy,      label = r'$U_B$', c='green')
        ax.plot(ftime_sec, electron_energy, label = r'$U_e$', c='orange')
        ax.plot(ptime_sec, total_energy,    label = r'$Total$', c='k')
        
        for jj in range(cf.Nj):
            ax.plot(ptime_sec, particle_energy[:, jj, 0],
                     label=r'$K_{E\parallel}$ %s' % cf.species_lbl[jj], c=cf.temp_color[jj], linestyle=':')
            
            ax.plot(ptime_sec, particle_energy[:, jj, 1],
                     label=r'$K_{E\perp}$ %s' % cf.species_lbl[jj], c=cf.temp_color[jj], linestyle='-')
    
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.2))
    fig.tight_layout()

    percent_ion = np.zeros(cf.Nj)
    for jj in range(cf.Nj):
        tot_ion         = particle_energy[:, jj, 0] + particle_energy[:, jj, 1]
        percent_ion[jj] = round(100.*(tot_ion[-1] - tot_ion[0]) / tot_ion[0], 2)

    percent_elec  = round(100.*(electron_energy[-1] - electron_energy[0]) / electron_energy[0], 2)
    percent_mag   = round(100.*(mag_energy[-1]      - mag_energy[0])      / mag_energy[0], 2)
    percent_total = round(100.*(total_energy[-1]    - total_energy[0])    / total_energy[0], 2)

    fsize = 14; fname='monospace'
    plt.figtext(0.85, 0.92, r'$\Delta E$ OVER RUNTIME',            fontsize=fsize+2, fontname=fname)
    plt.figtext(0.85, 0.92, '________________________',            fontsize=fsize+2, fontname=fname)
    plt.figtext(0.85, 0.88, 'TOTAL   : {:>7}%'.format(percent_total),  fontsize=fsize,  fontname=fname)
    plt.figtext(0.85, 0.84, 'MAGNETIC: {:>7}%'.format(percent_mag),    fontsize=fsize,  fontname=fname)
    plt.figtext(0.85, 0.80, 'ELECTRON: {:>7}%'.format(percent_elec),   fontsize=fsize,  fontname=fname)

    for jj in range(cf.Nj):
        plt.figtext(0.85, 0.76-jj*0.04, 'ION{}    : {:>7}%'.format(jj, percent_ion[jj]), fontsize=fsize,  fontname=fname)

    ax.set_xlabel('Time (seconds)')
    ax.set_xlim(0, ptime_sec[-1])

    if normalize == True:
        ax.set_title('Normalized Energy Distribution in Simulation Space')
        ax.set_ylabel('Normalized Energy', rotation=90)
        fullpath = cf.anal_dir + 'norm_energy_plot'
        fig.subplots_adjust(bottom=0.07, top=0.96, left=0.04)
    else:
        ax.set_title('Energy Distribution in Simulation Space')
        ax.set_ylabel('Energy (Joules)', rotation=90)
        fullpath = cf.anal_dir + 'energy_plot'
        fig.subplots_adjust(bottom=0.07, top=0.96, left=0.055)

    if save == True:
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
    
    plt.close('all')
    print('Energy plot saved')
    return


def plot_ion_energy_components(normalize=True, save=True, tmax=600):
    mag_energy, electron_energy, particle_energy, total_energy = bk.get_energies()
    
    if normalize == True:
        for jj in range(cf.Nj):
            particle_energy[:, jj] /= particle_energy[0, jj]
    
    lpad = 20
    plt.ioff()
    
    for jj in range(cf.Nj):
        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(figsize=(18, 10), nrows=2, ncols=2)
        fig.subplots_adjust(hspace=0)
        
        ax1.plot(cf.time_radperiods_particle, particle_energy[:, jj, 1])
        ax3.plot(cf.time_radperiods_particle, particle_energy[:, jj, 0])
        
        ax2.plot(cf.time_radperiods_particle, particle_energy[:, jj, 1])
        ax4.plot(cf.time_radperiods_particle, particle_energy[:, jj, 0])
        
        ax1.set_ylabel(r'Perpendicular Energy', rotation=90, labelpad=lpad)
        ax3.set_ylabel(r'Parallel Energy', rotation=90, labelpad=lpad)
        
        for ax in [ax1, ax2]:
            ax.set_xticklabels([])
                    
        for ax in [ax1, ax3]:
            ax.set_xlim(0, tmax)
            
        for ax in [ax2, ax4]:
            ax.set_xlim(0, cf.time_radperiods_field[-1])
                
        for ax in [ax3, ax4]:
            ax.set_xlabel(r'Time $(\Omega^{-1})$')
                
        plt.suptitle('{} ions'.format(cf.species_lbl[jj]), fontsize=20, x=0.5, y=.93)
        plt.figtext(0.125, 0.05, 'Total time: {:.{p}g}s'.format(cf.time_seconds_field[-1], p=6), fontweight='bold')
        fig.savefig(cf.anal_dir + 'ion_energy_species_{}.png'.format(jj), facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close('all')
    return


def plot_helical_waterfall(title='', save=True, overwrite=False, it_max=None):
    ftime, By_raw         = cf.get_array('By')
    ftime, Bz_raw         = cf.get_array('Bz')
    
    if it_max is None:
        it_max = ftime.shape[0]
    
    ftime, Bt_pos, Bt_neg = bk.get_helical_components(overwrite)

    By_pos = Bt_pos.real
    By_neg = Bt_neg.real
    Bz_pos = Bt_pos.imag
    Bz_neg = Bt_neg.imag
    
    sig_fig = 3
    
    # R: 2, 20 -- NR: 1, 15 -- 
    skip   = 10
    amp    = 50.                 # Amplitude multiplier of waves:
    
    sep    = 1.
    dark   = 1.0
    cells  = np.arange(cf.NC + 1)

    plt.ioff()
    fig1 = plt.figure(figsize=(18, 10))
    ax1  = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax2  = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

    for ii in np.arange(it_max):
        if ii%skip == 0:
            ax1.plot(cells, amp*(By_pos[ii] / By_pos.max()) + sep*ii, c='k', alpha=dark)
            ax2.plot(cells, amp*(By_neg[ii] / By_neg.max()) + sep*ii, c='k', alpha=dark)

    ax1.set_title('By: +ve Helicity')
    ax2.set_title('By: -ve Helicity')
    plt.suptitle(title)
    
    for ax in [ax1, ax2]:
        ax.set_xlim(0, cells.shape[0])
        ax.set_ylim(0, None)
        ax.set_xlabel('Cell Number')
    ax1.set_ylabel('Time slice, dt = {:g}s'.format(float('{:.{p}g}'.format(cf.dt_field, p=sig_fig))))
        
    fig2 = plt.figure(figsize=(18, 10))
    ax3 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax4 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

    for ii in np.arange(it_max):
        if ii%skip == 0:
            ax3.plot(cells, amp*(Bz_pos[ii] / Bz_pos.max()) + sep*ii, c='k', alpha=dark)
            ax4.plot(cells, amp*(Bz_neg[ii] / Bz_neg.max()) + sep*ii, c='k', alpha=dark)

    ax3.set_title('Bz: +ve Helicity')
    ax4.set_title('Bz: -ve Helicity')
    plt.suptitle(title)
    
    for ax in [ax3, ax4]:
        ax.set_xlim(0, cells.shape[0])
        ax.set_ylim(0, None)
        ax.set_xlabel('Cell Number')
    ax3.set_ylabel('Time slice, dt = {:g}s'.format(float('{:.{p}g}'.format(cf.dt_field, p=sig_fig))))
    
    fig3 = plt.figure(figsize=(18, 10))
    ax5  = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax6  = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

    for ii in np.arange(it_max):
        if ii%skip == 0:
            ax5.plot(cells, amp*(By_raw[ii] / By_raw.max()) + sep*ii, c='k', alpha=dark)
            ax6.plot(cells, amp*(Bz_raw[ii] / Bz_raw.max()) + sep*ii, c='k', alpha=dark)

    ax5.set_title('By Raw')
    ax6.set_title('Bz Raw')
    plt.suptitle(title)
    
    for ax in [ax5, ax6]:
        ax.set_xlim(0, cells.shape[0])
        ax.set_ylim(0, None)
        ax.set_xlabel('Cell Number')
    ax3.set_ylabel('Time slice, dt = {:g}s'.format(float('{:.{p}g}'.format(cf.dt_field, p=sig_fig))))
    
    if save == True:
        fig1.subplots_adjust(bottom=0.07, top=0.96, left=0.04)
        fig1.subplots_adjust(wspace=0.05)
        ax2.set_yticklabels([])
        fig1.savefig(cf.anal_dir + 'by_helicity_t{}.png'.format(it_max), facecolor=fig1.get_facecolor(), edgecolor='none')
        
        fig2.subplots_adjust(bottom=0.07, top=0.96, left=0.04)
        fig2.subplots_adjust(wspace=0.05)
        ax4.set_yticklabels([])
        fig2.savefig(cf.anal_dir + 'bz_helicity_t{}.png'.format(it_max), facecolor=fig2.get_facecolor(), edgecolor='none')
        
        fig3.subplots_adjust(bottom=0.07, top=0.96, left=0.04)
        fig3.subplots_adjust(wspace=0.05)
        ax6.set_yticklabels([])
        fig3.savefig(cf.anal_dir + 'raw_fields_t{}.png'.format(it_max), facecolor=fig3.get_facecolor(), edgecolor='none')

    plt.close('all')
    return


def plot_helicity_colourplot(title='', save=True, overwrite=False, log=False):
    
    ftime, Bt_pos, Bt_neg = bk.get_helical_components(overwrite)

    xarr   = np.arange(cf.NC + 1) * cf.dx

    By_pos = Bt_pos.real
    By_neg = Bt_neg.real
    Bz_pos = Bt_pos.imag
    Bz_neg = Bt_neg.imag

    sig_fig = 3
    
    plt.ioff()
    
    ## Y-AXIS FIELD ##
    fig1 = plt.figure(constrained_layout=True, figsize=(18, 10))
    gs   = GridSpec(2, 2, figure=fig1, hspace=0, width_ratios=[1.0, 0.05])
    
    ax1  = fig1.add_subplot(gs[0, 0])
    ax2  = fig1.add_subplot(gs[1, 0])
    cax1 = fig1.add_subplot(gs[0, 1])
    cax2 = fig1.add_subplot(gs[1, 1])

    if log == True:
        im1 = ax1.pcolormesh(xarr, ftime, abs(By_pos),
                       norm=colors.LogNorm(vmin=1e-3, vmax=None), cmap='nipy_spectral')
        
        im2 = ax2.pcolormesh(xarr, ftime, abs(By_neg),
                       norm=colors.LogNorm(vmin=1e-3, vmax=None), cmap='nipy_spectral')
        
        suffix1 = '_log'
    else:
        im1 = ax1.pcolormesh(xarr, ftime, abs(By_pos), cmap='nipy_spectral')
        
        im2 = ax2.pcolormesh(xarr, ftime, abs(By_neg), cmap='nipy_spectral')
        suffix1 = ''
    
    plt.colorbar(im1, cax=cax1).set_label('Flux\n$cm^{-2}s^{-1}sr^{-1}keV^{-1}$')
    plt.colorbar(im2, cax=cax2).set_label('Flux\n$cm^{-2}s^{-1}sr^{-1}keV^{-1}$')
    
    ax1.set_title('By: +ve Helicity')
    ax2.set_title('By: -ve Helicity')
    ax1.set_title(title)

    
    ## Z-AXIS FIELD
    fig2 = plt.figure(constrained_layout=True, figsize=(18, 10))
    gs   = GridSpec(2, 2, figure=fig2, hspace=0, width_ratios=[1.0, 0.05])
    
    ax3  = fig2.add_subplot(gs[0, 0])
    ax4  = fig2.add_subplot(gs[1, 0])
    cax3 = fig2.add_subplot(gs[0, 1])
    cax4 = fig2.add_subplot(gs[1, 1])

    if log == True:
        im3 = ax3.pcolormesh(xarr, ftime, abs(Bz_pos),
                       norm=colors.LogNorm(vmin=1e-3, vmax=None), cmap='nipy_spectral')
        
        im4 = ax4.pcolormesh(xarr, ftime, abs(Bz_neg),
                       norm=colors.LogNorm(vmin=1e-3, vmax=None), cmap='nipy_spectral')
        
        suffix2 = '_log'
    else:
        im3 = ax3.pcolormesh(xarr, ftime, abs(Bz_pos), cmap='nipy_spectral')
        
        im4 = ax4.pcolormesh(xarr, ftime, abs(Bz_neg), cmap='nipy_spectral')
        suffix2 = ''

    plt.colorbar(im3, cax=cax3).set_label('Flux\n$cm^{-2}s^{-1}sr^{-1}keV^{-1}$')
    plt.colorbar(im4, cax=cax4).set_label('Flux\n$cm^{-2}s^{-1}sr^{-1}keV^{-1}$')

    ax3.set_title('Bz: +ve Helicity')
    ax4.set_title('Bz: -ve Helicity')
    plt.suptitle(title)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(-cf.xmax, cf.xmax)
        ax.set_xlabel('x (m)')
        
    ax1.set_ylabel('Time slice, dt = {:g}s'.format(float('{:.{p}g}'.format(cf.dt_field, p=sig_fig))))
    ax3.set_ylabel('Time slice, dt = {:g}s'.format(float('{:.{p}g}'.format(cf.dt_field, p=sig_fig))))
    
    if save == True:
        fig1.subplots_adjust(bottom=0.07, top=0.96, left=0.04)
        fig1.subplots_adjust(wspace=0.05)
        ax2.set_yticklabels([])
        fig1.savefig(cf.anal_dir + 'by_helicity_{}.png'.format(suffix1), facecolor=fig1.get_facecolor(), edgecolor='none')
        
        fig2.subplots_adjust(bottom=0.07, top=0.96, left=0.04)
        fig2.subplots_adjust(wspace=0.05)
        ax4.set_yticklabels([])
        fig2.savefig(cf.anal_dir + 'bz_helicity_{}.png'.format(suffix2), facecolor=fig2.get_facecolor(), edgecolor='none')

        plt.close('all')
    return


def find_peaks(dat, x_thres=5, y_thres=1e-5):
    deriv = np.zeros(dat.shape[0])
    
    for ii in range(1, dat.shape[0] - 1):
        deriv[ii] = dat[ii + 1] - dat[ii - 1]
        
    plt.figure()
    plt.plot(dat, marker='o', c='b')
    plt.plot(deriv, marker='x', c='r')
    plt.axhline(0, c='k')
    plt.show()
    return


def animate_fields():
    '''
    Animates the fields in a dynamic plot. 
    
    Need to find some way to speed this up - how to update plot data without
    removing axes labels and ticks?
    '''
    Bx_raw         = 1e9 * (cf.get_array('Bx')  - cf.B0)
    By_raw         = 1e9 *  cf.get_array('By')
    Bz_raw         = 1e9 *  cf.get_array('Bz')
    
    Ex_raw         = cf.get_array('Ex') * 1e3
    Ey_raw         = cf.get_array('Ey') * 1e3
    Ez_raw         = cf.get_array('Ez') * 1e3
    
    max_B = max(abs(By_raw).max(), abs(Bz_raw).max())
    max_E = max(abs(Ey_raw).max(), abs(Ez_raw).max(), abs(Ex_raw).max())
    
    xticks = np.arange(0, cf.NX, cf.NX/4)
    lpad   = 20
    sf     = 3
    
    fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(nrows=3, ncols=2)
    fig.subplots_adjust(wspace=0, hspace=0)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    for ii in range(Bx_raw.shape[0]):
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.clear()
            ax.set_xticks(xticks)
            
        ax1.plot(Bz_raw[ii])
        ax3.plot(By_raw[ii])
        ax5.plot(Bx_raw[ii])
        
        ax1.set_ylabel(r'$\delta B_z$ (nT)', rotation=0, labelpad=lpad)
        ax3.set_ylabel(r'$\delta B_y$ (nT)', rotation=0, labelpad=lpad)
        ax5.set_ylabel(r'$\delta B_x$ (nT)', rotation=0, labelpad=lpad)
        
        ax2.plot(Ez_raw[ii])
        ax4.plot(Ey_raw[ii])
        ax6.plot(Ex_raw[ii])
        
        ax2.set_ylabel(r'$\delta E_z$ (mV/m)', rotation=0, labelpad=1.5*lpad)
        ax4.set_ylabel(r'$\delta E_y$ (mV/m)', rotation=0, labelpad=1.5*lpad)
        ax6.set_ylabel(r'$\delta E_x$ (mV/m)', rotation=0, labelpad=1.5*lpad)
        
        # B-field side
        for ax in [ax1, ax3, ax5]:
            ax.set_xlim(0, cf.NX-1)
            ax.set_ylim(-max_B, max_B)
        
        # E-field side
        for ax in [ax2, ax4, ax6]:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
            ax.set_xlim(0, cf.NX-1)
            ax.set_ylim(-max_E, max_E)
            
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xticklabels([])
            
        for ax in [ax5, ax6]:
            ax.set_label('Cell Number')
        
        ax5.text(0.0, -0.15, 'it = {}, t = {:.{p}g}s'.format(ii, cf.time_seconds_field[ii], p=sf), transform=ax5.transAxes)
        
        plt.suptitle('Perturbed fields')
        
        plt.pause(0.05)
    return


def plot_spatially_averaged_fields(save=True, tmax=None):
    '''
    Recreates Omidi et al. (2010) Figure 2
    
    Field arrays are shaped like (time, space)
    '''
    Bx_raw = 1e9 * (cf.get_array('Bx')  - cf.B0)
    By_raw = 1e9 *  cf.get_array('By')
    Bz_raw = 1e9 *  cf.get_array('Bz')
      
    lpad   = 20
    
    plt.ioff()
    fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(figsize=(18, 10), nrows=3, ncols=2)
    fig.subplots_adjust(wspace=0, hspace=0)
    
    ax1.plot(cf.time_seconds_field, abs(Bz_raw).mean(axis=1))
    ax3.plot(cf.time_seconds_field, abs(By_raw).mean(axis=1))
    ax5.plot(cf.time_seconds_field, abs(Bx_raw).mean(axis=1))
    
    ax2.plot(cf.time_seconds_field, abs(Bz_raw).mean(axis=1))
    ax4.plot(cf.time_seconds_field, abs(By_raw).mean(axis=1))
    ax6.plot(cf.time_seconds_field, abs(Bx_raw).mean(axis=1))
    
    ax1.set_ylabel(r'$\overline{|\delta B_z|}$ (nT)', rotation=0, labelpad=lpad)
    ax3.set_ylabel(r'$\overline{|\delta B_y|}$ (nT)', rotation=0, labelpad=lpad)
    ax5.set_ylabel(r'$\overline{|\delta B_x|}$ (nT)', rotation=0, labelpad=lpad)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticklabels([])
    
    for ax in [ax2, ax4, ax6]:
        ax.set_xlim(0, cf.time_seconds_field[-1])
        ax.set_ylim(0, None)
        ax.set_yticklabels([])
        
    for ax in [ax1, ax3, ax5]:
        if tmax is None:
            ax.set_xlim(0, cf.time_seconds_field[-1]/5)
        else:
            ax.set_xlim(0, tmax)
            
        ax.set_ylim(0, None)
                    
    for ax in [ax5, ax6]:
        ax.set_xlabel(r'Time (s)')
      
    ax1.set_title('Spatially averaged fields'.format(cf.method_type))
    
    if save == True:
        fig.savefig(cf.anal_dir + 'sp_av_fields.png', facecolor=fig.get_facecolor(), edgecolor='none')
        print('Spatially averaged B-field plot saved for run {}'.format(run_num))
    plt.close('all')
    return


def single_point_helicity_timeseries(cells=None, overwrite=False, save=True):
    '''
    Plot timeseries for raw, +ve, -ve helicities at single point
    
    Maybe do phases here too? (Although is that trivial for the helical components
    since they're separated based on a phase relationship between By,z ?)
    '''
    if cells is None:
        cells = np.arange(cf.NX)
    
    ts_folder = cf.anal_dir + '//single_point_helicity//'
    
    if os.path.exists(ts_folder) == False:
        os.makedirs(ts_folder)
    
    By_raw         = cf.get_array('By')
    Bz_raw         = cf.get_array('Bz')
    Bt_pos, Bt_neg = bk.get_helical_components(overwrite)

    By_pos = Bt_pos.real
    By_neg = Bt_neg.real
    Bz_pos = Bt_pos.imag
    Bz_neg = Bt_neg.imag
    
    plt.ioff()
    for raw, pos, neg, component in zip([By_raw, Bz_raw], [By_pos, Bz_pos], [By_neg, Bz_neg], ['y', 'z']):
        for x_idx in cells:
            fig = plt.figure(figsize=(18, 10))
            ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
            ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
            
            ax1.plot(cf.time_seconds_field, 1e9*raw[:, x_idx], label='Raw B{}'.format(component), c='blue')
            ax2.plot(cf.time_seconds_field, 1e9*pos[:, x_idx], label='B{}+'.format(component), c='green')
            ax2.plot(cf.time_seconds_field, 1e9*neg[:, x_idx], label='B{}-'.format(component), c='orange')
            
            ax1.set_title('Time-series at cell {}'.format(x_idx))
            ax2.set_xlabel('Time (s)')
            
            for ax in [ax1, ax2]:
                ax.set_ylabel('B{} (nT)'.format(component))
                ax.set_xlim(0, cf.time_seconds_field[-1])
                ax.legend()
            
            fig.tight_layout()
            fig.subplots_adjust(hspace=0)
            
            ax1.set_xticklabels([])
            
            if save==True:
                fig.savefig(ts_folder + 'single_point_field_B{}_{}.png'.format(component, x_idx), edgecolor='none')
            plt.close('all')
    return


def single_point_field_timeseries(cells=None, overwrite=False, save=True, tmax=None):
    '''
    Plot timeseries for raw fields at specified cells
    
    maxtime=time in seconds for endpoint (defaults to total runtime)
    '''
    print('Plotting single-point fields...')
    if cells is None:
        cells = np.arange(cf.NX)
    
    ts_folder_B = cf.anal_dir + '//single_point_fields//magnetic//'
    ts_folder_E = cf.anal_dir + '//single_point_fields//electric//'
    
    if os.path.exists(ts_folder_B) == False:
        os.makedirs(ts_folder_B)
        
    if os.path.exists(ts_folder_E) == False:
        os.makedirs(ts_folder_E)
    
    bx, by, bz, ex, ey, ez, vex, vey, vez, te, jx, jy, jz, qdens, fsim_time, damping_array = cf.get_array(get_all=True)
    
    plt.ioff()
    for x_idx in cells:
        print('Cell {}...'.format(x_idx))
        figB  = plt.figure(figsize=(18, 10))
        
        ######################
        ### MAGNETIC FIELD ### Could loop this but I'm lazy
        ######################
        figB  = plt.figure(figsize=(18, 10))
        
        ## FIELDS: One period ##
        axbx = plt.subplot2grid((3, 2), (0, 0))
        axby = plt.subplot2grid((3, 2), (1, 0))
        axbz = plt.subplot2grid((3, 2), (2, 0))
        
        axbx.plot(cf.time_seconds_field, 1e9*bx[:, x_idx])
        axbx.set_ylabel('$B_x (nT)$')
        
        axby.plot(cf.time_seconds_field, 1e9*by[:, x_idx])
        axby.set_ylabel('$B_y (nT)$')
        
        axbz.plot(cf.time_seconds_field, 1e9*bz[:, x_idx])
        axbz.set_ylabel('$B_z (nT)$')
        axbz.set_xlabel('Time (s)')
        
        ## FIELDS: Full time ##
        axbx_full = plt.subplot2grid((3, 2), (0, 1))
        axby_full = plt.subplot2grid((3, 2), (1, 1))
        axbz_full = plt.subplot2grid((3, 2), (2, 1))
        
        axbx_full.set_title('B-field at cell {}: Total time'.format(x_idx))
        axbx_full.plot(cf.time_seconds_field, 1e9*bx[:, x_idx])
        axby_full.plot(cf.time_seconds_field, 1e9*by[:, x_idx])
        axbz_full.plot(cf.time_seconds_field, 1e9*bz[:, x_idx])
        axbz_full.set_xlabel('Time (s)')
        
        if tmax is None:
            # Set it at 20% full runtime, just to get a bit better resolution
            tmax = cf.time_seconds_field[-1] / 5
            axbx.set_title('B-field at cell {}: 1/5 total time'.format(x_idx))
        else:
            axbx.set_title('B-field at cell {}: One period'.format(x_idx))
            
        for ax in [axbx, axby, axbz]:
            ax.set_xlim(0, tmax)
            
        for ax in [axbx_full, axby_full, axbz_full]:
            ax.set_xlim(0, cf.time_seconds_field[-1])
            ax.set_yticklabels([])
            
        for ax in [axbx, axby, axbx_full, axby_full]:
            ax.set_xticklabels([])
            
        axbx.set_ylim(axbx_full.get_ylim())
        axby.set_ylim(axby_full.get_ylim())
        axbz.set_ylim(axbz_full.get_ylim())
        
        figB.tight_layout()
        figB.subplots_adjust(hspace=0, wspace=0.02)
        
        if save==True:
            figB.savefig(ts_folder_B + 'single_point_Bfield_{}.png'.format(x_idx), edgecolor='none')
   
    
        ######################
        ### ELECTRIC FIELD ###
        ######################
        figE  = plt.figure(figsize=(18, 10))
        ## FIELDS: One period ##
        axex = plt.subplot2grid((3, 2), (0, 0))
        axey = plt.subplot2grid((3, 2), (1, 0))
        axez = plt.subplot2grid((3, 2), (2, 0))
        
        axex.plot(cf.time_seconds_field, 1e3*ex[:, x_idx])
        axex.set_ylabel('$E_x (mV/m)$')
        
        axey.plot(cf.time_seconds_field, 1e3*ey[:, x_idx])
        axey.set_ylabel('$E_y (mV/m)$')
        
        axez.plot(cf.time_seconds_field, 1e3*ez[:, x_idx])
        axez.set_ylabel('$E_z (mV/m)$')
        axez.set_xlabel('Time (s)')
        
        ## FIELDS: Full time ##
        axex_full = plt.subplot2grid((3, 2), (0, 1))
        axey_full = plt.subplot2grid((3, 2), (1, 1))
        axez_full = plt.subplot2grid((3, 2), (2, 1))
        
        axex_full.set_title('E-field at cell {}: Total time'.format(x_idx))
        axex_full.plot(cf.time_seconds_field, 1e3*ex[:, x_idx])
        axey_full.plot(cf.time_seconds_field, 1e3*ey[:, x_idx])
        axez_full.plot(cf.time_seconds_field, 1e3*ez[:, x_idx])
        axez_full.set_xlabel('Time (s)')
        
        if tmax is None:
            # Set it at 20% full runtime, just to get a bit better resolution
            tmax = cf.time_seconds_field[-1] / 5
            axbx.set_title('E-field at cell {}: 1/5 total time'.format(x_idx))
        else:
            axbx.set_title('E-field at cell {}: One period'.format(x_idx))
            
        for ax in [axex, axey, axez]:
            ax.set_xlim(0, tmax)
            
        for ax in [axex_full, axey_full, axez_full]:
            ax.set_xlim(0, cf.time_seconds_field[-1])
            ax.set_yticklabels([])
            
        for ax in [axex, axey, axex_full, axey_full]:
            ax.set_xticklabels([])
            
        axex.set_ylim(axex_full.get_ylim())
        axey.set_ylim(axey_full.get_ylim())
        axez.set_ylim(axez_full.get_ylim())
        
        figE.tight_layout()
        figE.subplots_adjust(hspace=0, wspace=0.02)
        
        if save==True:
            figE.savefig(ts_folder_E + 'single_point_Efield_{}.png'.format(x_idx), edgecolor='none')
        plt.close('all')
    return


def single_point_both_fields_AGU(cell=None, save=True):
    '''
    Plot timeseries for raw fields at specified cells
    
    maxtime=time in seconds for endpoint (defaults to total runtime)
    '''
    print('Plotting single-point fields...')
    
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    
    fontsize = 18
    
    if cell is None:
        cells = np.arange(cf.NX)
    
    ts_folder_BE = cf.anal_dir + '//single_point_fields//both//'
    
    if os.path.exists(ts_folder_BE) == False:
        os.makedirs(ts_folder_BE)

    bx, by, bz, ex, ey, ez, vex, vey, vez, te, jx, jy, jz, qdens, fsim_time, damping_array = cf.get_array(get_all=True)
    
    plt.ioff()
    for x_idx in cells:
        print('Cell {}...'.format(x_idx))
        fig, axes  = plt.subplots(3, figsize=(12, 8))
        
        axes[0].set_title('Fields at cell {}'.format(x_idx), fontsize=fontsize+4)
        
        axes[0].plot(cf.time_seconds_field, 1e9*bx[:, x_idx])
        axes[1].plot(cf.time_seconds_field, 1e3*ez[:, x_idx])
        axes[2].plot(cf.time_seconds_field, 1e9*by[:, x_idx])
        
        for ax in axes:
            ax.set_xlim(0, cf.time_seconds_field[-1])
            
        for ax in axes[:-1]:
            ax.set_xticklabels([])
        
        lpad=24
        
        axes[0].set_ylabel('$B_x$\n(nT)',   fontsize=fontsize, rotation=0, labelpad=lpad)
        axes[1].set_ylabel('$E_z$\n(mV/m)', fontsize=fontsize, rotation=0, labelpad=lpad)
        axes[2].set_ylabel('$B_y$\n(nT)',   fontsize=fontsize, rotation=0, labelpad=lpad)
        axes[-1].set_xlabel('Time (s)',     fontsize=fontsize, rotation=0)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        
        if save==True:
            fig.savefig(ts_folder_BE + 'single_point_BEfields_{}.png'.format(x_idx), edgecolor='none', bbox_inches='tight')
            plt.close('all')
    return


def analyse_helicity(overwrite=False, save=True):
    By_raw         = cf.get_array('By')
    Bz_raw         = cf.get_array('Bz')
    Bt_pos, Bt_neg = bk.get_helical_components(overwrite)

    By_pos = Bt_pos.real
    By_neg = Bt_neg.real
    Bz_pos = Bt_pos.imag
    Bz_neg = Bt_neg.imag
    
    t_idx1 = 200
    t_idx2 = 205    
    
    if False:
        '''
        Check that helicity preserves transverse amplitude on transformation : Affirmative
        '''
        hel_tot = np.sqrt(np.square(By_pos + By_neg) + np.square(Bz_pos + Bz_neg))
        raw_tot = np.sqrt(np.square(By_raw) + np.square(Bz_raw))
    
        plt.figure()
        plt.plot(raw_tot[t_idx1, :], label='raw B')
        plt.plot(hel_tot[t_idx1, :], label='helicty B')
        plt.legend()
    
    if False:
        '''
        Peak finder I was going to use for velocity
        '''
        peaks1 = bk.basic_S(By_pos[t_idx1, :], k=100)
        peaks2 = bk.basic_S(By_pos[t_idx2, :], k=100)
        
        plt.plot(1e9*By_pos[t_idx1, :])
        plt.scatter(peaks1, 1e9*By_pos[t_idx1, peaks1])
        
        plt.plot(1e9*By_pos[t_idx2, :])
        plt.scatter(peaks2, 1e9*By_pos[t_idx2, peaks2])
    return


def summary_plots(save=True, histogram=True):
    '''
    Plot summary plot of raw values for each particle timestep
    Field values are interpolated to this point
    
    To Do: Find some nice way to include species energies instead of betas
    '''  
    np.set_printoptions(suppress=True)
    
    if histogram == True:
        path = cf.anal_dir + '/summary_plots_histogram/'
    else:
        path = cf.anal_dir + '/summary_plots/'
        
    if os.path.exists(path) == False:                                   # Create data directory
        os.makedirs(path)
    
    num_particle_steps = len(os.listdir(cf.particle_dir))
    
    plt.ioff()
    ptime_sec, pbx, pby, pbz, pex, pey, pez, pvex, pvey,\
    pvez, pte, pjx, pjy, pjz, pqdens = cf.interpolate_fields_to_particle_time(num_particle_steps)

    time_seconds_particle    = ptime_sec
    time_gperiods_particle   = ptime_sec / cf.gyperiod 
    time_radperiods_particle = ptime_sec / cf.gyfreq
    
    # Normalized change density
    qdens_norm = pqdens / (cf.density*cf.charge).sum()     
    
    B_lim   = np.array([-1.0*pby.min(), pby.max(), -1.0*pbz.min(), pbz.max()]).max() * 1e9
    vel_lim = 20
    E_lim   = np.array([-1.0*pex.min(), pex.max(), -1.0*pey.min(), pey.max(), -1.0*pez.min(), pez.max()]).max() * 1e3
    den_max = qdens_norm.max()
    den_min = 2.0 - den_max

    # Set lims to ceiling values (Nope: Field limits are still doing weird shit. It's alright.)
    B_lim = np.ceil(B_lim)
    E_lim = np.ceil(E_lim)
    
    for ii in range(num_particle_steps):
        filename = 'summ%05d.png' % ii
        fullpath = path + filename
        
        if os.path.exists(fullpath):
            print('Summary plot already present for timestep [{}]{}'.format(run_num, ii))
            continue
        
        print('Creating summary plot for particle timestep [{}]{}'.format(run_num, ii))
        fig_size = 4, 7                                                             # Set figure grid dimensions
        fig = plt.figure(figsize=(20,10))                                           # Initialize Figure Space
        fig.patch.set_facecolor('w')   
        xp, vp, idx, psim_time = cf.load_particles(ii)
        
        pos       = xp  
        vel       = vp / cf.va 

        # Count particles lost to the simulation
        N_lost = np.zeros(cf.Nj, dtype=int)
        if idx is not None:
            Nl_idx  = idx[idx < 0]  # Collect indices of those < 0
            Nl_idx += 128           # Cast from negative to positive indexes ("reactivate" particles)
            for jj in range(cf.Nj):
                N_lost[jj] = Nl_idx[Nl_idx == jj].shape[0]
        else:
            N_lost = None
        ax_vx   = plt.subplot2grid(fig_size, (0, 0), rowspan=2, colspan=3)
        ax_vy   = plt.subplot2grid(fig_size, (2, 0), rowspan=2, colspan=3)
        
        if histogram == True:
            
            vel_tr = np.sqrt(vel[1] ** 2 + vel[2] ** 2)
            
            for jj in range(cf.Nj):
                num_bins = cf.nsp_ppc[jj] // 5
                
                xs, BinEdgesx = np.histogram(vel[0, cf.idx_start[jj]: cf.idx_end[jj]], bins=num_bins)
                bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
                ax_vx.plot(bx, xs, '-', c=cf.temp_color[jj], drawstyle='steps', label=cf.species_lbl[jj])
                
                ys, BinEdgesy = np.histogram(vel_tr[cf.idx_start[jj]: cf.idx_end[jj]], bins=num_bins)
                by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
                ax_vy.plot(by, ys, '-', c=cf.temp_color[jj], drawstyle='steps', label=cf.species_lbl[jj])
                
                ax_vx.set_ylabel(r'$n_{v_\parallel}$')
                ax_vx.set_ylabel(r'$n_{v_\perp}$')
                
                ax_vx.set_title('Velocity distribution of each species in simulation domain')
                ax_vy.set_xlabel(r'$v / v_A$')
                
                ax_vx.set_xlim(-vel_lim, vel_lim)
                ax_vy.set_xlim(0, np.sqrt(2)*vel_lim)
                
                for ax, comp in zip([ax_vx, ax_vy], ['v_\parallel', 'v_\perp']):
                    ax.set_ylim(0, int(cf.N / cf.NX) * 2.0)
                    ax.legend(loc='upper right')
                    ax.set_ylabel('$n_{%s}$'%comp)
        else:
        
            for jj in reversed(range(cf.Nj)):
                ax_vx.scatter(pos[0, cf.idx_start[jj]: cf.idx_end[jj]], vel[0, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj], lw=0, label=cf.species_lbl[jj])
                ax_vy.scatter(pos[0, cf.idx_start[jj]: cf.idx_end[jj]], vel[1, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj], lw=0)
        
            ax_vx.legend()
            ax_vx.set_title(r'Particle velocities vs. Position (x)')
            ax_vy.set_xlabel(r'Cell', labelpad=10)
            ax_vx.set_ylabel(r'$\frac{v_x}{v_A}$', rotation=0)
            ax_vy.set_ylabel(r'$\frac{v_y}{v_A}$', rotation=0)
            
            plt.setp(ax_vx.get_xticklabels(), visible=False)
            ax_vx.set_yticks(ax_vx.get_yticks()[1:])
        
            for ax in [ax_vy, ax_vx]:
                ax.set_xlim(-cf.xmax, cf.xmax)
                ax.set_ylim(-vel_lim, vel_lim)
    
        
        ## DENSITY ##
        B_nodes  = (np.arange(cf.NC + 1) - cf.NC // 2)            # B grid points position in space
        E_nodes  = (np.arange(cf.NC)     - cf.NC // 2 + 0.5)      # E grid points position in space

        ax_den  = plt.subplot2grid(fig_size, (0, 3), colspan=3)
        ax_den.plot(E_nodes, qdens_norm[ii], color='green')
                
        ax_den.set_title('Charge Density and Fields')
        ax_den.set_ylabel(r'$\frac{\rho_c}{\rho_{c0}}$', fontsize=14, rotation=0, labelpad=20)
        

        ax_Ex   = plt.subplot2grid(fig_size, (1, 3), colspan=3, sharex=ax_den)
        ax_Ex.plot(E_nodes, pex[ii]*1e3, color='red',   label=r'$E_x$')
        ax_Ex.plot(E_nodes, pey[ii]*1e3, color='cyan',  label=r'$E_y$')
        ax_Ex.plot(E_nodes, pez[ii]*1e3, color='black', label=r'$E_z$')
        ax_Ex.set_ylabel(r'$E (mV/m)$', labelpad=25, rotation=0, fontsize=14)
        
        ax_Ex.legend(loc=4, ncol=3)
        
        ax_By  = plt.subplot2grid(fig_size, (2, 3), colspan=3, sharex=ax_den)
        ax_B   = plt.subplot2grid(fig_size, (3, 3), colspan=3, sharex=ax_den)
        mag_B  = np.sqrt(pby[ii] ** 2 + pbz[ii] ** 2)
        
        ax_Bx = ax_B.twinx()
        ax_Bx.plot(B_nodes, pbx[ii]*1e9, color='k', label=r'$B_x$', ls=':', alpha=0.6) 
        ax_Bx.set_ylim(cf.B_eq*1e9, cf.Bc.max()*1e9)
        ax_Bx.set_ylabel(r'$B_{0x} (nT)$', rotation=0, labelpad=30, fontsize=14)
        
        ax_B.plot( B_nodes, mag_B*1e9, color='g')
        ax_By.plot(B_nodes, pby[ii]*1e9, color='g',   label=r'$B_y$') 
        ax_By.plot(B_nodes, pbz[ii]*1e9, color='b',   label=r'$B_z$') 
        ax_By.legend(loc=4, ncol=2)
        
        ax_B.set_ylabel( r'$B_\perp (nT)$', rotation=0, labelpad=30, fontsize=14)
        ax_By.set_ylabel(r'$B_{y,z} (nT)$', rotation=0, labelpad=20, fontsize=14)
        ax_B.set_xlabel('Cell Number')
        
        # SET FIELD RANGES #
        ax_den.set_ylim(den_min, den_max)
        ax_Ex.set_ylim(-E_lim, E_lim)
        ax_By.set_ylim(-B_lim, B_lim)
        ax_B.set_ylim(0, B_lim)
        
        for ax in [ax_den, ax_Ex, ax_By]:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_yticks(ax.get_yticks()[1:])
            
        for ax in [ax_den, ax_Ex, ax_By, ax_B]:
            ax.set_xlim(B_nodes[0], B_nodes[-1])
            ax.axvline(-cf.NX//2, c='k', ls=':', alpha=0.5)
            ax.axvline( cf.NX//2, c='k', ls=':', alpha=0.5)
            ax.grid()
                
        plt.tight_layout(pad=1.0, w_pad=1.8)
        fig.subplots_adjust(hspace=0.125)
        
        ###################
        ### FIGURE TEXT ###
        ###################
        anisotropy = (cf.Tper / cf.Tpar - 1).round(1)
        beta_per   = (2*(4e-7*np.pi)*(1.381e-23)*cf.Tper*cf.ne / (cf.B_eq**2)).round(1)
        beta_e     = round((2*(4e-7*np.pi)*(1.381e-23)*cf.Te0*cf.ne  / (cf.B_eq**2)), 2)
        rdens      = (cf.density / cf.ne).round(2)

        try:
            vdrift     = (cf.velocity / cf.va).round(1)
        except:
            vdrift     = (cf.drift_v / cf.va).round(1)
        
        if cf.ie == 0:
            estring = 'Isothermal electrons'
        elif cf.ie == 1:
            estring = 'Adiabatic electrons'
        else:
            'Electron relation unknown'
                    
        top  = 0.95
        gap  = 0.025
        fontsize = 12
        plt.figtext(0.855, top        , 'Simulation Parameters', fontsize=fontsize, family='monospace', fontweight='bold')
        plt.figtext(0.855, top - 1*gap, '{}[{}]'.format(series, run_num), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 2*gap, '{} cells'.format(cf.NX), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 3*gap, '{} particles'.format(cf.N_species.sum()), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 4*gap, '{}'.format(estring), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 5*gap, '', fontsize=fontsize, family='monospace')
        
        plt.figtext(0.855, top - 6*gap, 'B_eq      : {:.1f}nT'.format(cf.B_eq*1e9  ), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 7*gap, 'B_xmax    : {:.1f}nT'.format(cf.B_xmax*1e9), fontsize=fontsize, family='monospace')
        
        plt.figtext(0.855, top - 8*gap,  'ne        : {}cc'.format(cf.ne/1e6), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 9*gap,  'N_species : {}'.format(cf.N_species), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 10*gap, 'N_lost    : {}'.format(N_lost), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 11*gap, '', fontsize=fontsize, family='monospace')
        
        plt.figtext(0.855, top - 12*gap, r'$\beta_e$      : %.2f' % beta_e, fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 13*gap, 'dx      : {}km'.format(round(cf.dx/1e3, 2)), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 14*gap, 'L       : {}'.format(cf.L), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 15*gap, 'MLAT_max: $\pm$%.1f$^{\circ}$' % (cf.theta_xmax * 180. / np.pi), fontsize=fontsize, family='monospace')

        plt.figtext(0.855, top - 16*gap, '', fontsize=fontsize, family='monospace')
        
        ptop  = top - 17*gap
        pside = 0.855
        plt.figtext(pside, ptop, 'Particle Parameters', fontsize=fontsize, family='monospace', fontweight='bold')
        plt.figtext(pside, ptop - gap, ' SPECIES  ANI  XBET    VDR  RDNS', fontsize=fontsize-2, family='monospace')
        for jj in range(cf.Nj):
            plt.figtext(pside       , ptop - (jj + 2)*gap, '{:>10}  {:>3}  {:>5}  {:>4}  {:<5}'.format(
                    cf.species_lbl[jj], anisotropy[jj], beta_per[jj], vdrift[jj], rdens[jj]),
                    fontsize=fontsize-2, family='monospace')
 
        time_top = 0.1
        plt.figtext(0.88, time_top - 0*gap, 't_seconds   : {:>10}'.format(round(time_seconds_particle[ii], 3))   , fontsize=fontsize, family='monospace')
        plt.figtext(0.88, time_top - 1*gap, 't_gperiod   : {:>10}'.format(round(time_gperiods_particle[ii], 3))  , fontsize=fontsize, family='monospace')
        plt.figtext(0.88, time_top - 2*gap, 't_radperiod : {:>10}'.format(round(time_radperiods_particle[ii], 3)), fontsize=fontsize, family='monospace')
    
        if save == True:
            plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close('all')
    return


def standard_analysis_package():
    '''
    Need a high-pass option for the wk? Or will it do all of it?
    It should do all of it (show the Pc4 branch and the Pc1 branch)
    Since they're both in the magnetic field
    
    Actually, Pc5 is only in bx... affects By/Bz? Echo in other components?
    '''
    disp_folder = 'dispersion_plots/'
        
    if os.path.exists(cf.anal_dir + disp_folder) == False:
        os.makedirs(cf.anal_dir + disp_folder)
        
    for comp in ['By', 'Bz', 'Ex', 'Ey', 'Ez']:
        print('2D summary for {}'.format(comp))
        plot_tx(component=comp, saveas=disp_folder + 'tx_plot', save=True)
        try:
            plot_tx(component=comp, saveas=disp_folder + 'tx_plot', save=True, log=True)
        except:
            pass
        try:
            plot_wx(component=comp, saveas=disp_folder + 'wx_plot', save=True, linear_overlay=False,     pcyc_mult=1.1)
        except:
            pass
        plot_wk(component=comp, saveas=disp_folder + 'wk_plot', save=True, dispersion_overlay=False, pcyc_mult=1.1)
        plot_kt(component=comp, saveas=disp_folder + 'kt_plot', save=True)
        
    plot_spatial_poynting(save=True, log=True)
    plot_helical_waterfall(title='{}: Run {}'.format(series, run_num), save=True)
    
    if True:
        pass
        
    if False:
        check_fields()
        plot_energies(normalize=True, save=True)
        plot_ion_energy_components(save=True, tmax=1./cf.HM_frequency)
        single_point_helicity_timeseries()
        plot_spatially_averaged_fields()
        single_point_field_timeseries(tmax=1./cf.HM_frequency)
    return


def do_all_dynamic_spectra(ymax=None):
    
    for component in ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']:
        for cell in np.arange(cf.NX):
            plot_dynamic_spectra(component=component, ymax=ymax, save=True, cell=cell)
    return


def plot_damping_array(save=True):
    save_folder = cf.anal_dir + '//ABC_check//'
    
    ftime, by            = cf.get_array('By')
    ftime, bz            = cf.get_array('Bz')
    ftime, damping_array = cf.get_array('damping_array')
    xarr                 = cf.B_nodes / cf.dx
    
    plt.ioff()
    for ii in range(by.shape[0]):
        print('Plotting ABCs {} of {}'.format(ii, by.shape[0]))
        fig, axes = plt.subplots(3, sharex=True, figsize=(16,10))
        
        axes[0].plot(xarr, by[ii], c='b')
        axes[1].plot(xarr, bz[ii], c='r')
        axes[2].plot(xarr, damping_array[ii], c='g')
    
        axes[0].set_title('Absorbing Boundary Conditions checkplot')
        axes[0].set_ylabel('$B_y$' , rotation=0, labelpad=20)
        axes[1].set_ylabel('$B_z$' , rotation=0, labelpad=20)
        axes[2].set_ylabel('$r(x)$', rotation=0, labelpad=20)
        
        axes[0].set_ylim(-cf.B_xmax, cf.B_xmax)
        axes[1].set_ylim(-cf.B_xmax, cf.B_xmax)
    
        for ax in axes:
            ax.set_xlim(xarr[0], xarr[-1])
            ax.axvline(-cf.xmax/cf.dx, ls=':', alpha=0.75, c='k')
            ax.axvline( cf.xmax/cf.dx, ls=':', alpha=0.75, c='k')
            
        if save==True:
            if os.path.exists(save_folder) == False:
                os.makedirs(save_folder)
            
            fig.savefig(save_folder + 'Bfield_ABC_check_{:05}.png'.format(ii), edgecolor='none')
            plt.close('all')
    return


def check_fields(save=True):
    '''
    Plot summary plot of raw values for each particle timestep
    Field values are interpolated to this point
    '''    
    path = cf.anal_dir + '/field_plots_all/'
        
    if os.path.exists(path) == False:                                   # Create data directory
        os.makedirs(path)
        
    plt.ioff()
    ftime, bx, by, bz, ex, ey, ez, vex, vey, vez,\
    te, jx, jy, jz, qdens, fsim_time, rD = cf.get_array(get_all=True)

    # Convert to plot units
    bx *= 1e9; by *= 1e9; bz *= 1e9
    ex *= 1e3; ey *= 1e3; ez *= 1e3
    qdens *= 1e-6/q
    te /= 11603.
    
    for ii in range(bx.shape[0]):
        filename = 'summ%05d.png' % ii
        fullpath = path + filename
        fontsize = 14
        
        if os.path.exists(fullpath):
            print('Summary plot already present for timestep [{}]{}'.format(run_num, ii))
            continue
                
        fsize = 12; lpad = 20
        
        print('Creating summary field plots [{}]{}'.format(run_num, ii))
        fig, axes = plt.subplots(5, ncols=3, figsize=(20,10), sharex=True)
        fig.patch.set_facecolor('w')   

        axes[0, 0].set_title('Field outputs: {}[{}]'.format(series, run_num), fontsize=fontsize+4, family='monospace')

        # Wave Fields (Plots, Labels, Lims)
        axes[0, 0].plot(cf.B_nodes / cf.dx, rD[ii], color='k', label=r'$r_D(x)$') 
        axes[1, 0].plot(cf.B_nodes / cf.dx, by[ii], color='b', label=r'$B_y$') 
        axes[2, 0].plot(cf.B_nodes / cf.dx, bz[ii], color='g', label=r'$B_z$')
        axes[3, 0].plot(cf.E_nodes / cf.dx, ey[ii], color='b', label=r'$E_y$')
        axes[4, 0].plot(cf.E_nodes / cf.dx, ez[ii], color='g', label=r'$E_z$')
        
        axes[0, 0].set_ylabel('$r_D(x)$'     , rotation=0, labelpad=lpad, fontsize=fsize)
        axes[1, 0].set_ylabel('$B_y$\n(nT)'  , rotation=0, labelpad=lpad, fontsize=fsize)
        axes[2, 0].set_ylabel('$B_z$\n(nT)'  , rotation=0, labelpad=lpad, fontsize=fsize)
        axes[3, 0].set_ylabel('$E_y$\n(mV/m)', rotation=0, labelpad=lpad, fontsize=fsize)
        axes[4, 0].set_ylabel('$E_z$\n(mV/m)', rotation=0, labelpad=lpad, fontsize=fsize)
        
        axes[0, 0].set_ylim(rD.min(), rD.max())
        axes[1, 0].set_ylim(by.min(), by.max())
        axes[2, 0].set_ylim(bz.min(), bz.max())
        axes[3, 0].set_ylim(ey.min(), ey.max())
        axes[4, 0].set_ylim(ez.min(), ez.max())
        
        # Transverse Electric Field Variables (Plots, Labels, Lims)
        axes[0, 1].plot(cf.E_nodes / cf.dx, qdens[ii], color='k', label=r'$n_e$')
        axes[1, 1].plot(cf.E_nodes / cf.dx,   vey[ii], color='b', label=r'$V_{ey}$')
        axes[2, 1].plot(cf.E_nodes / cf.dx,   vez[ii], color='g', label=r'$V_{ez}$')
        axes[3, 1].plot(cf.E_nodes / cf.dx,    jy[ii], color='b', label=r'$J_{iy}$' )
        axes[4, 1].plot(cf.E_nodes / cf.dx,    jz[ii], color='g', label=r'$J_{iz}$' )
        
        axes[0, 1].set_ylabel('$n_e$\n$(cm^{-1})$', fontsize=fsize, rotation=0, labelpad=lpad)
        axes[1, 1].set_ylabel('$V_{ey}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
        axes[2, 1].set_ylabel('$V_{ez}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
        axes[3, 1].set_ylabel('$J_{iy}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
        axes[4, 1].set_ylabel('$J_{iz}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
        
        axes[0, 1].set_ylim(qdens.min(), qdens.max())
        axes[1, 1].set_ylim(vey.min(), vey.max())
        axes[2, 1].set_ylim(vez.min(), vez.max())
        axes[3, 1].set_ylim(jy.min() , jy.max())
        axes[4, 1].set_ylim(jz.min() , jz.max())
        
        # Parallel Variables (Plots, Labels, Lims)
        axes[0, 2].plot(cf.E_nodes / cf.dx,   te[ii], color='k', label=r'$T_e$')
        axes[1, 2].plot(cf.E_nodes / cf.dx,  vex[ii], color='r', label=r'$V_{ex}$')
        axes[2, 2].plot(cf.E_nodes / cf.dx,   jx[ii], color='r', label=r'$J_{ix}$' )
        axes[3, 2].plot(cf.E_nodes / cf.dx,   ex[ii], color='r', label=r'$E_x$')
        axes[4, 2].plot(cf.B_nodes / cf.dx,   bx[ii], color='r', label=r'$B_{0x}$')
        
        axes[0, 2].set_ylabel('$T_e$\n(eV)'     , fontsize=fsize, rotation=0, labelpad=lpad)
        axes[1, 2].set_ylabel('$V_{ex}$\n(m/s)' , fontsize=fsize, rotation=0, labelpad=lpad)
        axes[2, 2].set_ylabel('$J_{ix}$'        , fontsize=fsize, rotation=0, labelpad=lpad)
        axes[3, 2].set_ylabel('$E_x$\n(mV/m)'   , fontsize=fsize, rotation=0, labelpad=lpad)
        axes[4, 2].set_ylabel('$B_x$\n(nT)'     , fontsize=fsize, rotation=0, labelpad=lpad)

        axes[0, 2].set_ylim(te.min(), te.max())
        axes[1, 2].set_ylim(vex.min(), vex.max())
        axes[2, 2].set_ylim(jx.min(), jx.max())
        axes[3, 2].set_ylim(ex.min(), ex.max())
        
        fig.align_labels()
            
        for ii in range(3):
            axes[4, ii].set_xlabel('Position (m/dx)')
            for jj in range(5):
                axes[jj, ii].set_xlim(cf.B_nodes[0] / cf.dx, cf.B_nodes[-1] / cf.dx)
                axes[jj, ii].axvline(-cf.NX//2, c='k', ls=':', alpha=0.5)
                axes[jj, ii].axvline( cf.NX//2, c='k', ls=':', alpha=0.5)
                axes[jj, ii].grid()
                
        plt.tight_layout(pad=1.0, w_pad=1.8)
        fig.subplots_adjust(hspace=0.125)

        if save == True:
            plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close('all')
    return



#%% MAIN
if __name__ == '__main__':
    drive       = 'F:'
    
    series      = 'ABC_test_lowres_v5'
    series_dir  = '{}/runs//{}//'.format(drive, series)
    num_runs    = len([name for name in os.listdir(series_dir) if 'run_' in name])
    
    for run_num in [3]:#range(num_runs):
        print('Run {}'.format(run_num))
        cf.load_run(drive, series, run_num, extract_arrays=True)
        
        standard_analysis_package()
        
        summary_plots(save=True, histogram=False)
        check_fields()
        
        #plot_energies(normalize=False, save=True)
        #plot_helicity_colourplot()
        
        #disp_folder = 'dispersion_plots/'

        #single_point_both_fields_AGU()
        
        #do_all_dynamic_spectra(ymax=1.0)
        #do_all_dynamic_spectra(ymax=None)
        
        
        #ggg.get_linear_growth()

# =============================================================================
#         try:
#             single_point_field_timeseries()
#         except:
#             pass
# =============================================================================
        
# =============================================================================
#         try:
#             plot_spatially_averaged_fields()
#         except:
#             pass
# =============================================================================
        
# =============================================================================
#         try:
#             standard_analysis_package()
#         except:
#             pass
#         
#         try:
#             plot_energies(normalize=True, save=True)
#         except:
#             pass
# =============================================================================
        

