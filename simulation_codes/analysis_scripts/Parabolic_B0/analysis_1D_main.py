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
import matplotlib.dates    as mdates
from   mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
import pdb

import analysis_backend as bk
import analysis_config  as cf
import dispersions      as disp
import get_growth_rates as ggg

# Ignore common warnings. If shit goes funky, turn them back on by replacing 'ignore' with 'default'
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

qi     = 1.602e-19       # Elementary charge (C)
c      = 3e8             # Speed of light (m/s)
me     = 9.11e-31        # Mass of electron (kg)
mp     = 1.67e-27        # Mass of proton (kg)
e      = -qi             # Electron charge (C)
mu0    = (4e-7) * np.pi  # Magnetic Permeability of Free Space (SI units)
kB     = 1.38065e-23     # Boltzmann's Constant (J/K)
e0     = 8.854e-12       # Epsilon naught - permittivity of free space
B_surf = 3.12e-5         # Magnetic field strength at Earth surface

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

To do: 
    -- Refactor and split the most commonly used functions into their own files
        - 2D colour plots (wt, wx, kt, wk, dynamic spectra, etc.)
        - Particle plots (density histograms, distribution functions)
        - Summary plots (fields, winske plots, waterfall, energy plots etc.)
        - Event and run specific multiplot algorithms
        
    -- Change load files to HDF5 (including combining raw field files into one)
    
    -- Create a 'Run' class with each instance being a run containing all parameters,
        and pointers to the field and particle data. Should make things a lot
        cleaner! Also can create this class in a separate script since it will
        contain a lot of the load functions
'''
def shrink_cbar(ax, shrink=0.9):
    b = ax.get_position()
    new_h = b.height*shrink
    pad = (b.height-new_h)/2.
    new_y0 = b.y0 + pad
    new_y1 = b.y1 - pad
    b.y0 = new_y0
    b.y1 = new_y1
    ax.set_position(b)
    return ax


@nb.njit()
def calc_poynting(bx, by, bz, ex, ey, ez):
    '''
    Calculates point value of Poynting flux using E, B 3-vectors
    '''
    # Poynting Flux: Time, space, component
    S = np.zeros((bx.shape[0], bx.shape[1], 3))
    print('Calculating Poynting flux')
    for ii in range(bx.shape[0]):       # For each time
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
    for ii, comp in zip(range(3), ['x', 'y', 'z']):
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
                save_folder = cf.anal_dir + '/spatial_poynting_flux/'
                
                if os.path.exists(save_folder) == False:
                    os.makedirs(save_folder)
                
                fullpath = save_folder + saveas + '_s{}_{}'.format(comp, lbl) + suffix + '.png'
                plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
                plt.close('all')
    return


def plot_spatial_poynting_helical(saveas='poynting_helical_plot', save=False, log=False):
    '''
    Plots poynting flux at each gridpoint point in space
    
    -- Need to interpolate B to cell centers
    -- S = E x H = 1/mu0 E x B
    -- Loop through each cell     : E(t), B(t), S(t)
    -- Maybe also do some sort of S, x plot
    -- And/or do dynamic Poynting spectrum
    '''
    plt.ioff()

    ftime, Bt_pos, Bt_neg = bk.get_helical_components()

    By_pos = Bt_pos.real
    By_neg = Bt_neg.real
    Bz_pos = Bt_pos.imag
    Bz_neg = Bt_neg.imag
    Bx     = np.zeros(By_pos.shape)
    
    ftime, Et_pos, Et_neg = bk.get_helical_components(field='E')

    Ey_pos = Et_pos.real
    Ey_neg = Et_neg.real
    Ez_pos = Et_pos.imag
    Ez_neg = Et_neg.imag
    Ex     = np.zeros(Ey_pos.shape)

    Bx, By_pos, Bz_pos = bk.interpolate_B_to_center(Bx, By_pos, Bz_pos, zero_boundaries=False)
    Bx, By_neg, Bz_neg = bk.interpolate_B_to_center(Bx, By_neg, Bz_neg, zero_boundaries=False)
    
    S_pos              = calc_poynting(Bx, By_pos, Bz_pos, Ex, Ey_pos, Ez_pos)
    S_neg              = calc_poynting(Bx, By_neg, Bz_neg, Ex, Ey_neg, Ez_neg)
    
    ## PLOT IT (x component only)
    vlim = None
    for S, comp in zip([S_pos, S_neg], ['pos', 'neg']):
        print('Creating helical plots for S_{}'.format(comp))
        for tmax, lbl in zip([60, None], ['min', 'full']):
            fig, ax = plt.subplots(1, figsize=(15, 10))
            
            if log == True:
                im1 = ax.pcolormesh(cf.E_nodes, ftime, S[:, :, 0],
                     norm=colors.SymLogNorm(linthresh=1e-7, vmin=vlim, vmax=vlim),  
                     cmap='bwr')
                suffix = '_log'
            else:
                im1 = ax.pcolormesh(cf.E_nodes, ftime, S[:, :, 0], cmap='bwr', vmin=vlim, vmax=vlim)
                suffix = ''
            
            cb  = fig.colorbar(im1)
            cb.set_label('$S_%s$'%comp)
            
            if comp == 'pos':
                suff = 'Positive Helicity'
            else:
                suff = 'Negative Helicity'
            
            ax.set_title('Power Propagation :: Time vs. Space :: {}'.format(suff), fontsize=14)
            ax.set_ylabel(r't (s)', rotation=0, labelpad=15)
            ax.set_xlabel('x (m)')
            ax.set_ylim(0, tmax)
            
            ax.set_xlim(cf.grid_min, cf.grid_max)
            ax.axvline(cf.xmin, c='w', ls=':', alpha=1.0)
            ax.axvline(cf.xmax, c='w', ls=':', alpha=1.0)
            ax.axvline(cf.grid_mid, c='w', ls=':', alpha=0.75)   
                
            if save == True:
                save_folder = cf.anal_dir + '/spatial_poynting_flux/helical_components/'
                
                if os.path.exists(save_folder) == False:
                    os.makedirs(save_folder)
                    
                fullpath = save_folder + saveas + '_s{}_{}'.format(comp, lbl) + suffix + '.png'
                plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
                plt.close('all')
    return

def plot_tx(component='By', saveas='tx_plot', save=False, log=False, tmax=None, remove_ND=False, normalize=False,
            bmax=None):
    plt.ioff()

    t, arr = cf.get_array(component)
    
    fontsize = 18
    font     = 'monospace'
    
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    mpl.rcParams['ytick.labelsize'] = tick_label_size 
    
    if component[0] == 'B':
        arr *= 1e9
        x    = cf.B_nodes / cf.dx
    else:
        arr *= 1e3
        x    = cf.E_nodes / cf.dx
        
    if normalize == True and component[0].lower() == 'b':
        arr /= (cf.B_eq*1e9)
        
    if tmax is None:
        lbl = 'full'
    else:
        lbl = '{:04}'.format(tmax)
    
    if remove_ND == True:
        x_lim = [cf.xmin / cf.dx, cf.xmax / cf.dx]
    else:
        x_lim = [x[0], x[-1]]
    
    ## PLOT IT
    fig, ax = plt.subplots(1, figsize=(15, 10))
    
    if log == True:
        im1 = ax.pcolormesh(x, t, abs(arr),
                       norm=colors.LogNorm(vmin=1e-3, vmax=None), cmap='nipy_spectral')
        suffix = '_log'
    else:
        if bmax is None:
            vmin = arr.min()
            vmax = arr.max()
        else:
            vmin = -bmax
            vmax =  bmax
        im1 = ax.pcolormesh(x, t, arr, cmap='nipy_spectral', vmin=vmin, vmax=vmax)
        suffix = ''
    
    cb  = fig.colorbar(im1)
    
    if component[0] == 'B':
        if normalize == True:
            cb.set_label(r'$\frac{B_%s}{B_{0eq}}$' % component[1].lower(), rotation=0,
                         family=font, fontsize=fontsize, labelpad=30)
        else:
            cb.set_label('nT', rotation=0, family=font, fontsize=fontsize, labelpad=30)
    else:
        cb.set_label('mV/m', rotation=0, family=font, fontsize=fontsize, labelpad=30)

    ax.set_title('Time-Space ($t-x$) Plot :: {} component'.format(component.upper()), fontsize=fontsize, family=font)
    ax.set_ylabel('t (s)', rotation=0, labelpad=30, fontsize=fontsize, family=font)
    ax.set_xlabel('x ($\Delta x$)', fontsize=fontsize, family=font)
    ax.set_ylim(0, tmax)
    
    ax.axvline(cf.xmin     / cf.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(cf.xmax     / cf.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(0.0 / cf.dx, c='w', ls=':', alpha=0.75)   
    ax.set_xlim(x_lim[0], x_lim[1])
        
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}_{}'.format(component.lower(), lbl) + suffix + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('t-x Plot saved')
        plt.close('all')
    return


def plot_wx(component='By', saveas='wx_plot', linear_overlay=False, 
            save=False, pcyc_mult=None, remove_ND=False):
    plt.ioff()
    ftime, wx = disp.get_wx(component)
    
    if component[0] == 'B':
        x = cf.B_nodes
    else:
        x = cf.E_nodes
        
    f  = np.fft.rfftfreq(ftime.shape[0], d=cf.dt_field)
    
    if remove_ND == True:
        x_lim = [cf.xmin, cf.xmax]
    else:
        x_lim = [x[0], x[-1]]
    
    ## PLOT IT
    fig, ax = plt.subplots(1, figsize=(15, 10))
    im1 = ax.pcolormesh(x, f, wx, cmap='nipy_spectral')      # Remove f[0] since FFT[0] >> FFT[1, 2, ... , k]
    fig.colorbar(im1)
    
    lbl  = [r'$\Omega_{H^+}$', r'$\Omega_{He^+}$', r'$\Omega_{O^+}$']
    clr  = ['white', 'yellow', 'red']    
    M    = np.array([1., 4., 16.])
    
    try:
        pcyc = qi * cf.Bc[:, 0] / (2 * np.pi * mp)
    except:
        pcyc = qi * cf.B_eq * np.ones(cf.B_nodes.shape[0]) / (2 * np.pi * mp)
    
    for ii in range(3):
        if cf.species_present[ii] == True:
            try:
                cyc = qi * cf.Bc[:, 0] / (2 * np.pi * mp * M[ii])
            except:
                cyc = qi * cf.B_eq * np.ones(cf.B_nodes.shape[0]) / (2 * np.pi * mp * M[ii])
                
            ax.plot(cf.B_nodes, cyc, linestyle='--', c=clr[ii], label=lbl[ii])
    
# =============================================================================
#     if linear_overlay == True:
#         try:
#             freqs, cgr, stop = disp.get_cgr_from_sim()
#             max_idx          = np.where(cgr == cgr.max())
#             max_lin_freq     = freqs[max_idx]
#             plt.axhline(max_lin_freq, c='green', linestyle='--', label='CGR')
#         except:
#             pass
# =============================================================================

    ax.set_title('Frequency-Space ($\omega-x$) Plot :: {} component'.format(component.upper()), fontsize=14)
    ax.set_ylabel(r'f (Hz)', rotation=0, labelpad=15)
    ax.set_xlabel('x (m)')
    
    if pcyc_mult is not None:
        ax.set_ylim(0, pcyc_mult*pcyc.max())
    else:
        ax.set_ylim(0, None)

    ax.axvline(cf.xmin, c='w', ls=':', alpha=1.0)
    ax.axvline(cf.xmax, c='w', ls=':', alpha=1.0)
    ax.axvline(0.0, c='w', ls=':', alpha=0.75)   
    ax.set_xlim(x_lim[0], x_lim[1])

    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('w-x Plot saved')
        plt.close('all')
    return


def plot_kt(component='By', saveas='kt_plot', save=False, normalize_x=False, xlim=None):
    '''
    Note: k values from fftfreq() are their linear counterparts. Need to multiply by 2pi
    to compare with theory
    '''
    plt.ioff()
    k, ftime, kt, st, en = disp.get_kt(component)
    
    fig = plt.figure(1, figsize=(15, 10))
    ax  = fig.add_subplot(111)
    
    if normalize_x == False:
        im1 = ax.pcolormesh(k*1e6, ftime, kt, cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k] antialiased=True
        fig.colorbar(im1)
        ax.set_title('Wavenumber-Time ($k-t$) Plot :: {} component'.format(component.upper()), fontsize=14)
        ax.set_ylabel(r'$\Omega_i t$', rotation=0)
        ax.set_xlabel(r'$k (m^{-1}) \times 10^6$')
        #ax.set_ylim(0, 15)
    else:
        k_plot = 2*np.pi*k * c / cf.wpi
        
        im1 = ax.pcolormesh(k_plot, ftime, kt, cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k] antialiased=True
        fig.colorbar(im1)
        ax.set_title('Wavenumber-Time ($k-t$) Plot :: {} component'.format(component.upper()), fontsize=14)
        ax.set_ylabel(r'$\Omega_i t$', rotation=0)
        ax.set_xlabel(r'$kc/\omega_{pi}$')
        #ax.set_ylim(0, 15)
        ax.set_xlim(0, xlim)
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close('all')
        print('k-t Plot saved')
    return


def plot_wk(saveas='wk_plot', dispersion_overlay=False, save=True,
                     pcyc_mult=None, xmax=None, zero_cold=True,
                     linear_only=False, normalize_axes=False,
                     centre_only=False):
    '''
    21/02/2021 rewriting this to just take the w/k of the perp/parallel fields
    and do all fields in the one call.
    '''
    if normalize_axes == True:
        xfac = c / cf.wpi
        yfac = 2*np.pi / cf.gyfreq
        xlab = '$\mathtt{kc/\omega_{pi}}$'
        ylab = 'f\n$(\omega / \Omega_H)$'
        cyc  = 1.0 / np.array([1., 4., 16.])
    else:
        xfac = 1e6
        yfac = 1.0
        xlab = '$\mathtt{k (\\times 10^{-6}m^{-1})}$'
        ylab = 'f\n(Hz)'
        cyc  = qi * cf.B_eq / (2 * np.pi * mp * np.array([1., 4., 16.]))
        
        
    for field in ['B', 'E']:
    
        # Calculate dispersion relations from model data
        k, f, wk_para, tf = disp.get_wk(field+'x', linear_only=linear_only, norm_z=normalize_axes, centre_only=centre_only)
        k, f, wky,     tf = disp.get_wk(field+'y', linear_only=linear_only, norm_z=normalize_axes, centre_only=centre_only)
        k, f, wkz,     tf = disp.get_wk(field+'z', linear_only=linear_only, norm_z=normalize_axes, centre_only=centre_only)
        
        wk_perp = wky + wkz
        
        if field == 'B':
            clab = 'Pwr\n$\left(\\frac{nT^2}{Hz}\\right)$'
        else:
            clab = 'Pwr\n$\left(\\frac{mV^2}{m^2Hz}\\right)$'
    
        plt.ioff()
        
        fontsize = 18
        font     = 'monospace'
        mpl.rcParams['xtick.labelsize'] = 14 
        mpl.rcParams['ytick.labelsize'] = 14 
        
        fig1, ax1 = plt.subplots(1, figsize=(15, 10))
        
        im1 = ax1.pcolormesh(xfac*k[1:], yfac*f[1:], wk_perp[1:, 1:].real, cmap='jet',
                            norm=colors.LogNorm(vmin=wk_perp[1:, 1:].real.min(),
                                                vmax=wk_perp[1:, 1:].real.max()))
    
        fig1.colorbar(im1, extend='both', fraction=0.05).set_label(clab, rotation=0, fontsize=fontsize, family=font, labelpad=30)
        ax1.set_title(r'$\omega/k$ Plot :: {}_\perp :: Linear Theory up to {:.3f}s'.format(field, tf),
                     fontsize=fontsize, family=font)
            
        for ax in [ax1]:
            ax.set_ylabel(ylab, fontsize=fontsize, family=font, rotation=0, labelpad=30)
            ax.set_xlabel(xlab, fontsize=fontsize, family=font)
        
            ## -- EXTRAS to add to both plots
            lbl  = [r'$f_{eq, H^+}$', r'$f_{eq, He^+}$', r'$f_{eq, O^+}$']
            
            # Add labelled cyclotron frequencies
            from matplotlib.transforms import blended_transform_factory
            trans = blended_transform_factory(ax.transAxes, ax.transData)
            for ii in range(3):
                if cf.species_present[ii] == True:
                    ax.axhline(cyc[ii], linestyle=':', c='k')
                    ax.text(1.025, cyc[ii], lbl[ii], transform=trans, ha='center', 
                            va='center', color='k', fontsize=fontsize, family=font)
            
            ax.set_xlim(0, xmax)
            if pcyc_mult is not None:
                ax.set_ylim(0, pcyc_mult*cyc[0])
            else:
                ax.set_ylim(0, None)
            
            alpha=0.2
            if dispersion_overlay == True:
                k_vals, CPDR_solns, WPDR_solns, HPDR_solns = disp.get_linear_dispersion_from_sim(k, zero_cold=zero_cold)
                for ii in range(CPDR_solns.shape[1]):
                    ax.plot(xfac*k_vals, yfac*CPDR_solns[:, ii].real, c='k', linestyle='-' , label='CPDR' if ii == 0 else '', alpha=alpha)
                    ax.plot(xfac*k_vals, yfac*WPDR_solns[:, ii].real, c='k', linestyle='--', label='WPDR' if ii == 0 else '', alpha=alpha)
                    ax.plot(xfac*k_vals, yfac*HPDR_solns[:, ii].real, c='k', linestyle=':' , label='HPDR' if ii == 0 else '', alpha=alpha)
                ax.legend(loc='upper right', facecolor='white', prop={'size': fontsize-2, 'family':font})


            
        if save == True:
            zero_suff = '' if zero_cold is False else 'zero'
            fullpath1  = cf.anal_dir + saveas + '_{}perp'.format(field.upper()) + '_{}'.format(zero_suff)
            #fullpath2  = cf.anal_dir + saveas + '_{}para'.format(field.upper()) + '_{}'.format(zero_suff)
                
            fig1.savefig(fullpath1, facecolor=fig1.get_facecolor(), edgecolor='none', bbox_inches='tight')
            #fig2.savefig(fullpath2, facecolor=fig1.get_facecolor(), edgecolor='none', bbox_inches='tight')

            print('w-k for {} field saved'.format(field.upper()))
            plt.close('all')
        else:
            plt.show()
    return


def plot_wk_thesis_good(saveas='wk_plot_thesis', dispersion_overlay=False, save=True,
                     pcyc_mult=None, xmax=None, zero_cold=True,
                     linear_only=False, normalize_axes=False,
                     centre_only=False):
    '''
    Just do magnetic field, make axes big enough to see.
    
    Also need to put CPDR on these
    '''
    if normalize_axes == True:
        xfac = c / cf.wpi
        yfac = 2*np.pi / cf.gyfreq
        xlab = '$\mathtt{kc/\omega_{pi}}$'
        ylab = 'f\n$(\omega / \Omega_H)$'
        cyc  = 1.0 / np.array([1., 4., 16.])
    else:
        xfac = 1e6
        yfac = 1.0
        xlab = '$\mathtt{k (\\times 10^{-6}m^{-1})}$'
        ylab = 'f\n(Hz)'
        cyc  = qi * cf.B_eq / (2 * np.pi * mp * np.array([1., 4., 16.]))
            
    # Calculate dispersion relations from model data
    k, f, wk_para, tf = disp.get_wk('bx', linear_only=linear_only, norm_z=normalize_axes, centre_only=centre_only)
    k, f, wky,     tf = disp.get_wk('by', linear_only=linear_only, norm_z=normalize_axes, centre_only=centre_only)
    k, f, wkz,     tf = disp.get_wk('bz', linear_only=linear_only, norm_z=normalize_axes, centre_only=centre_only)
    
    wk_perp = wky + wkz
    
    clab = 'Pwr\n$\left(\\frac{nT^2}{Hz}\\right)$'

    plt.ioff()
    
    fontsize = 10
    font     = 'monospace'
    mpl.rcParams['xtick.labelsize'] = 14 
    mpl.rcParams['ytick.labelsize'] = 14 
    #cmin, cmax = 1e-5, 1e4
    cmin, cmax = None, None
    
    fig1, [ax, cax] = plt.subplots(nrows=1, ncols=2, figsize=(6.0, 4.0), 
                             gridspec_kw={'width_ratios':[1, 0.02]})
    
    im1 = ax.pcolormesh(xfac*k[1:], yfac*f[1:], wk_perp[1:, 1:].real, cmap='jet',
                        norm=colors.LogNorm(vmin=cmin,
                                            vmax=cmax))

    cbar = fig1.colorbar(im1, ax=ax, cax=cax, extend='both')
    cbar.set_label(clab, rotation=0, fontsize=fontsize, family=font, labelpad=30)
    ax.set_ylabel(ylab, fontsize=fontsize, family=font, rotation=0, labelpad=30)
    ax.set_xlabel(xlab, fontsize=fontsize, family=font)
    
    # Add labelled cyclotron frequencies
    #from matplotlib.transforms import blended_transform_factory
    #lbl  = [r'$f_{eq, H^+}$', r'$f_{eq, He^+}$', r'$f_{eq, O^+}$']
    #trans = blended_transform_factory(ax.transAxes, ax.transData)
    for ii in range(3):
        if cf.species_present[ii] == True:
            ax.axhline(cyc[ii], linestyle=':', c='k')
            #ax.text(1.025, cyc[ii], lbl[ii], transform=trans, ha='center', 
            #        va='center', color='k', fontsize=fontsize, family=font)
    
    ax.set_xlim(0, xmax)
    if pcyc_mult is not None:
        ax.set_ylim(0, pcyc_mult*cyc[0])
    else:
        ax.set_ylim(0, None)
    
    alpha=0.25
    if dispersion_overlay == True:
        k_vals, CPDR_solns, WPDR_solns, HPDR_solns = disp.get_linear_dispersion_from_sim(k, zero_cold=zero_cold)
        for ii in range(CPDR_solns.shape[1]):
            ax.plot(xfac*k_vals, yfac*CPDR_solns[:, ii].real, c='k', linestyle='--', label='CPDR' if ii == 0 else '', alpha=alpha)

    fig1.tight_layout()
    fig1.subplots_adjust(wspace=0.05)
    
    if save == True:
        zero_suff = '' if zero_cold is False else 'zero'
        fullpath1  = cf.anal_dir + saveas + '_Bperp' + '_{}'.format(zero_suff)
            
        fig1.savefig(fullpath1, facecolor=fig1.get_facecolor(), edgecolor='none', bbox_inches='tight')

        print('w-k for B-field saved')
        plt.close('all')
    else:
        plt.show()
    return


def plot_abs_FB(saveas='absFB_plot', save=False, log=False, tmax=None, normalize=False, B0_lim=None, remove_ND=False):
    '''
    Plot pcolormesh of tranverse magnetic field in space (x) and time (y).
    
    kwargs:
        saveas    -- Filename (without suffixes) to save as
        save      -- Save (True) or show (False)
        log       -- Plot colorbar on log scale
        tmax      -- Maximum time (y axis limit) to plot to
        normalize -- Normalize B_tranverse by equatorial B0 (True) or plot in nT (False)
        B0_lim    -- Colorbar limit (in multiples of B0 OR nT). If None, plot up to maximum value.
    '''
    plt.ioff()

    ftime, B_fwd, B_bwd, B_raw = bk.get_FB_waves(overwrite=False, field='B') 

    fontsize = 18
    font     = 'monospace'
    
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    mpl.rcParams['ytick.labelsize'] = tick_label_size 
    
    if normalize == False:
        bt     = np.sqrt(by ** 2 + bz ** 2) * 1e9
        clabel = '$|B_\perp|$\nnT'
        suff   = '' 
    else:
        bt     = np.sqrt(by ** 2 + bz ** 2) / cf.B_eq
        clabel = '$\\frac{|B_\perp|}{B_{eq}}$'
        suff   = '_norm' 
        
    x = cf.B_nodes / cf.dx
        
    if tmax is None:
        lbl = 'full'
    else:
        lbl = '{:04}'.format(tmax)
        
    if remove_ND == True:
        xlim = [cf.xmin/cf.dx, cf.xmax/cf.dx]
    else:
        xlim = [x[0], x[-1]]
    
    ## PLOT IT
    fig, ax = plt.subplots(1, figsize=(15, 10))

    vmin = 0.0
    
    if B0_lim is None:
        vmax = bt.max()
    else:
        if normalize == True:
            vmax = cf.B_eq * B0_lim * 1e9
        else:
            vmax = B0_lim

    if log == False:
        im1     = ax.pcolormesh(x, t, bt, cmap='jet', vmin=vmin, vmax=vmax)
        logsuff = ''
    else:
        im1     = ax.pcolormesh(x, t, np.log10(bt), cmap='jet', vmin=0.0, vmax=-3)
        logsuff = '_log'
        
    cb   = fig.colorbar(im1)
    
    cb.set_label(clabel, rotation=0, family=font, fontsize=fontsize, labelpad=30)

    ax.set_ylabel('t (s)', rotation=0, labelpad=30, fontsize=fontsize, family=font)
    ax.set_xlabel('x ($\Delta x$)', fontsize=fontsize, family=font)
    ax.set_ylim(0, tmax)
    
    ax.axvline(cf.xmin     / cf.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(cf.xmax     / cf.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(0.0                , c='w', ls=':', alpha=0.75)   
    ax.set_xlim(xlim[0], xlim[1])    
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_BPERP_{}{}{}'.format(lbl, suff, logsuff) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('B abs(t-x) Plot saved')
        plt.close('all')
    return


def plot_abs_T(saveas='abs_plot', save=False, log=False, tmax=None, normalize=False, B0_lim=None, remove_ND=False):
    '''
    Plot pcolormesh of tranverse magnetic field in space (x) and time (y).
    
    kwargs:
        saveas    -- Filename (without suffixes) to save as
        save      -- Save (True) or show (False)
        log       -- Plot colorbar on log scale
        tmax      -- Maximum time (y axis limit) to plot to
        normalize -- Normalize B_tranverse by equatorial B0 (True) or plot in nT (False)
        B0_lim    -- Colorbar limit (in multiples of B0 OR nT). If None, plot up to maximum value.
    '''
    plt.ioff()

    t, bx, by, bz, ex, ey, ez, vex, vey, vez, te, jx, jy, jz, qdens, field_sim_time, damping_array\
        = cf.get_array(get_all=True)

    fontsize = 10
    font     = 'monospace'
    
    tick_label_size = 8
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    mpl.rcParams['ytick.labelsize'] = tick_label_size 
    
    if normalize == False:
        bt     = np.sqrt(by ** 2 + bz ** 2) * 1e9
        clabel = '$|B_\perp|$\nnT'
        suff   = '' 
    else:
        bt     = np.sqrt(by ** 2 + bz ** 2) / cf.B_eq
        clabel = '$\\frac{|B_\perp|}{B_{eq}}$'
        suff   = '_norm' 
        
    x = cf.B_nodes / cf.dx
        
    if tmax is None:
        lbl = 'full'
    else:
        lbl = '{:04}'.format(tmax)
        
    if remove_ND == True:
        xlim = [cf.xmin/cf.dx, cf.xmax/cf.dx]
    else:
        xlim = [x[0], x[-1]]
    
    ## PLOT IT
    fig, ax = plt.subplots(1, figsize=(6.0, 4.0))

    vmin = 0.0
    
    if B0_lim is None:
        vmax = bt.max()
    else:
        if normalize == True:
            vmax = cf.B_eq * B0_lim * 1e9
        else:
            vmax = B0_lim

    if log == False:
        im1     = ax.pcolormesh(x, t, bt, cmap='jet', vmin=vmin, vmax=vmax)
        logsuff = ''
    else:
        im1     = ax.pcolormesh(x, t, np.log10(bt), cmap='jet', vmin=0.0, vmax=-3)
        logsuff = '_log'
        
    cb   = fig.colorbar(im1)
    
    cb.set_label(clabel, rotation=0, family=font, fontsize=fontsize, labelpad=30)

    ax.set_ylabel('t (s)', rotation=0, labelpad=30, fontsize=fontsize, family=font)
    ax.set_xlabel('x ($\Delta x$)', fontsize=fontsize, family=font)
    ax.set_ylim(0, tmax)
    
    ax.axvline(cf.xmin     / cf.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(cf.xmax     / cf.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(0.0                , c='w', ls=':', alpha=0.75)   
    ax.set_xlim(xlim[0], xlim[1])    
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_BPERP_{}{}{}'.format(lbl, suff, logsuff) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=200)
        print('B abs(t-x) Plot saved')
        plt.close('all')
    return


def plot_abs_T_w_Bx(saveas='abs_plot', save=False, tmax=None,
                    B0_lim=None, remove_ND=False):
    '''
    Plot pcolormesh of tranverse magnetic field in space (x) and time (y).
    
    kwargs:
        saveas    -- Filename (without suffixes) to save as
        save      -- Save (True) or show (False)
        log       -- Plot colorbar on log scale
        tmax      -- Maximum time (y axis limit) to plot to
        normalize -- Normalize B_tranverse by equatorial B0 (True) or plot in nT (False)
        B0_lim    -- Colorbar limit (in multiples of B0 OR nT). If None, plot up to maximum value.
    '''
    plt.ioff()

    t, bx, by, bz, ex, ey, ez, vex, vey, vez, te, jx, jy, jz, qdens, field_sim_time, damping_array\
        = cf.get_array(get_all=True)
    t, bxc = cf.get_array(component='bxc', get_all=False, timebase=None)

    bxc = bxc[:, cf.NX//2]*1e9
    
    fontsize = 18
    font     = 'monospace'
    
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    mpl.rcParams['ytick.labelsize'] = tick_label_size 
    
    bt = np.sqrt(by ** 2 + bz ** 2) * 1e9
    x  = cf.B_nodes / cf.dx
        
    if tmax is None:
        lbl = 'full'
        tmax = t[-1]
    else:
        lbl = '{:04}'.format(tmax)
        
    if remove_ND == True:
        xlim = [cf.xmin/cf.dx, cf.xmax/cf.dx]
    else:
        xlim = [x[0], x[-1]]
    
    ## PLOT IT
    plt.ioff()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), gridspec_kw={'width_ratios':[0.2, 1.0, 0.02]})

    if B0_lim is None:
        vmax = bt.max()
    else:
        vmax = B0_lim
    
    axes[0].plot(bxc, t, c='b')
    axes[0].set_ylabel('t (s)', rotation=0, labelpad=30, fontsize=fontsize, family=font)
    axes[0].set_ylim(0, tmax)
    axes[0].set_xlabel('$B_0 (nT)$', fontsize=fontsize, family=font)    
    im1 = axes[1].pcolormesh(x, t, bt, cmap='jet', vmin=0.0, vmax=vmax)
    cb  = fig.colorbar(im1, cax=axes[2])
    cb.set_label('$|B_\perp|$\nnT', rotation=0, family=font, fontsize=fontsize, labelpad=30)

    
    axes[1].set_xlabel('x ($\Delta x$)', fontsize=fontsize, family=font)
    axes[1].set_ylim(0, tmax)
    axes[1].set_yticklabels([])
    
    axes[1].axvline(cf.xmin / cf.dx, c='w', ls=':', alpha=1.0)
    axes[1].axvline(cf.xmax / cf.dx, c='w', ls=':', alpha=1.0)
    axes[1].axvline(0.0                , c='w', ls=':', alpha=0.75)   
    axes[1].set_xlim(xlim[0], xlim[1])    
    
    fig.subplots_adjust(wspace=0.0)
    if save == True:
        fullpath = cf.anal_dir + saveas + '_BPERP_{}'.format(lbl) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('B abs(t-x) Plot saved')
        plt.close('all')
    else:
        plt.show()
    return


def plot_abs_J(saveas='abs_plot', save=False, log=False, tmax=None, remove_ND=False):
    '''
    Plot pcolormesh of tranverse magnetic field in space (x) and time (y).
    
    kwargs:
        saveas    -- Filename (without suffixes) to save as
        save      -- Save (True) or show (False)
        log       -- Plot colorbar on log scale
        tmax      -- Maximum time (y axis limit) to plot to
        normalize -- Normalize B_tranverse by equatorial B0 (True) or plot in nT (False)
        B0_lim    -- Colorbar limit (in multiples of B0). If None, plot up to maximum value.
    '''
    plt.ioff()

    t, bx, by, bz, ex, ey, ez, vex, vey, vez, te, jx, jy, jz, qdens, field_sim_time, damping_array\
        = cf.get_array(get_all=True)

    fontsize = 18
    font     = 'monospace'
    
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    mpl.rcParams['ytick.labelsize'] = tick_label_size 
    
    Jt     = np.sqrt(jx ** 2 + jy ** 2 + jz ** 2) * 1e3
    clabel = '$|J_T|$\nmA/m'
    suff   = '' 
        
    x    = cf.E_nodes / cf.dx
        
    if tmax is None:
        lbl = 'full'
    else:
        lbl = '{:04}'.format(tmax)
        
    if remove_ND == True:
        xlim = [cf.xmin/cf.dx, cf.xmax/cf.dx]
    else:
        xlim = [x[0], x[-1]]
    
    ## PLOT IT
    fig, ax = plt.subplots(1, figsize=(15, 10))

    vmin = 0.0
    vmax = Jt.max()

    if log == False:
        im1     = ax.pcolormesh(x, t, Jt, cmap='jet', vmin=vmin, vmax=vmax)
        logsuff = ''
    else:
        im1     = ax.pcolormesh(x, t, np.log10(Jt), cmap='jet', vmin=0.0, vmax=-3)
        logsuff = '_log'
        
    cb   = fig.colorbar(im1)
    
    cb.set_label(clabel, rotation=0, family=font, fontsize=fontsize, labelpad=30)

    ax.set_ylabel('t (s)', rotation=0, labelpad=30, fontsize=fontsize, family=font)
    ax.set_xlabel('x ($\Delta x$)', fontsize=fontsize, family=font)
    ax.set_ylim(0, tmax)
    
    ax.axvline(cf.xmin     / cf.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(cf.xmax     / cf.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(0.0                , c='w', ls=':', alpha=0.75)   
    ax.set_xlim(xlim[0], xlim[1])    
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_JTOT_{}{}{}'.format(lbl, suff, logsuff) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('J abs(t-x) Plot saved')
        plt.close('all')
    return


def plot_stability_check():
    '''
    Diagnostic. Plots maximum value of stability criteria against current timestep
    -- Black lines: dt and dt/subcycle
    -- Resolution criteria:
        -- Fastest particle, wL = k*v = pi*v/dx
        -- Electric field,   wE = qs*E / ms*v
        -- Gyrofrequency,    wG = qs*B / ms
        -- Dispersion,       wD = k^2*B / mu0*pc
    -- Stability criteria:
        -- Phase speed? Compare this to c or vA? Does it even apply to hybrids?
        
    For particle criteria, need to fetch fields at particle position, calculate
    wG and wE for each particle, and take the max value. For wL, wD, just take the 
    max value for the particle and field quantities, respectively.
    '''
    # Each should be a timeseries
    # G, D depend on fields; L, E depend on particles
    ftime, ptime, wL, wE, wG, wD, vdt_max = bk.get_stability_criteria()
    print('Plotting...')

    fig, ax = plt.subplots()
    # Could use orbit_res and freq_res to check
    ax.semilogy(ftime, 0.05/wG, label='$0.05\omega_G^{-1}$')
    ax.semilogy(ftime, 0.05/wD, label='$0.05\omega_D^{-1}$')
    #ax.semilogy(ptime, 0.05/wL, label='$0.05\omega_L^{-1}$')
    ax.semilogy(ptime, 0.05/wE, label='$0.05\omega_E^{-1}$')
    ax.semilogy(ptime, vdt_max, label='$v\Delta t < 0.5\Delta x$')
    
    ax.axhline(cf.dt_sim, c='k', label='$\Delta t$', alpha=0.50)
    ax.axhline(cf.dt_sim/cf.subcycles, c='k', label='\Delta t_s', alpha=0.50, ls='--')
    ax.set_xlabel('Simulation Time')
    ax.set_ylabel('Timestep Stability Limits')
    ax.legend()
    return


def single_point_FB(nx=None, overlap=0.5, f_res_mHz=50, fmax=1.0):
    '''
    Frequency resolution in fraction of gyfreq (defaults to 1/50th)
    '''
    analysis_scripts_dir = 'D://Google Drive//Uni//PhD 2017//Data//Scripts//'
    sys.path.append(analysis_scripts_dir)
    
    import fast_scripts as fscr
    
    dynspec_folderpath = cf.anal_dir + '//Bfield_FB_dynspec//'
    if not os.path.exists(dynspec_folderpath): os.makedirs(dynspec_folderpath)
    
    if nx is None:
        nx = np.array([cf.NC // 2])
    
    # (time, space)
    ftime, B_fwd, B_bwd, B_raw = bk.get_FB_waves(overwrite=False, field='B')  
    _dt  = ftime[1] - ftime[0]
    if fmax is None:
        fmax = cf.gyfreq / (2*np.pi)    # Gyfreq in Hz
    
    for _NX in nx:
        fwd_tseries = B_fwd[:, _NX]*1e9
        bwd_tseries = B_bwd[:, _NX]*1e9
        
        # Forward-wave spectrum
        fypow, fytime, fyfreq = fscr.autopower_spectrum_all(ftime, fwd_tseries, _dt, overlap=overlap,
                                                         df=f_res_mHz, window_data=True)
        fzpow, fztime, fzfreq = fscr.autopower_spectrum_all(ftime, fwd_tseries, _dt, overlap=overlap,
                                                         df=f_res_mHz, window_data=True)
        fpow = fypow + fzpow
        
        # Backward-wave spectrum
        bypow, bytime, byfreq = fscr.autopower_spectrum_all(ftime, bwd_tseries, _dt, overlap=overlap,
                                                         df=f_res_mHz, window_data=True)
        bzpow, bztime, bzfreq = fscr.autopower_spectrum_all(ftime, bwd_tseries, _dt, overlap=overlap,
                                                         df=f_res_mHz, window_data=True)
        bpow = bypow + bzpow
        
        ## PLOT ##
        plt.ioff()
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.0, 3.0),
                                 gridspec_kw={'width_ratios':[1, 0.01]})
        
        # Forward plot
        im1 = axes[0, 0].pcolormesh(fytime, fyfreq, fpow.T,
                       norm=colors.LogNorm(vmin=1e-5, vmax=1e0), cmap='jet')
        fig.colorbar(im1, cax=axes[0, 1], extend='max').set_label(
                '$|P_f|$\n$nT^2/Hz$', fontsize=12, rotation=0, labelpad=30)
        
        # Backward plot
        im2 = axes[1, 0].pcolormesh(bytime, byfreq, bpow.T,
                       norm=colors.LogNorm(vmin=1e-5, vmax=1e0), cmap='jet')
        fig.colorbar(im2, cax=axes[1, 1], extend='max').set_label(
                '$|P_b|$\n$nT^2/Hz$', fontsize=12, rotation=0, labelpad=30)
        
        for ax in axes[:, 0]:
            ax.set_ylim(0, fmax)
            ax.set_xlim(0, ftime[-1])
            ax.set_ylabel('Hz', rotation=0, labelpad=30)
        
        axes[0, 0].set_xticklabels([])
        axes[-1, 0].set_xlabel('Time (s)')
        
        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0.05)
        fig.savefig(dynspec_folderpath + 'SP_spectra_nx{:04}.png'.format(_NX), 
                    facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
        
        plt.close('all')
        print(f'SP spectra for node {_NX} saved')
    return


def single_point_spectra(nx=None, overlap=0.5, f_res_mHz=50, fmax=1.0):
    '''
    Frequency resolution in fraction of gyfreq (defaults to 1/50th)
    '''
    analysis_scripts_dir = 'D://Google Drive//Uni//PhD 2017//Data//Scripts//'
    sys.path.append(analysis_scripts_dir)
    
    import fast_scripts as fscr
    
    dynspec_folderpath = cf.anal_dir + '//Bfield_SP_dynspec//'
    if not os.path.exists(dynspec_folderpath): os.makedirs(dynspec_folderpath)
    
    if nx is None:
        nx = np.array([cf.NC // 2])
    
    # (time, space)
    ftime, By = cf.get_array('By')
    ftime, Bz = cf.get_array('Bz')    
    _dt  = ftime[1] - ftime[0]
    if fmax is None:
        fmax = cf.gyfreq / (2*np.pi)    # Gyfreq in Hz

    for _NX in nx:
        ypow, ytime, yfreq = fscr.autopower_spectrum_all(ftime, By[:, _NX]*1e9, _dt, overlap=overlap,
                                                         df=f_res_mHz, window_data=True)
        zpow, ztime, zfreq = fscr.autopower_spectrum_all(ftime, Bz[:, _NX]*1e9, _dt, overlap=overlap,
                                                         df=f_res_mHz, window_data=True)
        spow = ypow + zpow
        
        ## PLOT ##
        plt.ioff()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.0, 3.0),
                                 gridspec_kw={'width_ratios':[1, 0.01]})
            
        im1 = axes[0].pcolormesh(ytime, yfreq, spow.T,
                       norm=colors.LogNorm(vmin=1e-5, vmax=1e1), cmap='jet')
        fig.colorbar(im1, cax=axes[1], extend='max').set_label(
                '$|P|$\n$nT^2/Hz$', fontsize=12, rotation=0, labelpad=30)
        
        
        B_NX  = cf.B_eq * (1 + cf.a * cf.B_nodes[_NX]**2)
        gy_H  = qi * B_NX / (2 * np.pi * mp) 
        gy_He = 0.2500 * gy_H
        gy_O  = 0.0625 * gy_H 
        
        #axes[0].axhline(gy_H , c='w', label='$f_{cH^+}$')
        axes[0].axhline(gy_He, c='yellow', label='$f_{cHe^+}$')
        axes[0].axhline(gy_O , c='r', label='$f_{cO^+}$')
        
        axes[0].legend(loc='upper left', ncol=2)
        
        axes[0].set_ylabel('Hz', rotation=0, labelpad=30)
        axes[0].set_ylim(0, fmax)
        axes[0].set_xlim(0, ftime[-1])
        axes[0].set_xlabel('Time (s)')
        
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.05)
        fig.savefig(dynspec_folderpath + 'SP_spectra_nx{:04}_revised.png'.format(_NX), 
                    facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
        
        plt.close('all')
        print(f'SP spectra for node {_NX} at {cf.B_nodes[_NX]*1e-6}Mm saved')
    return


def plot_energies(normalize=True, save=False):
    '''
    Parameters
    ----------
    normalize : Bool, optional
        Normalize each energy to its initial value. Default True.
    save : Bool, optional
        Save as an image file, else plot in figure. Default False.
        
    Note: Particle energies are (time, species, direction) where direction specifies
        energy parallel (0) or perpendicular (1) to the magnetic field/simulation space
    '''
    print('Plotting energies...')
    ftime_sec, ptime_sec, mag_energy, electron_energy, particle_energy, total_energy = bk.get_energies()

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
    skip   = 2
    amp    = 10.                 # Amplitude multiplier of waves:
    
    sep    = 1.
    dark   = 1.0
    cells  = np.arange(cf.NX)

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

    st  = cf.ND
    en  = cf.ND + cf.NX

    for ii in np.arange(it_max):
        if ii%skip == 0:
            ax5.plot(cells, amp*(By_raw[ii, st:en] / By_raw.max()) + sep*ii, c='k', alpha=dark)
            ax6.plot(cells, amp*(Bz_raw[ii, st:en] / Bz_raw.max()) + sep*ii, c='k', alpha=dark)

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


def winske_summary_plots(save=True):
    '''
    Plot summaries as per Winske et al. (1993)
    Really only works for Winske load files.
    0: Cold
    1: Hot
    '''  
    np.set_printoptions(suppress=True)
    
    num_particle_steps = len(os.listdir(cf.particle_dir))
    if num_particle_steps == 0:
        print('ABORT: No particle data present to create summary plots.')
        return

    path = cf.anal_dir + '/winske_plots/'
    if os.path.exists(path) == False:                                   # Create data directory
        os.makedirs(path)
    
    plt.ioff()
    ptime_sec, pbx, pby, pbz, pex, pey, pez, pvex, pvey,\
    pvez, pte, pjx, pjy, pjz, pqdens = cf.interpolate_fields_to_particle_time(num_particle_steps)

    radperiods = ptime_sec * cf.gyfreq
    gperiods   = ptime_sec / cf.gyperiod
    
    # Normalize units
    qdens_norm = pqdens / (cf.density*cf.charge).sum()     
    BY_norm    = pby / cf.B_eq
    BZ_norm    = pbz / cf.B_eq
    PHI        = np.arctan2(BZ_norm, BY_norm)
    
    for ii in range(num_particle_steps):
        filename = 'winske_summary%05d.png' % ii
        fullpath = path + filename
        
        if os.path.exists(fullpath):
            sys.stdout.write('\rSummary plot already present for timestep [{}]{}'.format(run_num, ii))
            sys.stdout.flush()
            continue
        
        sys.stdout.write('\rCreating summary plot for particle timestep [{}]{}'.format(run_num, ii))
        sys.stdout.flush()

        fig, axes = plt.subplots(5, figsize=(20,10), sharex=True)                  # Initialize Figure Space

        xp, vp, idx, psim_time, idx_start, idx_end = cf.load_particles(ii)
        
        pos       = xp / cf.dx
        vel       = vp / c
        cell_B    = cf.B_nodes/cf.dx
        cell_E    = cf.E_nodes/cf.dx
        st, en    = idx_start[1], idx_end[1]
        xmin      = cf.xmin / cf.dx
        xmax      = cf.xmax / cf.dx

        axes[0].scatter(pos[st: en], vel[0, st: en], s=1, c='k')
        axes[1].scatter(pos[st: en], vel[1, st: en], s=1, c='k')
        axes[2].plot(cell_E, qdens_norm[ii], color='k', lw=1.0)
        axes[3].plot(cell_B, BY_norm[ii], color='k')
        axes[4].plot(cell_B, PHI[ii],     color='k')
        
        axes[0].set_ylabel('VX ', labelpad=20, rotation=0)
        axes[1].set_ylabel('VY ', labelpad=20, rotation=0)
        axes[2].set_ylabel('DN ', labelpad=20, rotation=0)
        axes[3].set_ylabel('BY ', labelpad=20)
        axes[4].set_ylabel('PHI', labelpad=20)
        fig.align_ylabels()
        
        axes[0].set_title('BEAM')
        axes[1].set_title('BEAM')
        
        axes[0].set_ylim(-2e-3, 2e-3)
        axes[1].set_ylim(-2e-3, 2e-3)
        axes[2].set_ylim(0.5, 1.5)
        axes[4].set_ylim(-np.pi, np.pi)
        
        axes[0].set_title('WINSKE RUNS :: IT={:04d} :: T={:5.2f} :: GP={:5.2f}'.format(ii, radperiods[ii], gperiods[ii]), family='monospace')
        axes[4].set_xlabel('X')
        
        for ax in axes:
            ax.set_xlim(xmin, xmax)
        
        if save == True:
            plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close('all')

    print('\n')
    return


def winske_magnetic_density_plot(save=True):
    np.set_printoptions(suppress=True)

    ftime, by = cf.get_array('By')
    ftime, bz = cf.get_array('Bz')
    b_squared = (by ** 2 + bz ** 2).mean(axis=1) / cf.B_eq ** 2

    radperiods = ftime * cf.gyfreq

    fig, ax = plt.subplots(figsize=(20,10), sharex=True)                  # Initialize Figure Space

    ax.plot(radperiods, b_squared, color='k')
    ax.set_ylabel('B**2', labelpad=20, rotation=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.48)
    ax.set_xlabel('T')
    
    if save == True:
        fullpath = cf.anal_dir + 'winske_magnetic_timeseries' + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
        print('t-x Plot saved')
        plt.close('all')
    else:
        plt.show()
    return


def summary_plots(save=True, histogram=True, skip=1, ylim=True):
    '''
    Plot summary plot of raw values for each particle timestep
    Field values are interpolated to this point
    
    To Do: Find some nice way to include species energies instead of betas
    '''  
    np.set_printoptions(suppress=True)
    
    num_particle_steps = len(os.listdir(cf.particle_dir))
    if num_particle_steps == 0:
        print('ABORT: No particle data present to create summary plots.')
        return
    
    if histogram == True:
        path = cf.anal_dir + '/summary_plots_histogram/'
    else:
        path = cf.anal_dir + '/summary_plots/'
        
    if os.path.exists(path) == False:                                   # Create data directory
        os.makedirs(path)
    
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
    den_max = None#qdens_norm.max()
    den_min = None#2.0 - den_max

    # Set lims to ceiling values (Nope: Field limits are still doing weird shit. It's alright.)
    B_lim = np.ceil(B_lim)
    E_lim = np.ceil(E_lim)
    
    for ii in range(num_particle_steps):
        if ii%skip == 0:
            filename = 'summ%05d.png' % ii
            fullpath = path + filename
            
            if os.path.exists(fullpath):
                sys.stdout.write('\rSummary plot already present for timestep [{}]{}'.format(run_num, ii))
                sys.stdout.flush()
                continue
            
            sys.stdout.write('\rCreating summary plot for particle timestep [{}]{}'.format(run_num, ii))
            sys.stdout.flush()
    
            fig_size = 4, 7                                                             # Set figure grid dimensions
            fig = plt.figure(figsize=(20,10))                                           # Initialize Figure Space
            fig.patch.set_facecolor('w')   
            xp, vp, idx, psim_time, idx_start, idx_end = cf.load_particles(ii)
            
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
                    try:
                        num_bins = cf.nsp_ppc[jj] // 5
                        
                        xs, BinEdgesx = np.histogram(vel[0, idx_start[jj]: idx_end[jj]], bins=num_bins)
                        bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
                        ax_vx.plot(bx, xs, '-', c=cf.temp_color[jj], drawstyle='steps', label=cf.species_lbl[jj])
                        
                        ys, BinEdgesy = np.histogram(vel_tr[idx_start[jj]: idx_end[jj]], bins=num_bins)
                        by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
                        ax_vy.plot(by, ys, '-', c=cf.temp_color[jj], drawstyle='steps', label=cf.species_lbl[jj])
                    except: 
                        pass
                    
                    ax_vx.set_ylabel(r'$n_{v_\parallel}$')
                    ax_vx.set_ylabel(r'$n_{v_\perp}$')
                    
                    ax_vx.set_title('Velocity distribution of each species in simulation domain')
                    ax_vy.set_xlabel(r'$v / v_A$')
                    
                    ax_vx.set_xlim(-vel_lim, vel_lim)
                    ax_vy.set_xlim(0, np.sqrt(2)*vel_lim)
                    
                    for ax, comp in zip([ax_vx, ax_vy], ['v_\parallel', 'v_\perp']):
                        ax.legend(loc='upper right')
                        ax.set_ylabel('$n_{%s}$'%comp)
                        if ylim == True:
                            ax.set_ylim(0, int(cf.N / cf.NX) * 10.0)
            else:
            
                for jj in reversed(range(cf.Nj)):
                    ax_vx.scatter(pos[idx_start[jj]: idx_end[jj]], vel[0, idx_start[jj]: idx_end[jj]], s=1, c=cf.temp_color[jj], lw=0, label=cf.species_lbl[jj])
                    ax_vy.scatter(pos[idx_start[jj]: idx_end[jj]], vel[1, idx_start[jj]: idx_end[jj]], s=1, c=cf.temp_color[jj], lw=0)
            
                ax_vx.legend(loc='upper right')
                ax_vx.set_title(r'Particle velocities (in $v_A$) vs. Position (x)')
                ax_vy.set_xlabel(r'Cell', labelpad=10)
                ax_vx.set_ylabel(r'$v_x$', rotation=0)
                ax_vy.set_ylabel(r'$v_y$', rotation=0)
                
                plt.setp(ax_vx.get_xticklabels(), visible=False)
                ax_vx.set_yticks(ax_vx.get_yticks()[1:])
            
                for ax in [ax_vy, ax_vx]:
                    ax.set_xlim(-cf.xmax, cf.xmax)
                    if ylim == True:
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
            
            ax_Ex.legend(loc='upper right', ncol=3)
            
            ax_By  = plt.subplot2grid(fig_size, (2, 3), colspan=3, sharex=ax_den)
            ax_B   = plt.subplot2grid(fig_size, (3, 3), colspan=3, sharex=ax_den)
            mag_B  = np.sqrt(pby[ii] ** 2 + pbz[ii] ** 2)
            
            ax_Bx = ax_B.twinx()
            ax_Bx.plot(B_nodes, pbx[ii]*1e9, color='k', label=r'$B_x$', ls=':', alpha=0.6) 
            
            if cf.B_eq == cf.Bc.max():
                pass
            else:
                if ylim == True:
                    ax_Bx.set_ylim(cf.B_eq*1e9, cf.Bc.max()*1e9)
                
            ax_Bx.set_ylabel(r'$B_{0x} (nT)$', rotation=0, labelpad=30, fontsize=14)
            
            ax_B.plot( B_nodes, mag_B*1e9, color='g')
            ax_By.plot(B_nodes, pby[ii]*1e9, color='g',   label=r'$B_y$') 
            ax_By.plot(B_nodes, pbz[ii]*1e9, color='b',   label=r'$B_z$') 
            ax_By.legend(loc='upper right', ncol=2)
            
            ax_B.set_ylabel( r'$B_\perp (nT)$', rotation=0, labelpad=30, fontsize=14)
            ax_By.set_ylabel(r'$B_{y,z} (nT)$', rotation=0, labelpad=20, fontsize=14)
            ax_B.set_xlabel('Cell Number')
            
            # SET FIELD RANGES #
            if ylim == True:
                try:
                    ax_den.set_ylim(den_min, den_max)
                except:
                    pass
                
                try:
                    ax_Ex.set_ylim(-E_lim, E_lim)
                except:
                    pass
                
                try:
                    ax_By.set_ylim(-B_lim, B_lim)
                    ax_B.set_ylim(0, B_lim)
                except:
                    pass
            
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
            Tperp = cf.mass * cf.vth_perp ** 2 / kB
            anisotropy = (cf.vth_perp ** 2 / cf.vth_par ** 2 - 1).round(1)
            beta_per   = (2*(4e-7*np.pi)*(1.381e-23)*Tperp*cf.ne / (cf.B_eq**2)).round(1)
            #beta_e     = np.round((2*(4e-7*np.pi)*(1.381e-23)*cf.Te0*cf.ne  / (cf.B_eq**2)), 2)
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
            
            #plt.figtext(0.855, top - 12*gap, r'$\beta_e$      : %.2f' % beta_e, fontsize=fontsize, family='monospace')
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
    print('\n')
    return


def standard_analysis_package(disp_overlay=False, pcyc_mult=1.25,
                              tx_only=False, tmax=None, remove_ND=False):
    '''
    Need a high-pass option for the wk? Or will it do all of it?
    It should do all of it (show the Pc4 branch and the Pc1 branch)
    Since they're both in the magnetic field
    
    Actually, Pc5 is only in bx... affects By/Bz? Echo in other components?
    '''
    disp_folder = 'dispersion_plots/'
        
    if os.path.exists(cf.anal_dir + disp_folder) == False:
        os.makedirs(cf.anal_dir + disp_folder)
    
    #plot_wk_polished(component='by', saveas=disp_folder + 'wk_plot', save=False, dispersion_overlay=True, pcyc_mult=1.25)
    
    if disp_overlay: 
        plot_wk(saveas='wk_plot_zc', dispersion_overlay=disp_overlay, save=True,
                         pcyc_mult=pcyc_mult, xmax=None, zero_cold=True,
                         linear_only=True, normalize_axes=False)
                
    plot_wk(saveas='wk_plot', dispersion_overlay=disp_overlay, save=True,
         pcyc_mult=pcyc_mult, xmax=None, zero_cold=False,
         linear_only=False, normalize_axes=False)

    if True:    
        for comp in ['By', 'Bz', 'Ex', 'Ey', 'Ez']:
            print('2D summary for {}'.format(comp))

            plot_tx(component=comp, saveas=disp_folder + 'tx_plot', save=True, tmax=None, remove_ND=remove_ND)
            
            if tx_only == False:
                plot_wx(component=comp, saveas=disp_folder + 'wx_plot_pcyc', remove_ND=remove_ND, save=True, linear_overlay=False, pcyc_mult=pcyc_mult)
                plot_wx(component=comp, saveas=disp_folder + 'wx_plot'     , remove_ND=remove_ND, save=True, linear_overlay=False, pcyc_mult=None)
                plot_kt(component=comp, saveas=disp_folder + 'kt_plot', save=True)
                
                
                
                if False:
                    plot_spatial_poynting(save=True, log=True)
                    plot_spatial_poynting_helical(save=True, log=True)
                    #plot_helical_waterfall(title='{}: Run {}'.format(series, run_num), save=True)
                    #plot_initial_configurations()
                
                if False:
                    check_fields()
                    plot_energies(normalize=True, save=True)
                    plot_ion_energy_components(save=True, tmax=1./cf.HM_frequency)
                    single_point_helicity_timeseries()
                    plot_spatially_averaged_fields()
                    single_point_field_timeseries(tmax=1./cf.HM_frequency)
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


def check_fields(save=True, ylim=True, skip=1):
    '''
    Plot summary plot of raw values for each particle timestep
    Field values are interpolated to this point
    '''    
    if ylim == True:
        path = cf.anal_dir + '/field_plots_all_ylim/'
    else:
        path = cf.anal_dir + '/field_plots_all/'
        
    if os.path.exists(path) == False:                                   # Create data directory
        os.makedirs(path)
        
    plt.ioff()
    ftime, bx, by, bz, ex, ey, ez, vex, vey, vez,\
    te, jx, jy, jz, qdens, fsim_time, rD = cf.get_array(get_all=True)
    
    # Convert to plot units
    bx *= 1e9; by *= 1e9; bz *= 1e9
    ex *= 1e3; ey *= 1e3; ez *= 1e3
    qdens *= 1e-6/qi;    te /= 11603.
    
    ## Initialize plots and prepare plotspace
    fontsize = 14; fsize = 12; lpad = 20

    ## Do actual plotting and saving of data
    for ii in range(bx.shape[0]):
        if ii%skip == 0:
            
            filename = 'summ%05d.png' % ii
            fullpath = path + filename
            
            if os.path.exists(fullpath):
                continue
            
            fig, axes = plt.subplots(5, ncols=3, figsize=(20,10), sharex=True)
            fig.patch.set_facecolor('w') 
    
            axes[0, 1].set_title('Time :: {:<7.4f}s'.format(ftime[ii]), fontsize=fontsize+4, family='monospace')
    
            sys.stdout.write('\rCreating summary field plots [{}]{}'.format(run_num, ii))
            sys.stdout.flush()
            
            # Wave Fields (Plots, Labels, Lims)
            axes[0, 0].plot(cf.B_nodes / cf.dx, rD[ii], color='k', label=r'$r_D(x)$') 
            axes[1, 0].plot(cf.B_nodes / cf.dx, by[ii], color='b', label=r'$B_y$') 
            axes[2, 0].plot(cf.B_nodes / cf.dx, bz[ii], color='g', label=r'$B_z$')
            axes[3, 0].plot(cf.E_nodes / cf.dx, ey[ii], color='b', label=r'$E_y$')
            axes[4, 0].plot(cf.E_nodes / cf.dx, ez[ii], color='g', label=r'$E_z$')
            
            # Transverse Electric Field Variables (Plots, Labels, Lims)
            axes[0, 1].plot(cf.E_nodes / cf.dx, qdens[ii], color='k', label=r'$n_e$')
            axes[1, 1].plot(cf.E_nodes / cf.dx, vey[ii], color='b', label=r'$V_{ey}$')
            axes[2, 1].plot(cf.E_nodes / cf.dx, vez[ii], color='g', label=r'$V_{ez}$')
            axes[3, 1].plot(cf.E_nodes / cf.dx, jy[ii], color='b', label=r'$J_{iy}$' )
            axes[4, 1].plot(cf.E_nodes / cf.dx, jz[ii], color='g', label=r'$J_{iz}$' )
            
            # Parallel Variables (Plots, Labels, Lims)
            axes[0, 2].plot(cf.E_nodes / cf.dx, te[ii], color='k', label=r'$T_e$')
            axes[1, 2].plot(cf.E_nodes / cf.dx, vex[ii], color='r', label=r'$V_{ex}$')
            axes[2, 2].plot(cf.E_nodes / cf.dx, jx[ii], color='r', label=r'$J_{ix}$' )
            axes[3, 2].plot(cf.E_nodes / cf.dx, ex[ii], color='r', label=r'$E_x$')
            axes[4, 2].plot(cf.B_nodes / cf.dx, bx[ii], color='r', label=r'$B_{0x}$')
            
            axes[0, 0].set_title('Field outputs: {}[{}]'.format(series, run_num), fontsize=fontsize+4, family='monospace')

            axes[0, 0].set_ylabel('$r_D(x)$'     , rotation=0, labelpad=lpad, fontsize=fsize)
            axes[1, 0].set_ylabel('$B_y$\n(nT)'  , rotation=0, labelpad=lpad, fontsize=fsize)
            axes[2, 0].set_ylabel('$B_z$\n(nT)'  , rotation=0, labelpad=lpad, fontsize=fsize)
            axes[3, 0].set_ylabel('$E_y$\n(mV/m)', rotation=0, labelpad=lpad, fontsize=fsize)
            axes[4, 0].set_ylabel('$E_z$\n(mV/m)', rotation=0, labelpad=lpad, fontsize=fsize)
        
            axes[0, 1].set_ylabel('$n_e$\n$(cm^{-1})$', fontsize=fsize, rotation=0, labelpad=lpad)
            axes[1, 1].set_ylabel('$V_{ey}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
            axes[2, 1].set_ylabel('$V_{ez}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
            axes[3, 1].set_ylabel('$J_{iy}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
            axes[4, 1].set_ylabel('$J_{iz}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
            
            axes[0, 2].set_ylabel('$T_e$\n(eV)'     , fontsize=fsize, rotation=0, labelpad=lpad)
            axes[1, 2].set_ylabel('$V_{ex}$\n(m/s)' , fontsize=fsize, rotation=0, labelpad=lpad)
            axes[2, 2].set_ylabel('$J_{ix}$'        , fontsize=fsize, rotation=0, labelpad=lpad)
            axes[3, 2].set_ylabel('$E_x$\n(mV/m)'   , fontsize=fsize, rotation=0, labelpad=lpad)
            axes[4, 2].set_ylabel('$B_x$\n(nT)'     , fontsize=fsize, rotation=0, labelpad=lpad)
            fig.align_labels()
            
            for ii in range(3):
                axes[4, ii].set_xlabel('Position (m/dx)')
                for jj in range(5):
                    axes[jj, ii].set_xlim(cf.B_nodes[0] / cf.dx, cf.B_nodes[-1] / cf.dx)
                    axes[jj, ii].axvline(-cf.NX//2, c='k', ls=':', alpha=0.5)
                    axes[jj, ii].axvline( cf.NX//2, c='k', ls=':', alpha=0.5)
                    axes[jj, ii].grid()
            
            if ylim == True:
                try:
                    axes[0, 0].set_ylim(rD.min(), rD.max())
                    axes[1, 0].set_ylim(by.min(), by.max())
                    axes[2, 0].set_ylim(bz.min(), bz.max())
                    axes[3, 0].set_ylim(ey.min(), ey.max())
                    axes[4, 0].set_ylim(ez.min(), ez.max())
                    
                    axes[0, 1].set_ylim(qdens.min(), qdens.max())
                    axes[1, 1].set_ylim(vey.min(), vey.max())
                    axes[2, 1].set_ylim(vez.min(), vez.max())
                    axes[3, 1].set_ylim(jy.min() , jy.max())
                    axes[4, 1].set_ylim(jz.min() , jz.max())
                    
                    axes[0, 2].set_ylim(te.min(), te.max())
                    axes[1, 2].set_ylim(vex.min(), vex.max())
                    axes[2, 2].set_ylim(jx.min(), jx.max())
                    axes[3, 2].set_ylim(ex.min(), ex.max())
                except:
                    pass
            
            plt.tight_layout(pad=1.0, w_pad=1.8)
            fig.subplots_adjust(hspace=0.125)
    
            if save == True:
                plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close('all')
    print('\n')
    return


def analyse_particle_motion(it_max=None):
    '''
    Mainly looking at a few particles at a time to get a sense of the motion
    of these particles in a bottle/with waves
    '''
    # To Do:
    #   - Track bounce period of some hot/cold particles (maybe a handful each?)
    #   - Look at their magnetic moments with time
    savedir = cf.anal_dir + '/Sample_Particle_Motion/'
    
    if os.path.exists(savedir) == False:
        os.makedirs(savedir)

    if it_max is None:
        num_particle_steps = len(os.listdir(cf.particle_dir))
    
    ptime = np.zeros(num_particle_steps)
    np.random.seed(cf.seed)
    
    # CREATE SAMPLE ARRAY :: Either equal number from each, or just from the one
    N_samples = 5
    pos, vel, idx, sim_time, idx_start, idx_end = cf.load_particles(0)    
    
    if False:
        # Collect a sample from each species
        sloc = np.zeros((cf.Nj * N_samples), dtype=int)  # Sample location (to not confuse with particle index)
        for ii in range(cf.Nj):
            sloc[ii*N_samples: (ii + 1)*N_samples] = np.random.randint(idx_start[ii], idx_end[ii], N_samples, dtype=int)
    elif True:
        # Collect a sample from just one species
        jj   = 0
        sloc = np.random.randint(idx_start[jj], idx_end[jj], N_samples, dtype=int)
    
    ## COLLECT DATA ON THESE PARTICLES
    sidx      = np.zeros((num_particle_steps, sloc.shape[0]), dtype=int)    # Sample particle index
    spos      = np.zeros((num_particle_steps, sloc.shape[0]))            # Sample particle position
    svel      = np.zeros((num_particle_steps, sloc.shape[0], 3))            # Sample particle velocity
    
    # Load up species index and particle position, velocity for samples
    for ii in range(num_particle_steps):
        pos, vel, idx, ptime[ii], idx_start, idx_end = cf.load_particles(ii)
        print('Loading sample particle data for particle file {}'.format(ii))
        for jj in range(sloc.shape[0]):
            sidx[ii, jj]    = idx[sloc[jj]]
            spos[ii, jj]    = pos[sloc[jj]]
            svel[ii, jj, :] = vel[:, sloc[jj]]

    if True:
        # Plot position/velocity (will probably have to put a catch in here for absorbed particles: ylim?)
        plt.ioff()
        fig, axes = plt.subplots(2, sharex=True, figsize=(16, 10))
        for ii in range(sloc.shape[0]):
            axes[0].plot(ptime, spos[:, ii]   , c=cf.temp_color[sidx[0, ii]])
            axes[1].plot(ptime, svel[:, ii, 0], c=cf.temp_color[sidx[0, ii]])
            
            axes[0].set_title('Sample Positions/Velocities of Particles :: Indices {}'.format(sloc))
            axes[0].set_ylabel('Position (m)')
            axes[1].set_ylabel('Velocity (m/s)') 
            
            axes[-1].set_xlabel('Time (s)')
            
        fig.savefig(savedir + 'sample.png')
        plt.close('all')
    return


def plot_particle_loss_with_time(it_max=None, save=True):
    #   - What is the initial pitch angle of particles that have been lost?
    #   - Sum all magnetic moments to look for conservation of total mu
    #
    # 1) Plot of N particles lost vs. time (per species in color)
    # 2) Some sort of 2D plot to look at the (initial equatorial?) pitch angle of the particles lost?
    # 3) Do the mu thing, also by species?
    print('Calculating particle loss during run...')
    savedir = cf.anal_dir + '/Particle_Loss_Analysis/'
    
    if os.path.exists(savedir) == False:
        os.makedirs(savedir)
    
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
           
    ptime        = np.zeros(it_max)
    N_lost       = np.zeros((it_max, cf.Nj), dtype=int)
    
    last_pos, last_vel, last_idx, last_time = cf.load_particles(len(os.listdir(cf.particle_dir)) - 1)
    all_lost_idx, N_lost_total = locate_lost_ions(last_idx, cf.Nj)

    ## Load up species index and particle position, velocity for calculations
    for ii in range(it_max):
        pos, vel, idx, ptime[ii] = cf.load_particles(ii)

        lost_idx, N_lost[ii, :]  = locate_lost_ions(idx, cf.Nj)

    norm = np.ones(cf.Nj)

    plt.ioff()
    
    for div, suff, lbl in zip([norm, 0.01*cf.N_species], ['', '_percent'], ['N', 'Percent']): 
        fig, axes = plt.subplots(figsize=(15, 10))
        for ii in range(cf.Nj):
            axes.plot(ptime, N_lost[:, ii] / div[ii], c=cf.temp_color[ii], label=cf.species_lbl[ii])
            
        axes.set_title('# Particles Lost from Simulation :: N_species {}'.format(cf.N_species))
        axes.set_xlabel('Time (s)', fontsize=14)
        axes.set_ylabel(lbl, rotation=0, labelpad=20, fontsize=14)
            
        if save == True:
            fpath = savedir + 'particle_loss_vs_time{}.png'.format(suff)
            fig.savefig(fpath)
            plt.close('all')
            print('Saved as {}'.format(fpath))
        else:
            plt.show()
    return


@nb.njit()
def locate_lost_ions(idx, nsp):
    '''
    Checked this. Works great. Returns a 1/0 array indicating if a particular
    particle has been lost (1: Lost). Indices of these particles can be called
    via lost_indices.nonzero() (just remember species boundaries)
    N_lost is just a counter per species of how many lost particles there are.
    '''
    lost_indices = np.zeros(idx.shape[0],  dtype=nb.int64)
    N_lost       = np.zeros(nsp, dtype=nb.int64)
    for ii in range(idx.shape[0]):
        if idx[ii] < 0:
            lost_indices[ii]     = 1    # Locate in index list
            N_lost[idx[ii]+128] += 1    # Count in lost array
    return lost_indices, N_lost


def plot_initial_configurations(it_max=None, save=True, plot_lost=True):
    ## Count those that have been lost by the end of the simulation
    ## and plot that against initial distro phase spaces
    #
    ## Notes:
    ##  -- Why are lost particles only in the negative side of the simulation space?
    ##  -- Why is there seemingly no connection between lost particles and loss cone?
    print('Plotting particle initial phase space configs...')
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
    
    savedir = cf.anal_dir + '/Particle_Loss_Analysis/Initial Particle Configuration/'

    if os.path.exists(savedir) == False:                                   # Create directories
        os.makedirs(savedir)
    
    if plot_lost == True:
        final_pos, final_vel, final_idx, ptime2 = cf.load_particles(it_max-1)
        lost_indices, N_lost     = locate_lost_ions(final_idx, cf.Nj)

    init_pos , init_vel , init_idx , ptime1 = cf.load_particles(0)
    v_mag  = np.sqrt(init_vel[0] ** 2 + init_vel[1] ** 2 + init_vel[2] ** 2)
    v_perp = np.sign(init_vel[2]) * np.sqrt(init_vel[1] ** 2 + init_vel[2] ** 2)
    v_para = init_vel[0]
    
    plt.ioff()
    cf.temp_color[0] = 'c'
    
    plt.ioff()
    for jj in range(cf.Nj):
        fig1, ax1 = plt.subplots(figsize=(15, 10))
        fig2, ax2 = plt.subplots(figsize=(15, 10))
        fig3, ax3 = plt.subplots(3, sharex=True, figsize=(15, 10))
    
        lost_vals = lost_indices[cf.idx_start[jj]: cf.idx_end[jj]].nonzero()[0] + cf.idx_start[jj]

        # Loss cone diagram
        ax1.scatter(v_perp[cf.idx_start[jj]: cf.idx_end[jj]], v_para[cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
        
        if plot_lost == True:
            ax1.scatter(v_perp[lost_vals], v_para[lost_vals], c='k', marker='x', s=20, label='Lost particles')
        
        ax1.set_title('Initial Loss Cone Distribution :: {}'.format(cf.species_lbl[jj]))
        ax1.set_ylabel('$v_\parallel$ (m/s)')
        ax1.set_xlabel('$v_\perp$ (m/s)')
        ax1.legend(loc='upper right')
        
        # v_mag vs. x
        ax2.scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], v_mag[cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
        
        if plot_lost == True:
            ax2.scatter(init_pos[0, lost_vals], v_mag[lost_vals], c='k', marker='x', s=20, label='Lost particles')
        ax2.set_title('Initial Velocity vs. Position :: {}'.format(cf.species_lbl[jj]))
        ax2.set_xlabel('Position (m)')
        ax2.set_ylabel('Velocity |v| (m/s)')
        ax2.legend(loc='upper right')
            
        # v components vs. x (3 plots)
        ax3[0].scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], init_vel[0, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
        ax3[1].scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], init_vel[1, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
        ax3[2].scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], init_vel[2, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
   
        if plot_lost == True:
            ax3[0].scatter(init_pos[0, lost_vals], init_vel[0, lost_vals], c='k', marker='x', s=20, label='Lost particles')
            ax3[1].scatter(init_pos[0, lost_vals], init_vel[1, lost_vals], c='k', marker='x', s=20, label='Lost particles')
            ax3[2].scatter(init_pos[0, lost_vals], init_vel[2, lost_vals], c='k', marker='x', s=20, label='Lost particles')
        
        ax3[0].set_ylabel('$v_x$ (m/s)')
        ax3[1].set_ylabel('$v_y$ (m/s)')
        ax3[2].set_ylabel('$v_z$ (m/s)')
        
        ax3[0].set_title('Initial Velocity Components vs. Position :: {}'.format(cf.species_lbl[jj]))
        ax3[2].set_xlabel('Position (m)')
        
        for ax in ax3:
            ax.legend(loc='upper right')
            
        if save == True:
            fig1.savefig(savedir + 'loss_velocity_space_species_{}'.format(jj))
            fig2.savefig(savedir + 'loss_position_velocity_magnitude_species_{}'.format(jj))
            fig3.savefig(savedir + 'loss_position_velocity_components_species_{}'.format(jj))
 
            plt.close('all')
        else:
            plt.show()
    return


def plot_initial_configurations_loss_with_time(it_max=None, save=True, skip=1):
    ## Count those that have been lost by the end of the simulation
    ## and plot that against initial distro phase spaces
    #
    ## Notes:
    ##  -- Why are lost particles only in the negative side of the simulation space?
    ##  -- Why is there seemingly no connection between lost particles and loss cone?
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
    
    savedir = cf.anal_dir + '/Particle_Loss_Analysis/Initial Config - Loss with Time/'
    
    dir1 = savedir + '/velocity_space/'
    dir2 = savedir + '/v_mag_vs_x/'
    dir3 = savedir + '/v_components_vs_x/'
    
    for this_dir in[dir1, dir2, dir3]:
        for ii in range(cf.Nj):
            this_path = this_dir + '/species_{}/'.format(ii)
            if os.path.exists(this_path) == False:                                   # Create directories
                os.makedirs(this_path)
    
    init_pos , init_vel , init_idx , ptime1 = cf.load_particles(0)
    
    v_mag  = np.sqrt(init_vel[0] ** 2 + init_vel[1] ** 2 + init_vel[2] ** 2) / cf.va
    v_perp = np.sign(init_vel[2]) * np.sqrt(init_vel[1] ** 2 + init_vel[2] ** 2)
    v_para = init_vel[0]
    
    plt.ioff()
    cf.temp_color[0] = 'c'
    
    plt.ioff()
    for ii in range(0, it_max, skip):
        final_pos, final_vel, final_idx, ptime2 = cf.load_particles(ii)
        lost_indices, N_lost = locate_lost_ions(final_idx, cf.Nj)
        print('Plotting Init. Config. loss {}'.format(ii))
        for jj in range(cf.Nj):
            lost_vals = lost_indices[cf.idx_start[jj]: cf.idx_end[jj]].nonzero()[0] + cf.idx_start[jj]

            
            fig1, ax1 = plt.subplots(figsize=(15, 10))
            fig2, ax2 = plt.subplots(figsize=(15, 10))
            fig3, ax3 = plt.subplots(3, sharex=True, figsize=(15, 10))
        
            # Loss cone diagram
            ax1.scatter(v_perp[cf.idx_start[jj]: cf.idx_end[jj]], v_para[cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
            ax1.scatter(v_perp[lost_vals], v_para[lost_vals], c='k', marker='x', s=20, label='Lost particles')
            
            ax1.set_title('Initial Velocity Distribution :: {} :: Lost Particles at t={:.2f}s'.format(cf.species_lbl[jj], ptime2))
            ax1.set_ylabel('$v_\parallel$ (m/s)')
            ax1.set_xlabel('$v_\perp$ (m/s)')
            ax1.legend(loc='upper right')
            
            # v_mag vs. x
            ax2.scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], v_mag[cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
            ax2.scatter(init_pos[0, lost_vals], v_mag[lost_vals], c='k', marker='x', s=20, label='Lost particles')
            
            ax2.set_title('Initial Velocity vs. Position :: {} :: Lost Particles at t={:.2f}s'.format(cf.species_lbl[jj], ptime2))
            ax2.set_xlabel('Position (m)')
            ax2.set_ylabel('Velocity |v| (/vA)')
            ax2.legend(loc='upper right')
                
            # v components vs. x (3 plots)
            ax3[0].scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], init_vel[0, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
            ax3[1].scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], init_vel[1, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
            ax3[2].scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], init_vel[2, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
       
            ax3[0].scatter(init_pos[0, lost_vals], init_vel[0, lost_vals], c='k', marker='x', s=20, label='Lost particles')
            ax3[1].scatter(init_pos[0, lost_vals], init_vel[1, lost_vals], c='k', marker='x', s=20, label='Lost particles')
            ax3[2].scatter(init_pos[0, lost_vals], init_vel[2, lost_vals], c='k', marker='x', s=20, label='Lost particles')
            
            ax3[0].set_ylabel('$v_x$ (m/s)')
            ax3[1].set_ylabel('$v_y$ (m/s)')
            ax3[2].set_ylabel('$v_z$ (m/s)')
            
            ax3[0].set_title('Initial Velocity Components vs. Position :: {} :: Lost Particles at t={:.2f}s'.format(cf.species_lbl[jj], ptime2))
            ax3[2].set_xlabel('Position (m)')
            
            for ax in ax3:
                ax.legend(loc='upper right')
                
            if save == True:
                savedir1 = dir1 + '/species_{}/'.format(jj)
                savedir2 = dir2 + '/species_{}/'.format(jj)
                savedir3 = dir3 + '/species_{}/'.format(jj)
                
                fig1.savefig(savedir1 + 'loss_velocity_space_species_{}_t{:05}'.format(jj, ii))
                fig2.savefig(savedir2 + 'loss_position_velocity_magnitude_species_{}_t{:05}'.format(jj, ii))
                fig3.savefig(savedir3 + 'loss_position_velocity_components_species_{}_t{:05}'.format(jj, ii))

                plt.close('all')
            else:
                plt.show()
    return


def plot_phase_space_with_time(it_max=None, skip=1):
    ## Same plotting routines as above, just for all times, and saving output
    ## to a file
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
        
    path_cone = cf.anal_dir + '/Particle_Loss_Analysis/Config vs Time/velocity_phase_space/'
    path_mag  = cf.anal_dir + '/Particle_Loss_Analysis/Config vs Time/velocity_mag_vs_x/'
    path_comp = cf.anal_dir + '/Particle_Loss_Analysis/Config vs Time/velocity_components_vs_x/'
    
    for path in [path_cone, path_mag, path_comp]:
        if os.path.exists(path) == False:                                   # Create directories
            os.makedirs(path)
    
    v_max = 4.0 * np.sqrt(kB * cf.Tperp.max() / cf.mass[0]) / cf.va
    
    for ii in range(0, it_max, skip):
        print('Plotting phase space diagrams for particle output {}'.format(ii))
        pos, vel, idx, ptime, idx_start, idx_end = cf.load_particles(ii, shuffled_idx=True)
    
        vel   /= cf.va 
        v_mag  = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        v_perp = np.sign(vel[2]) * np.sqrt(vel[1] ** 2 + vel[2] ** 2)
        v_para = vel[0]
        
        plt.ioff()
        fig1, ax1 = plt.subplots(figsize=(15, 10))
        fig2, ax2 = plt.subplots(figsize=(15, 10))
        fig3, ax3 = plt.subplots(3, sharex=True, figsize=(15, 10))
        
        for jj in range(cf.Nj):
    
            if True:
                # Loss cone diagram
                ax1.scatter(v_perp[idx_start[jj]: idx_end[jj]], v_para[idx_start[jj]: idx_end[jj]], s=1, c=cf.temp_color[jj])
                ax1.set_title('Initial Loss Cone Distribution :: t = {:5.4f}'.format(ptime))
                ax1.set_ylabel('$v_\parallel$ (m/s)')
                ax1.set_xlabel('$v_\perp$ (m/s)')
                ax1.set_xlim(-v_max, v_max)
                ax1.set_ylim(-v_max, v_max)
            
            if False:
                # v_mag vs. x
                ax2.scatter(pos[idx_start[jj]: idx_end[jj]], v_mag[idx_start[jj]: idx_end[jj]], s=1, c=cf.temp_color[jj])       
                ax2.set_title('Initial Velocity vs. Position :: t = {:5.4f}'.format(ptime))
                ax2.set_xlabel('Position (m)')
                ax2.set_ylabel('Velocity |v| (m/s)')
                
                ax2.set_xlim(cf.xmin, cf.xmax)
                ax2.set_ylim(0, v_max)
            
            if True:
                # v components vs. x (3 plots)
                ax3[0].scatter(pos[idx_start[jj]: idx_end[jj]], vel[0, idx_start[jj]: idx_end[jj]], s=1, c=cf.temp_color[jj])
                ax3[1].scatter(pos[idx_start[jj]: idx_end[jj]], vel[1, idx_start[jj]: idx_end[jj]], s=1, c=cf.temp_color[jj])
                ax3[2].scatter(pos[idx_start[jj]: idx_end[jj]], vel[2, idx_start[jj]: idx_end[jj]], s=1, c=cf.temp_color[jj])
                
                ax3[0].set_ylabel('$v_x$ (m/s)')
                ax3[1].set_ylabel('$v_y$ (m/s)')
                ax3[2].set_ylabel('$v_z$ (m/s)')
                
                for ax in ax3:
                    ax.set_xlim(cf.xmin, cf.xmax)
                    ax.set_ylim(-v_max, v_max)
                
                ax3[0].set_title('Initial Velocity Components vs. Position :: t = {:5.4f}'.format(ptime))
                ax3[2].set_xlabel('Position (m)')
                       
        fig1.savefig(path_cone + 'cone%06d.png' % ii)
        fig2.savefig(path_mag  +  'mag%06d.png' % ii)
        fig3.savefig(path_comp + 'comp%06d.png' % ii)
        
        plt.close('all')
    return




@nb.njit(parallel=True, fastmath=True)
def interrogate_B0(Px, Py, Pz, B0_out):
    print('Interrogating B0 function')
    for ii in nb.prange(Px.shape[0]):
        for jj in range(Py.shape[0]):
            for kk in range(Pz.shape[0]):
                pos = np.array([Px[ii], Py[jj], Pz[kk]])
                bk.eval_B0_particle(pos, B0_out[ii, jj, kk, :])
    return


def plot_B0():
    savedir = cf.anal_dir + '/B0_quiver_plots_2D_slices/'
    yz_dir  = savedir + 'yz_plane//'
    xz_dir  = savedir + 'xz_plane//'
    xy_dir  = savedir + 'xy_plane//'
    
    for dpath in [yz_dir, xz_dir, xy_dir]:
        if os.path.exists(dpath) == False:                                   # Create directories
            os.makedirs(dpath)
            
    q    = 1.602177e-19                  # Elementary charge (C)
    mp   = 1.672622e-27                  # Mass of proton (kg)

    vmax = 20                                    # Maximum expected particle velocity 
    rmax = mp * vmax * cf.va / (q * cf.B_eq)     # Maximum expected Larmor radii
    
    # Sample number
    Nx = 80
    Ny = 40
    Nz = 40
    
    Px = np.linspace(cf.xmin, cf.xmax, Nx, dtype=np.float64)
    Py = np.linspace(  -rmax,    rmax, Ny, dtype=np.float64)
    Pz = np.linspace(  -rmax,    rmax, Nz, dtype=np.float64)
    
    B0_out = np.zeros((Px.shape[0], Py.shape[0], Pz.shape[0], 3), dtype=np.float64)

    interrogate_B0(Px, Py, Pz, B0_out)
    
    # Convert distances to km
    Px *= 1e-3; Py *= 1e-3; Pz *= 1e-3
    
    # Normalize vector lengths (for constant arrow shape)
    Uyz = B0_out[:, :, :, 1] / np.sqrt(B0_out[:, :, :, 1] ** 2 + B0_out[:, :, :, 2] ** 2)
    Vyz = B0_out[:, :, :, 2] / np.sqrt(B0_out[:, :, :, 1] ** 2 + B0_out[:, :, :, 2] ** 2)
    
    Uxz = B0_out[:, :, :, 0] / np.sqrt(B0_out[:, :, :, 0] ** 2 + B0_out[:, :, :, 2] ** 2)
    Vxz = B0_out[:, :, :, 2] / np.sqrt(B0_out[:, :, :, 0] ** 2 + B0_out[:, :, :, 2] ** 2)
    
    Uxy = B0_out[:, :, :, 0] / np.sqrt(B0_out[:, :, :, 0] ** 2 + B0_out[:, :, :, 1] ** 2)
    Vxy = B0_out[:, :, :, 1] / np.sqrt(B0_out[:, :, :, 0] ** 2 + B0_out[:, :, :, 1] ** 2)

    B0r   = np.sqrt(B0_out[:, :, :, 1] ** 2 + B0_out[:, :, :, 2] ** 2)*1e9
    B0xy  = np.sqrt(B0_out[:, :, :, 0] ** 2 + B0_out[:, :, :, 1] ** 2)*1e9
    B0xz  = np.sqrt(B0_out[:, :, :, 0] ** 2 + B0_out[:, :, :, 2] ** 2)*1e9
    
    plt.ioff()
    # yz slice at some x
    for ii in range(Nx):
        # Magnitude (for quiver colour)
        fig, ax = plt.subplots(figsize=(16,10))
        im1     = ax.quiver(Py, Pz, Uyz[ii, :, :].T, Vyz[ii, :, :].T, B0r[ii, :, :], clim=(B0r.min(), B0r.max()))
        
        ax.set_xlabel('y (km)')
        ax.set_ylabel('z (km)')
        ax.set_title('YZ slice at X = {:5.2f} km :: B0r vectors :: Vmax = {}vA'.format(Px[ii], vmax))
        fig.colorbar(im1).set_label('B0r (nT)')
        
        filename = 'yz_plane_%05d.png' % ii
        savepath = yz_dir + filename
        plt.savefig(savepath)
        print('yz plot {} saved'.format(ii))
        plt.close('all')
    
    # xy slice at some z
    for jj in range(Ny):
        fig, ax = plt.subplots(figsize=(16,10))
        im2     = ax.quiver(Px, Py, Uxy[:, :, jj].T, Vxy[:, :, jj].T, B0xy[:, :, jj], clim=(B0xy.min(), B0xy.max()))
        
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_title('XY slice at Z = {:5.2f} km :: B0xy vectors :: Vmax = {}vA'.format(Pz[jj], vmax))
        fig.colorbar(im2).set_label('B0xy (nT)')   
        
        filename = 'xy_plane_%05d.png' % jj
        savepath = xy_dir + filename
        plt.savefig(savepath)
        print('xy plot {} saved'.format(jj))
        plt.close('all')
    
    # xz slice at some y
    for kk in range(Nz):        
        fig, ax = plt.subplots(figsize=(16,10))
        im3 = ax.quiver(Px, Pz, Uxz[:, kk, :].T, Vxz[:, kk, :].T, B0xz[:, kk, :], clim=(B0xz.min(), B0xz.max()))
        
        ax.set_xlabel('x (km)')
        ax.set_ylabel('z (km)')
        ax.set_title('XZ slice at Y = {:5.2f} km :: B0xz vectors :: Vmax = {}vA'.format(Py[kk], vmax))
        fig.colorbar(im3).set_label('B0xz (nT)')  
        
        filename = 'xz_plane_%05d.png' % kk
        savepath = savedir + 'xz_plane//' + filename
        plt.savefig(savepath)
        print('xz plot {} saved'.format(kk))
        plt.close('all')    
    return


def plot_adiabatic_parameter(save=True):
    '''
    Change later to plot for each species charge/mass ratio, but for now its just protons
    
    What are the units for this? Does it require some sort of normalization? No, because its
    just larmor radius (mv/qB) divided by spatial length
    '''
    print('Plotting spatial adiabatic parameter...')
    max_v  = 20 * cf.va
    N_plot = 1000
    B_av   = 0.5 * (cf.B_xmax + cf.B_eq)
    z0     = cf.xmax
    
    v_perp = np.linspace(0, max_v, N_plot)
    
    epsilon = mp * v_perp / (qi * B_av * z0)
    
    plt.ioff()
    fig, ax = plt.subplots(figsize=(15, 10))
    
    ax.set_title(r'Adiabatic Parameter $\epsilon$ vs. Expected v_perp range :: {}[{}]'.format(series, run_num))
    ax.set_ylabel(r'$\epsilon$', rotation=0)
    ax.set_xlabel(r'$v_\perp (/v_A)$')
    ax.set_xlim(0, max_v/cf.va)
    ax.plot(v_perp/cf.va, epsilon)
    
    if save == True:
        fig.savefig(cf.anal_dir + 'adiabatic_parameter.png')
        print('Saved.')
        plt.close('all') 
    else:
        plt.show()
    return


def plot_pitch_and_radius_with_time(it_max=None, skip=1):
    '''
    Plots Larmor radius and pitch angle vs. space at points in time
    
    PA vs. x
    rL vs. x
    '''
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
        
    fpath = cf.anal_dir + '/Particle_Loss_Analysis/pitch_radius_vs_x/'
    
    if os.path.exists(fpath) == False:                                   # Create directories
        os.makedirs(fpath)
        
    final_pos, final_vel, final_idx, ptime2 = cf.load_particles(it_max-1)
    lost_indices, N_lost                    = locate_lost_ions(final_idx, cf.Nj)
    
    max_r = mp * 20*cf.va / (qi * cf.B_eq) * 1e-3
    
    for ii in range(0, it_max, skip):
        print('Plotting phase space diagrams for particle output {}'.format(ii))
        pos, vel, idx, ptime = cf.load_particles(ii)
        
        loss_cone  = np.arcsin(np.sqrt(cf.B_eq / cf.B_xmax))*180 / np.pi
        
        v_perp = np.sign(vel[2]) * np.sqrt(vel[1] ** 2 + vel[2] ** 2)
        v_para = vel[0]
        
        pitch  = np.arctan(v_perp / v_para) * 180. / np.pi
        radius = np.sqrt(pos[1] ** 2 + pos[2] ** 2) * 1e-3
        
        plt.ioff()
        fig, axes = plt.subplots(2, sharex=True, figsize=(15, 10))
        
        for jj in range(cf.Nj):
            lost_vals = lost_indices[cf.idx_start[jj]: cf.idx_end[jj]].nonzero()[0] + cf.idx_start[jj]
            
            if True:
                # v components vs. x (3 plots)
                axes[0].scatter(pos[0, cf.idx_start[jj]: cf.idx_end[jj]], pitch[ cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
                axes[1].scatter(pos[0, cf.idx_start[jj]: cf.idx_end[jj]], radius[cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
                
                axes[0].scatter(pos[0, lost_vals], pitch[ lost_vals], c='k', marker='x', s=20, label='Lost Particles')
                axes[1].scatter(pos[0, lost_vals], radius[lost_vals], c='k', marker='x', s=20, label='Lost Particles')
     
                axes[0].axhline(loss_cone, c='k', ls='--', alpha=0.5)
        
                axes[0].set_ylabel(r'$\alpha$ (deg)')
                axes[0].set_ylim(0, 90)
                
                axes[1].set_ylabel(r'$r_L$ (km)')
                axes[1].set_ylim(0, max_r)

                for ax in axes:
                    ax.set_xlim(cf.xmin, cf.xmax)
                
                axes[0].set_title('Particle Pitch Angles and Gyroadius vs. Position :: t = {:5.4f}'.format(ptime))
                axes[1].set_xlabel('Position (km)')
                  
        fname = 'pitchrad%06d.png' % ii
        fig.savefig(fpath + fname)
        print('Saved {}'.format(fname))
        plt.close('all')
    return


def plot_loss_paths(it_max=None, save_to_file=True):
    savedir = cf.anal_dir + 'Particle_Loss_Analysis/Lost_Particle_Paths/'
    
    if os.path.exists(savedir) == False:                                   # Create directories
        os.makedirs(savedir)
            
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))

    # Find lost particles        
    final_pos, final_vel, final_idx, ptime2 = cf.load_particles(it_max-1)
    lost_indices, N_lost                    = locate_lost_ions(final_idx, cf.Nj)
    lval                                    = lost_indices.nonzero()[0]
    
    ptime    = np.zeros(it_max) 
    lost_pos = np.zeros((it_max, 3, N_lost.sum())) 
    lost_vel = np.zeros((it_max, 3, N_lost.sum())) 
    lost_idx = np.zeros((it_max, N_lost.sum()), dtype=int) 
    
    for ii in range(it_max):
        print('Getting particle loss data from dump file {}'.format(ii))
        pos, vel, idx, ptime[ii] = cf.load_particles(ii)

        lost_pos[ii, :, :] = pos[:, lval]
        lost_vel[ii, :, :] = vel[:, lval]
        lost_idx[ii, :]    = idx[   lval]
    
    if save_to_file == True:  
        print('Saving lost particle information to file.')
        np.savez(cf.temp_dir + 'lost_particle_info', lval=lval, lost_pos=lost_pos,
                 lost_vel=lost_vel, lost_idx=lost_idx)
    
    lost_vel /= cf.va
    lost_pos /= cf.dx
    
    v_mag  = np.sqrt(lost_vel[:, 0] ** 2 + lost_vel[:, 1] ** 2 + lost_vel[:, 2] ** 2)
    v_perp = np.sign(lost_vel[:, 2]) * np.sqrt(lost_vel[:, 1] ** 2 + lost_vel[:, 2] ** 2)
    v_para = lost_vel[:, 0]
    rL     =                           np.sqrt(lost_pos[:, 1] ** 2 + lost_pos[:, 2] ** 2)
    pitch  = np.arctan(v_perp / v_para) * 180. / np.pi

    plt.ioff()
    
    for ii in range(N_lost.sum()):
        print('Plotting diagnostic outputs for particle {}'.format(lval[ii]))
        particle_path = savedir + 'pidx_%05d//' % lval[ii]
        
        if os.path.exists(particle_path) == False:                                   # Create directories
            os.makedirs(particle_path)
        
        fig1, ax1 = plt.subplots(figsize=(15, 10))
        fig2, ax2 = plt.subplots(figsize=(15, 10))
        fig3, ax3 = plt.subplots(4, sharex=True, figsize=(15, 10))
        fig4, ax4 = plt.subplots(5, sharex=True, figsize=(15, 10))
    
        # Phase space plots as above
        # But for a single (or small group of) particle/s with time
        
        # Fig 1 : Loss Cone
        if True:
            ax1.plot(v_perp[:, ii], v_para[:, ii], c=cf.temp_color[lost_idx[0, ii]], marker='o')
            ax1.set_title('Loss Cone Path :: {:07}' % lval[ii])
            ax1.set_ylabel('$v_\parallel$ (m/s)')
            ax1.set_xlabel('$v_\perp$ (m/s)')
            
            filename1 = 'velocity_space_idx_%07d.png' % lval[ii]
            fig1.savefig(particle_path + filename1)
            plt.close('all')  

        
        # Fig 2 : 2D plot of gyromotion
        if True:
            ax2.scatter(lost_pos[:, 1, ii], lost_pos[:, 2, ii], c=cf.temp_color[lost_idx[0, ii]], marker='o')
            ax2.set_title('Particle gyromotion :: {:07}' % lval[ii])
            ax2.set_ylabel('y ($dx^{-1}$)')
            ax2.set_xlabel('z ($dx^{-1}$)')
            
            filename2 = 'gyromotion_%07d.png' % lval[ii]
            fig2.savefig(particle_path + filename2)
            plt.close('all') 
        
        # Fig 3 : v_components vs. x
        if True:
            ax3[0].plot(lost_pos[:, 0, ii], lost_vel[:, 0, ii], c=cf.temp_color[lost_idx[0, ii]])
            ax3[1].plot(lost_pos[:, 0, ii], lost_vel[:, 1, ii], c=cf.temp_color[lost_idx[0, ii]])
            ax3[2].plot(lost_pos[:, 0, ii], lost_vel[:, 2, ii], c=cf.temp_color[lost_idx[0, ii]])
            ax3[3].plot(lost_pos[:, 0, ii], v_mag[:, ii], c=cf.temp_color[lost_idx[0, ii]], marker='o') 
               
            ax3[0].set_ylabel('$v_x (v_A^{-1})$')
            ax3[1].set_ylabel('$v_y (v_A^{-1})$')
            ax3[2].set_ylabel('$v_z (v_A^{-1})$')
            ax3[3].set_ylabel('$|v| (v_A^{-1})$')
            
            ax3[0].set_title('Velocity Components vs. Position :: {:07}' % lval[ii])
            ax3[3].set_xlabel('Position ($dx^{-1}$)')
            #ax3[3].set_xlim(cf.xmin / cf.dx, cf.xmax  / cf.dx)
            
            filename3 = 'vcomp_vs_x_idx_%07d.png' % lval[ii]
            fig3.savefig(particle_path + filename3)
            plt.close('all') 
        
        # Fig 4 : x, v vs. t (4-5 plot)
        if True:
            fn_idx = np.argmax(lost_idx[:, ii] < 0)
            
            ax4[0].plot(ptime, lost_pos[:, 0, ii], c=cf.temp_color[lost_idx[0, ii]])
            ax4[1].plot(ptime, lost_pos[:, 1, ii], c=cf.temp_color[lost_idx[0, ii]], label='y')
            ax4[1].plot(ptime, lost_pos[:, 2, ii], c=cf.temp_color[lost_idx[0, ii]], label='z')
            ax4[2].plot(ptime,          rL[:, ii], c=cf.temp_color[lost_idx[0, ii]])
            ax4[3].plot(ptime,      v_perp[:, ii], c=cf.temp_color[lost_idx[0, ii]])
            ax4[4].plot(ptime,       pitch[:, ii], c=cf.temp_color[lost_idx[0, ii]])

            ax4[0].set_ylabel('$x_\parallel (dx^{-1})$')
            ax4[1].set_ylabel('$x_\perp (dx^{-1})$')
            ax4[2].set_ylabel('$r_L (dx^{-1})$')
            ax4[3].set_ylabel('$v_\perp (v_A^{-1})$')
            ax4[4].set_ylabel(r'$\alpha$ (deg)')
            
            ax4[0].set_title('Position and Transverse Quantities vs. Time :: {:07}' % lval[ii])
            ax4[4].set_xlabel('Time (s)')
            ax4[4].set_xlim(0, ptime[-1])
            
            ax4[1].legend(loc='upper right')
            for ax in ax4:
                ax.axvline(ptime[fn_idx], c='k', ls='--', alpha=0.5)
            
            filename4 = 'components_vs_time_idx_%07d.png' % lval[ii]
            fig4.savefig(particle_path + filename4)
            plt.close('all') 
    return


@nb.njit()
def accrue_vector_information(inarr, idxarr, outarr, firstcomp=1, lastcomp=2):
    for kk in range(outarr.shape[1]):
        if idxarr[kk] >= 0:
            for ii in range(firstcomp, lastcomp+1):
                outarr[ii, kk] += inarr[ii, kk]
    return


def plot_average_GC(it_max=None, save=True):
    '''
    In a perfect guiding center approximation, the average of the y, z positions
    should tend towards zero (and be exactly zero at each full revolution)
    
    By adding together all the y and z components, can we get an average GC position
    for each particle? See if the lost particles are the ones furthest away from the
    origin
    '''
    print('Calculating average particle guiding center...')
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
    
    final_pos, final_vel, final_idx, ptime2 = cf.load_particles(it_max-1)
    lost_indices, N_lost = locate_lost_ions(final_idx, cf.Nj)
        
    plt.ioff()
    cf.temp_color[0] = 'c'
    
    gc_av = np.zeros((2, cf.N))
    
    plt.ioff()
    for ii in range(it_max):
        pos, vel, idx, ptime = cf.load_particles(ii)
        accrue_vector_information(pos, idx, gc_av)
        
# =============================================================================
#         for kk in range(cf.N):
#             if idx[kk] >= 0:
#                 gc_av[0, kk] += pos[1, kk]
#                 gc_av[1, kk] += pos[2, kk]
# =============================================================================
   
    gc_av *= 1e-3  / it_max
    
    plt.ioff()

    for black, suff in zip([True, False], ['_loss', '']):
        for jj in range(cf.Nj):
            fig, ax = plt.subplots(figsize=(15, 10))
            
            lost_vals = lost_indices[cf.idx_start[jj]: cf.idx_end[jj]].nonzero()[0] + cf.idx_start[jj]
            
            ax.scatter(gc_av[0, cf.idx_start[jj]: cf.idx_end[jj]], gc_av[1, cf.idx_start[jj]: cf.idx_end[jj]], c=cf.temp_color[jj], s=1)
            
            if black == True:
                ax.scatter(gc_av[0, lost_vals], gc_av[1, lost_vals], c='k', marker='x', label='Lost Particles', s=20)
            
            ax.set_title('Average gyrocenters :: {}'.format(cf.species_lbl[jj]))
            ax.set_xlabel('y (km)')
            ax.set_ylabel('z (km)')
                        
            if save == True:
                fpath = cf.anal_dir + 'particle_average_GC_sp{}{}.png'.format(jj, suff)
                fig.savefig(fpath)
                plt.close('all')
                print('Guiding centre graph saved as {}'.format(fpath))
            else:
                plt.show()
    return


@nb.njit()
def calc_mu(pos, vel, idx):
    '''
    Do for a single particle
    '''    
    ms = cf.mass[idx]
    ch = cf.charge[idx]
    v_perp2 = vel[1] ** 2 + vel[2] ** 2
    Bp      = bk.eval_B0_particle(pos, vel, ms, ch)
    B_mag   = np.sqrt(Bp[0] ** 2 + Bp[1] ** 2 + Bp[2] ** 2)

    if idx >= 0:
        mu = 0.5 * cf.mass[idx] * v_perp2 / B_mag
    else:
        mu = np.nan
    return mu, B_mag
    

def thesis_plot_mu_and_position(it_max=None, save_plot=True, save_data=True):
    '''
    Generate publication quality plot to show average mu of particles
    as well as the individual mu and locations of sample particles.
    
    Limit sample search to particles that are only within 1dx of the equator
    
    Need function to extract particle paths?
    '''
    import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    plot_path = cf.anal_dir + '//mu_and_position_SI//'
    if not os.path.exists(plot_path): os.makedirs(plot_path)
    save_data_file = cf.temp_dir + 'thesis_mu_position_data.npz'
    
    print('Collecting particle first invariants...')
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
    
    ################
    ### GET DATA ###
    ################
    if os.path.exists(save_data_file) == False or save_data == False:
        
        # Initial positions and velocities (hard coded because reasons)
        ipos, ivel, _, _, _, _ = cf.load_particles(0)
        count = np.arange(ipos.shape[0])
        print('Finding all particle indices within 1 cell of equator')
        sidx = count[np.abs(ipos) < cf.dx]
        Ns = sidx.shape[0]
        
        # Define arrays
        Bp    = np.zeros((it_max, Ns))
        mu    = np.zeros((it_max, Ns))
        pos   = np.zeros((it_max, Ns))
        vel   = np.zeros((it_max, Ns, 3))
        ptime = np.zeros(it_max)
        
        # Retrieve relevant data into arrays
        for ii in range(it_max):
            print('Loading particle information at timestep {}'.format(ii))
            this_pos, this_vel, this_idx, ptime[ii], idx_start, idx_end = cf.load_particles(ii)
            for ss in range(sidx.shape[0]):
                _idx                   = sidx[ss] 
                mu[ii, ss], Bp[ii, ss] = calc_mu(this_pos[_idx], this_vel[:, _idx], this_idx[_idx])
                pos[ii, ss]            = this_pos[_idx]
                vel[ii, ss, :]         = this_vel[:, _idx]
                
        if save_data == True and os.path.exists(save_data_file) == False:
            np.savez(save_data_file, Bp=Bp, pos=pos, vel=vel, mu=mu, ptime=ptime, sidx=sidx)
    else:
        print('Loading from file')
        data = np.load(save_data_file)
        Bp    = data['Bp']
        mu    = data['mu']
        pos   = data['pos']
        vel   = data['vel']
        ptime = data['ptime']
        sidx  = data['sidx']

    #################
    ## Do PLOTTING ##
    #################
    yfac       = 1e8
    ylabel_pad = 20
    fontsize   = 8
    tick_size  = 8
    
    mpl.rcParams['xtick.labelsize'] = tick_size 
    mpl.rcParams['ytick.labelsize'] = tick_size 

    cf.temp_color[0] = 'c'

    plt.ioff()
    for ss in range(sidx.shape[0]):
        mu_av = mu[ :, ss].mean()
        
        fig, axes = plt.subplots(2, figsize=(6.0, 4.0), sharex=True)
        
        axes[0].plot(ptime, pos[:, ss] / cf.dx)
        axes[1].plot(ptime, mu[ :, ss] * yfac)
        
        axes[0].set_ylabel('$\\frac{x}{\Delta x}$', rotation=0, fontsize=fontsize+6, labelpad=ylabel_pad)
        axes[1].set_ylabel('$\mu$\n$(\\times 10^{%d})$' % np.log10(yfac), rotation=0,
                           labelpad=ylabel_pad, fontsize=fontsize)

        axes[1].set_xlabel('Time (s)', fontsize=fontsize)
        axes[1].set_xlim(0, ptime[-1])
        axes[1].set_ylim(0.9*mu_av*yfac, 1.1*mu_av*yfac)
            
        fig.subplots_adjust(hspace=0.1)
        fig.align_ylabels()
    
        if True:
            fpath = plot_path + f'mu_position_idx_{ss:08}.png'
            fig.savefig(fpath, bbox_inches='tight', dpi=200)
            plt.close('all')
            print('1st adiabatic invariant graph for particle {} saved as {}'.format(ss, fpath))
        else:
            plt.show()
    return


def check_posvel_ortho():
    init_pos, init_vel, init_idx, ptime = cf.load_particles(0)

    dot_product = init_pos[1] * init_vel[1] + init_pos[2] * init_vel[2]
    mag_a       = np.sqrt(init_pos[1] ** 2 + init_pos[2] ** 2)
    mag_b       = np.sqrt(init_vel[1] ** 2 + init_vel[2] ** 2)
    rel_angle   = np.arccos(dot_product / (mag_a * mag_b)) * 180. / np.pi

    print('------------------')
    print('Perpendicular Position/Velocity Initiation')
    print('Maximum dot product    : {}'.format(dot_product.max()))
    print('Minimum relative angle : {}'.format(rel_angle.min()))
    print('------------------')
    return


def plot_E_components(save=False, vs_space=True, vs_time=False, E_lim=None):
    # Ji x B / qn               Convective Term
    # del(p) / qn               Ambipolar term
    # Bx(curl B) / qn*mu0       Hall Term
    fontsize=14
    
    print('Plotting E components...')
    ftime, bx, by, bz, ex, ey, ez, vex, vey, vez,\
    te, jx, jy, jz, qdens, fsim_time, rD = cf.get_array(get_all=True)
        
    hall, amb, conv = bk.calculate_E_components(bx, by, bz, jx, jy, jz, qdens)    

# =============================================================================
#     # Sanity check :: Reconstruct what E should be, compare to real
#     # Verified, don't really need it anymore
#     E_reconstruction = np.zeros((hall.shape[0], hall.shape[1], 3), dtype=float)
# 
#     E_reconstruction[:, :, 0] = - (hall[:, :, 0] + conv[:, :, 0] + amb[:, :])
#     E_reconstruction[:, :, 1] = - (hall[:, :, 1] + conv[:, :, 1])
#     E_reconstruction[:, :, 2] = - (hall[:, :, 2] + conv[:, :, 2])
# =============================================================================
    
    plt.ioff()
    
    # Plot components for each point in space (E vs. time)
    if vs_time == True:
        xtime_path  = cf.anal_dir + '/E_component_analysis/components_vs_time/'
        
        if os.path.exists(xtime_path) == False:
            os.makedirs(xtime_path)
        
        for ii in range(cf.NC):
            fig, axes = plt.subplots(4, sharex=True, figsize=(15, 10))
            
            # Ambipolar (del P) term
            axes[0].plot(ftime, amb[:, ii], c='r')
            axes[0].set_ylabel('Ambipolar')
            axes[0].set_title('E components at cell {}'.format(ii))
            
            # Convective (JxB) term :: x,y,z components
            axes[1].plot(ftime, conv[:, ii, 0], c='r', label='x')
            axes[1].plot(ftime, conv[:, ii, 1], c='b', label='y')
            axes[1].plot(ftime, conv[:, ii, 2], c='k', label='z')
            axes[1].set_ylabel('Convective')
            axes[1].legend(loc='upper right')
            
            # Hall (B x curl(B)) term :: x, y, z components
            axes[2].plot(ftime, hall[:, ii, 0], c='r', label='x')
            axes[2].plot(ftime, hall[:, ii, 1], c='b', label='y')
            axes[2].plot(ftime, hall[:, ii, 2], c='k', label='z')
            axes[2].set_ylabel('Hall')
            axes[2].legend(loc='upper right')
            
            axes[3].plot(ftime, ex[:, ii], c='r', ls='-', label='x')
            axes[3].plot(ftime, ey[:, ii], c='b', ls='-', label='y')
            axes[3].plot(ftime, ez[:, ii], c='k', ls='-', label='z')
            axes[3].set_ylabel('E')
            axes[3].legend(loc='upper right')
            axes[3].set_xlabel('Time (s)')
            
            
            if E_lim is None:
                max_E = max(ex.max(), ey.max(), ez.max())
                min_E = min(ex.min(), ey.min(), ez.min())

                axes[0].set_ylim(amb.min(), amb.max())
                axes[1].set_ylim(conv.min(), conv.max())
                axes[2].set_ylim(hall.min(), hall.max())
                axes[3].set_ylim(min_E, max_E)
            else:
                axes[0].set_ylim(-E_lim, E_lim)
                axes[1].set_ylim(-E_lim, E_lim)
                axes[2].set_ylim(-E_lim, E_lim)
                axes[3].set_ylim(-E_lim, E_lim)
                                
            fig.subplots_adjust(hspace=0.05)
            fig.align_ylabels(axes)
            
            filename = 'E_comp_vs_time_{:05}.png'.format(ii)
            savepath = xtime_path + filename
            plt.savefig(savepath, bbox_inches='tight')
            sys.stdout.write('\rE components vs. time for cell {} plotted'.format(ii))
            sys.stdout.flush()
            plt.close('all')
        print('\n')

    # Plot components for each point in time (E vs. space)
    if vs_space == True:
        xspace_path = cf.anal_dir + '/E_component_analysis/components_vs_space/'
            
        if os.path.exists(xspace_path) == False:
            os.makedirs(xspace_path)
                
        for ii in range(ftime.shape[0]):
            fig, axes = plt.subplots(4, sharex=True, figsize=(15, 10))
            
            # Ambipolar (del P) term
            axes[0].plot(cf.E_nodes/cf.dx, amb[ii, :], c='r')
            axes[0].set_ylabel('Ambipolar', labelpad=20, rotation=90, fontsize=fontsize)
            axes[0].set_title('E components (in V/m) at time {:.3f}s'.format(ftime[ii]))
            
            # Convective (JxB) term :: x,y,z components
            axes[1].plot(cf.E_nodes/cf.dx, conv[ii, :, 0], c='r', label='x')
            axes[1].plot(cf.E_nodes/cf.dx, conv[ii, :, 1], c='b', label='y')
            axes[1].plot(cf.E_nodes/cf.dx, conv[ii, :, 2], c='k', label='z')
            axes[1].set_ylabel('Convective', labelpad=20, rotation=90, fontsize=fontsize)
            axes[1].legend(loc='upper right')
            
            # Hall (B x curl(B)) term :: x, y, z components
            axes[2].plot(cf.E_nodes/cf.dx, hall[ii, :, 0], c='r', label='x')
            axes[2].plot(cf.E_nodes/cf.dx, hall[ii, :, 1], c='b', label='y')
            axes[2].plot(cf.E_nodes/cf.dx, hall[ii, :, 2], c='k', label='z')
            axes[2].set_ylabel('Hall', labelpad=20, rotation=90, fontsize=fontsize)
            axes[2].legend(loc='upper right')
            
            axes[3].plot(cf.E_nodes/cf.dx, ex[ii, :], c='r', ls='-', label='x')
            axes[3].plot(cf.E_nodes/cf.dx, ey[ii, :], c='b', ls='-', label='y')
            axes[3].plot(cf.E_nodes/cf.dx, ez[ii, :], c='k', ls='-', label='z')
            axes[3].set_ylabel('E', labelpad=20, rotation=0, fontsize=fontsize)
            axes[3].legend(loc='upper right')
            axes[3].set_xlabel('Position (cell)', fontsize=fontsize)
            axes[3].set_xlim(cf.xmin/cf.dx, cf.xmax/cf.dx)
            
            if E_lim is None:
                max_E = max(ex.max(), ey.max(), ez.max())
                min_E = min(ex.min(), ey.min(), ez.min())

                axes[0].set_ylim(amb.min(), amb.max())
                axes[1].set_ylim(conv.min(), conv.max())
                axes[2].set_ylim(hall.min(), hall.max())
                axes[3].set_ylim(min_E, max_E)
            else:
                axes[0].set_ylim(-E_lim, E_lim)
                axes[1].set_ylim(-E_lim, E_lim)
                axes[2].set_ylim(-E_lim, E_lim)
                axes[3].set_ylim(-E_lim, E_lim)
            
            fig.subplots_adjust(hspace=0.10)
            fig.align_ylabels(axes)
            
            if save == True:
                filename = 'E_components_vs_x_{:05}.png'.format(ii)
                savepath = xspace_path + filename
                plt.savefig(savepath)
                sys.stdout.write('\rE components vs. space for time {} plotted'.format(ii))
                sys.stdout.flush()
                plt.close('all')
        print('\n')  
    return


def compare_B0_to_dipole():
    '''
    To do: Calculate difference in magnetic strength along a field line, test
    how good this parabolic approx. is. Use dipole code to compare B0_x to
    B0_mu and the radial component B0_r to mod(B0_nu, B0_phi). Plot for Colin, 
    along with method Chapter.
    
    Shoji has a simulation extent on the order of R_E (0, 800 is about 6.3R_E,
    but is that symmetric?)
    
    Coded my own a based on equivalent values at +-30 degrees off equator. Maybe 
    alter code to 
    '''
    L         = 5.35       # Field line L shell
    dtheta    = 0.01       # Angle increment
    
    min_theta = np.arcsin(np.sqrt(1 / (L))) * 180 / np.pi
    
    # Calculate dipole field intensity (nT) and locations in (r, theta)
    theta = np.arange(min_theta, 180. + dtheta - min_theta, dtheta) * np.pi / 180
    r     = L * np.sin(theta) ** 2
    B_mu  = (B_surf / (r ** 3)) * np.sqrt(3*np.cos(theta)**2 + 1) * 1e9

    if False:
        # Convert to (x,y) for plotting
        x     = r * np.cos(theta)
        y     = r * np.sin(theta)
        
        plt.figure(1)
        plt.gcf().gca().add_artist(plt.Circle((0,0), 1.0, color='k', fill=False))
        
        plt.scatter(y, x, c=B_mu, s=1)
        plt.colorbar().set_label('|B| (nT)', rotation=0, labelpad=20, fontsize=14)
        plt.clim(None, 1000)
        plt.xlabel(r'x ($R_E$)', rotation=0)
        plt.ylabel(r'y ($R_E$)', rotation=0, labelpad=10)
        plt.title('Geomagnetic Field Intensity at L = {}'.format(L))
        plt.axis('equal')
        plt.axhline(0, ls=':', alpha=0.2, color='k')
        plt.axvline(0, ls=':', alpha=0.2, color='k')    
    
    else:
        # Calculate cylindrical/parabolic approximation between lat st/en
        lat_width = 30         # Latitudinal width (from equator)

        st, en = cf.boundary_idx64(theta * 180 / np.pi, 90 - lat_width, 90 + lat_width)
        
        length = 0
        for ii in range(st, en):
            length += r[ii] * dtheta * np.pi / 180
        
        RE   = 1.0
        sfac = 1.1
        z    = np.linspace(-length/2, length/2, en - st, endpoint=True) * RE
        a    = sfac / (L * RE)
        B0_z = B_mu.min() * (1 + a * z ** 2)
        
        print('Domain length : {:5.2f}RE'.format(length))
        print('Minimum field : {:5.2f}nT'.format(B_mu.min()))
        print('Maximum field : {:5.2f}nT'.format(B_mu[st:en].max()))
        print('Max/Min ratio : {:5.2f}'.format(B_mu[st:en].max() / B_mu.min()))
        
        plt.figure(2)
        plt.scatter(z/RE, B0_z,        label='Cylindrical approximation', s=4)
        plt.scatter(z/RE, B_mu[st:en], label='Dipole field intensity', s=4)
        plt.title(r'Approximation for $a = \frac{%.1f}{LR_E}$, lat. width %s deg' % (sfac, lat_width), fontsize=18)
        plt.xlabel(r'z ($R_E$)',     rotation=0, fontsize=14)
        plt.ylabel(r'$B_\parallel$', rotation=0, fontsize=14)
        plt.legend()
    return





def plot_vi_vs_t_for_cell(cell=None, comp=0, it_max=None, jj=1, save=True, hexbin=False):
    '''
    For each point in time
     - Collect particle information for particles near cell, plus time component
     - Store in array
     - Plot using either hexbin or hist2D
     
    Fix this: Make it better. Manually specify bins maybe? Or use 1D hist and compile at each timestep
                now that I know how to make sure bin limits are the same at each time
    '''
    if cell == None:
        cell = cf.NX//2
    cell  += cf.ND
    x_node = cf.E_nodes[cell]
    
    print('Calculating distribution vs. time...')
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
       
    vel_i = np.zeros(0, dtype=float)
    time  = np.zeros(0, dtype=float)
        
    # Collect all particle information for specified cell
    for ii in range(it_max):
        sys.stdout.write('\rAccruing particle data from p-file {}'.format(ii))
        sys.stdout.flush()

        pos, vel, idx, ptime= cf.load_particles(ii)
        
        f = np.zeros((0, 3))    ;   count = 0
        for ii in np.arange(cf.idx_start[jj], cf.idx_end[jj]):
            if (abs(pos[0, ii] - x_node) <= 0.5*cf.dx):
                f = np.append(f, [vel[0:3, ii]], axis=0)
                count += 1
                
        vel_i = np.append(vel_i, f[:, comp])
        time  = np.append(time, np.ones(count) * ptime)
    print('\n')

    vel_i /= cf.va

    # Do the plotting
    plt.ioff()
    
    xmin = time.min()
    xmax = time.max()
    ymin = vel_i.min()
    ymax = vel_i.max()
    
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title("F(v) vs. t :: {} :: Cell {}".format(cf.species_lbl[jj], cell))
    
    if hexbin == True:
        im1 = ax.hexbin(time, vel_i, gridsize=50, cmap='inferno')
        ax.axis([xmin, xmax, ymin, ymax])
    else:
        im1 = ax.hist2d(time, vel_i, bins=100)
               
    #cb = fig.colorbar(im1, ax=ax)
    #cb.set_label('Counts')
    if save == True:
        save_dir = cf.anal_dir + '//Particle Distribution Histograms//Time//'
        filename = 'v{}_vs_t_cell_{}_species_{}'.format(comp, cell, jj)
        
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        
        plt.savefig(save_dir + filename, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('Distribution histogram for v{}, cell {}, species {} saved'.format(comp, cell, jj))
        plt.close('all')
    else:
        plt.show()
    return


def plot_vi_vs_x(it_max=None, sp=None, save=True, shuffled_idx=False, skip=1, ppd=False):
    '''
    jj can be list of species or int
    
    For each point in time
     - Collect particle information for particles near cell, plus time component
     - Store in array
     - Plot using either hexbin or hist2D
     
    Issue : Bins along v changing depending on time (how to set max/min bins? Specify arrays manually)
    '''
    # Do checks on species specification
    if sp       is None: sp = np.arange(cf.Nj)
    if type(sp) is int:  sp = [sp]
        
    if ppd == True and os.path.exists(cf.data_dir + '//equil_particles//') == False:
            print('No equilibrium data to plot. Aborting.')
            return
        
    lt = ['x', 'y', 'z']
    
    if it_max is None:
        if ppd == False:
            it_max = len(os.listdir(cf.particle_dir))
        else:
            it_max = len(os.listdir(cf.data_dir + '//equil_particles//'))
            
    xbins = np.linspace(cf.xmin/cf.dx, cf.xmax/cf.dx, cf.NX + 1, endpoint=True)
            
    for ii in range(it_max):
        if ii%skip == 0:
            
            # Decide if continue
            sp_do = []
            for jj in sp:           
                if ppd == False:
                    save_dir = cf.anal_dir + '//Particle Spatial Distribution Histograms//Species {}//'.format(jj)
                    filename = 'fv_vs_x_species_{}_{:05}'.format(jj, ii)
                else:
                    save_dir = cf.anal_dir + '//EQUIL_Particle Spatial Distribution Histograms//Species {}//'.format(jj)
                    filename = 'EQ_fv_vs_x_species_{}_{:05}'.format(jj, ii)
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                
                if not os.path.exists(save_dir + filename + '.png'):
                    sp_do.append(jj)
            if len(sp_do) == 0: continue
        
            pos, vel, idx, ptime, idx_start, idx_end = cf.load_particles(ii, shuffled_idx=shuffled_idx,
                                                                         preparticledata=ppd)
            if cf.disable_waves:
                ptime = cf.dt_particle*ii
            for jj in sp_do:  
                if ppd == False:
                    save_dir = cf.anal_dir + '//Particle Spatial Distribution Histograms//Species {}//'.format(jj)
                    filename = 'fv_vs_x_species_{}_{:05}'.format(jj, ii)
                else:
                    save_dir = cf.anal_dir + '//EQUIL_Particle Spatial Distribution Histograms//Species {}//'.format(jj)
                    filename = 'EQ_fv_vs_x_species_{}_{:05}'.format(jj, ii)
                    
                print(f'Plotting particle data for species {jj}, time {ii}')
                    
                # Do the calculations and plotting
                cfac = 10  if cf.temp_type[jj] == 1 else 5
                vlim = 5*cf.vth_perp[jj]/cf.va

                # Manually specify bin edges for histogram
                vbins = np.linspace(-vlim, vlim, 101, endpoint=True)
                
                # Do the plotting
                plt.ioff()
                
                fig, axes = plt.subplots(3, figsize=(15, 10), sharex=True)
                axes[0].set_title('f(v) vs. x :: {} :: t = {:.3f}s'.format(cf.species_lbl[jj], ptime))
                
                st = idx_start[jj]
                en = idx_end[jj]
        
                for kk in range(3):
                    counts, xedges, yedges, im1 = axes[kk].hist2d(pos[st:en]/cf.dx, vel[kk, st:en]/cf.va, 
                                                            bins=[xbins, vbins],
                                                            vmin=0, vmax=cf.nsp_ppc[jj] / cfac)
        
                    cb = fig.colorbar(im1, ax=axes[kk], pad=0.015)
                    cb.set_label('Counts')
                    
                    axes[kk].set_ylim(-vlim, vlim)
                    axes[kk].set_ylabel('v{}\n($v_A$)'.format(lt[kk]), rotation=0)
                    
                axes[kk].set_xlim(cf.xmin/cf.dx, cf.xmax/cf.dx)
                axes[kk].set_xlabel('Position (cell)')
        
                fig.subplots_adjust(hspace=0.065)
                
                if save:
                    plt.savefig(save_dir + filename, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
                    plt.close('all')
                else:
                    plt.show()
    return


def plot_number_density(it_max=None, save=True, skip=1, ppd=False):
    '''
    jj can be list of species or int
    
    For each point in time
     - Collect particle information for particles near cell, plus time component
     - Store in array
     - Plot using either hexbin or hist2D
     
    Issue : Bins along v changing depending on time (how to set max/min bins? Specify arrays manually)
    '''
    if cf.particle_open == 1:
        shuffled_idx = True
    else:
        shuffled_idx = False
        
    if ppd == True and os.path.exists(cf.data_dir + '//equil_particles//') == False:
        print('No equilibrium data to plot. Aborting.')
        return
    
    if it_max is None:
        if ppd == False:
            it_max = len(os.listdir(cf.particle_dir))
        else:
            it_max = len(os.listdir(cf.data_dir + '//equil_particles//'))

    save_dir = cf.anal_dir + '//Particle Number Density//'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    for ii in range(it_max):
        if ii%skip == 0:     
            
            if not ppd == False:
                filename = 'species_number_density_{:05}'.format(ii)
            else:
                filename = 'EQ_species_number_density_{:05}'.format(ii)
            
            if os.path.exists(save_dir + filename + '.png'):
                print('Plot already exists.')
                continue
            else:
                print('Plotting moments for part-ts', ii)
            pos, vel, idx, ptime, idx_start, idx_end = cf.load_particles(ii, shuffled_idx=shuffled_idx,
                                                                         preparticledata=ppd)
            
            if cf.disable_waves:
                ptime = cf.dt_particle*ii
            gf_time = ptime * cf.gyfreq_xmax
            
            ni, nu = bk.get_number_densities(pos, vel, idx)
                          
            # Do the plotting
            plt.ioff()
            
            fig, axes = plt.subplots(nrows=cf.Nj, ncols=2, figsize=(16, 10), sharex=True)
            axes[0, 0].set_title('Partial Moments vs. x :: t = {:.3f}s = {:.2f} wcinv'.format(
                ptime, gf_time))
    
            # ni: (Nodes, Species)
            # nu: (Nodes, Species, Component)
            st = cf.ND
            en = cf.ND + cf.NX + 1
            for jj in range(cf.Nj):
                axes[jj, 0].plot(cf.E_nodes[st:en]/cf.dx, ni[st:en, jj], c=cf.temp_color[jj])
                axes[jj, 0].set_ylabel('$n_i$\n'+cf.species_lbl[jj], rotation=0)
            
                axes[jj, 1].plot(cf.E_nodes[st:en]/cf.dx, nu[st:en, jj, 0], c='r', label='x')
                axes[jj, 1].plot(cf.E_nodes[st:en]/cf.dx, nu[st:en, jj, 1], c='b', label='y')
                axes[jj, 1].plot(cf.E_nodes[st:en]/cf.dx, nu[st:en, jj, 2], c='k', label='z')
                axes[jj, 1].set_ylabel('$n_u$\n'+cf.species_lbl[jj], rotation=0)
                
            for ax in axes[:, 0]:
                ax.set_xlim(cf.xmin/cf.dx, cf.xmax/cf.dx)
            for ax in axes[:, 1]:
                if ax is axes[0, 1]:
                    ax.legend()
                ax.set_xlim(cf.xmin/cf.dx, cf.xmax/cf.dx)
            
            axes[-1, 0].set_xlabel('Position (cell)')
            axes[-1, 0].set_xlabel('Position (cell)')
            fig.subplots_adjust(hspace=0.065)
            
            if save:
                plt.savefig(save_dir + filename + '.png',
                            facecolor=fig.get_facecolor(),
                            edgecolor='none',
                            bbox_inches='tight')
                plt.close('all')
            else:
                plt.show()
    return

def plot_sample(it_max=None, N_sample=1000):
    print('Plotting sample of particles...')
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))

    save_dir = cf.anal_dir + '//Particle Sample//'
    
    vlim = [5, 20]
    
    if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
            
    # Get sample of existing particles
    sample_idx = np.zeros((cf.Nj, N_sample), dtype=int)
    
    pos, vel, idx, ptime, idx_start, idx_end = cf.load_particles(0)
    
    for jj in range(cf.Nj):
        sample_idx[jj] = np.random.randint(idx_start[jj], idx_end[jj], N_sample)
            
    for ii in range(it_max):
        
        filename = 'particle_sample_{:05}'.format(ii)
        
        if os.path.exists(save_dir + filename + '.png') == True:
            print('Particle data plot from p-file {} already exists.'.format(ii))
            continue
        else:
            print('Plotting particle data from p-file {}'.format(ii))
    
        pos, vel, idx, ptime, idx_start, idx_end = cf.load_particles(ii)
        
        #v_mag = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2) / cf.va
        
        # Do the plotting
        plt.ioff()
        
        fig, axes = plt.subplots(cf.Nj, figsize=(15, 10), sharex=True)
        axes[0].set_title('Particle sample :: t = {:.3f}s :: x vs. vx'.format(ptime))
        
        for jj in range(cf.Nj):
            #axes[jj].scatter(pos[0, sample_idx[jj]]/cf.dx, v_mag[sample_idx[jj]], c=cf.temp_color[jj], s=5, marker='x')
            axes[jj].scatter(pos[0, sample_idx[jj]]/cf.dx, vel[0, sample_idx[jj]], c=cf.temp_color[jj], s=5, marker='x')

            axes[jj].set_ylabel(cf.species_lbl[jj])
            axes[jj].set_ylim(0, vlim[jj])
            
        axes[-1].set_xlim(cf.xmin/cf.dx, cf.xmax/cf.dx)
        axes[-1].set_xlabel('x (dx)')

        fig.subplots_adjust(hspace=0)
        
        plt.savefig(save_dir + filename, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close('all')
    return


def scatterplot_velocities(it_max=None, skip=1):
    print('Plotting scatterplot of all particle velocities...')
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))

    save_dir = cf.anal_dir + '//velocity_scatterplots//'
    
    if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
            
    ptime_sec, pbx, pby, pbz, pex, pey, pez, pvex, pvey,\
    pvez, pte, pjx, pjy, pjz, pqdens = cf.interpolate_fields_to_particle_time(it_max)
    
    for ii in range(it_max):
        if ii%skip == 0:
            sys.stdout.write('\rPlotting particle data from p-file {}'.format(ii))
            sys.stdout.flush()
    
            filename = 'velocity_scatterplot_{:05}.png'.format(ii)
            if os.path.exists(save_dir + filename):
                print('Plot already exists, skipping...')
                continue
    
            pos, vel, idx, ptime, idx_start, idx_end = cf.load_particles(ii)
            v_perp = np.sqrt(vel[1, :] ** 2 + vel[2, :] ** 2) * np.sign(vel[2, :])
            # Do the plotting
            plt.ioff()
            
            fig, axes = plt.subplots(4, figsize=(15, 10), sharex=True)
            axes[0].set_title('All particles :: t = {:.3f}s :: x vs. vx'.format(ptime))
            
            for jj in range(cf.Nj):
                st, en = idx_start[jj], idx_end[jj]
                axes[0].scatter(pos[st:en]/cf.dx, vel[0, st:en]/cf.va, c=cf.temp_color[jj], s=1)
                axes[1].scatter(pos[st:en]/cf.dx, v_perp[st:en]/cf.va, c=cf.temp_color[jj], s=1)
            
            axes[0].set_ylabel('$v_\parallel$')
            axes[1].set_ylabel('$v_\perp$')
            
            axes[0].set_ylim(-15, 15)
            axes[1].set_ylim(-15, 15) 
            
            axes[-1].set_xlim(cf.xmin/cf.dx, cf.xmax/cf.dx)
            axes[-1].set_xlabel('x (dx)')
            
            axes[2].plot(cf.E_nodes/cf.dx, pjx[ii])
            axes[2].set_ylabel('$J_x$', rotation=0)
            axes[2].set_ylim(pjx.min(), pjx.max())
            
            axes[3].plot(cf.E_nodes/cf.dx, pqdens[ii])
            axes[3].set_ylabel('dens')
            axes[3].set_ylim(pqdens.min(), pqdens.max())
    
            fig.subplots_adjust(hspace=0)
            
            plt.savefig(save_dir + filename, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
            plt.close('all')
    print('\n')
    return


def get_closest_idx(arr, value):
    return (np.abs(arr - value)).argmin()


def get_reflection_coefficient(save=True, incl_damping_cells=True):
    '''
    Just do it in the time series
    But getting a reflection coefficient as a function of frequency could be cool.
    Would need to have a run with a chirp instead of a set pulse though. Would
    definitely need it polarised, too.
    
    Would definitely be easier with a Poynting method
    
    What if we did a frequency-based reflection coefficient: Incident J(w) vs. outgoing J(w)
    or even by spatial? Incident J(k) vs outgoing J(k)
    
    OR
    
    Pick two times and compare field energy at t1 with field energy at t2. Maybe even field
    energy vs. time?
    
    I think my indices for space/time are ass backwards.
    '''
    t_arr, bx, by, bz, ex, ey, ez, vex, vey, vez, te, jx, jy, jz, qdens, field_sim_time, damping_array\
        = cf.get_array(get_all=True)
        
    # Specify box points :: Good check!
    boundary_distance   = 50.0          # Boxes will be this far from boundary
    ti_start            = 7.0           # And start at these times
    tr_start            = 14.5
    
    # box dimensions (in units of dx and seconds)
    del_t = 7.5
    del_x = 100.0
    
    # Rectangle boundaries in data coords
    x_startL = cf.xmin/cf.dx + boundary_distance
    x_endL   = x_startL + del_x
    
    x_endR   = cf.xmax/cf.dx - boundary_distance
    x_startR = x_endR - del_x
    
    ti_end   = ti_start + del_t
    tr_end   = tr_start + del_t
    
    # Rectangle boundaries as indices
    x_arr = cf.B_nodes // cf.dx
    x1L   = get_closest_idx(x_arr, x_startL)
    x2L   = get_closest_idx(x_arr, x_endL)
    x1R   = get_closest_idx(x_arr, x_startR)
    x2R   = get_closest_idx(x_arr, x_endR)+1
    ti1   = get_closest_idx(t_arr, ti_start)
    ti2   = get_closest_idx(t_arr, ti_end)+1
    tr1   = get_closest_idx(t_arr, tr_start)
    tr2   = get_closest_idx(t_arr, tr_end)+1
    
    bt2 = (by*1e9) ** 2 + (bz*1e9) ** 2
    bt  = np.sqrt(bt2)
    
    # Magnetic field energy is simply the total summation of values at each 
    # point in space and time, within some boundaries.
    # Why am I getting wildly different values for each side?
    I_inL  = bt2[ti1:ti2, x1L:x2L].sum()
    I_outL = bt2[tr1:tr2, x1L:x2L].sum()
    reflL  = 10 * np.log10(I_outL/I_inL)
    
    I_inR  = bt2[ti1:ti2, x1R:x2R].sum()
    I_outR = bt2[tr1:tr2, x1R:x2R].sum()
    reflR  = 10 * np.log10(I_outR/I_inR)
    
    plt.ioff()
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    mpl.rcParams['ytick.labelsize'] = tick_label_size 
    fontsize = 18
    font     = 'monospace'
    
    rect1L = mpl.patches.Rectangle((x_arr[x1L], t_arr[ti1]), del_x, del_t, linewidth=1,
                                   edgecolor='w', facecolor='none')
    
    rect2L = mpl.patches.Rectangle((x_arr[x1L], t_arr[tr1]), del_x, del_t, linewidth=1,
                                   edgecolor='r', facecolor='none')
    
    rect1R = mpl.patches.Rectangle((x_arr[x1R], t_arr[ti1]), del_x, del_t, linewidth=1,
                                   edgecolor='w', facecolor='none')
    
    rect2R = mpl.patches.Rectangle((x_arr[x1R], t_arr[tr1]), del_x, del_t, linewidth=1,
                                   edgecolor='r', facecolor='none')
    
    # Plot, showing box and reflection coeff
    fig, ax = plt.subplots(1, figsize=(15, 10))

    im1  = ax.pcolormesh(x_arr, t_arr, bt, cmap='jet', vmin=0.0, vmax=bt.max())
    cb   = fig.colorbar(im1)
    
    ax.add_patch(rect1L)
    ax.add_patch(rect2L)
    ax.add_patch(rect1R)
    ax.add_patch(rect2R)
    
    cb.set_label('$|B_\perp|$\nnT', rotation=0, family=font, fontsize=fontsize, labelpad=30)

    ax.set_title('Transverse B :: Reflection Coefficients :: L {:.2f} dB :: R {:.2f} dB'.format(reflL, reflR), fontsize=fontsize, family=font)
    ax.set_ylabel('t (s)', rotation=0, labelpad=30, fontsize=fontsize, family=font)
    ax.set_xlabel('x ($\Delta x$)', fontsize=fontsize, family=font)
    ax.set_ylim(0, None)
    
    if incl_damping_cells == True:
        ax.set_xlim(x_arr[0], x_arr[-1])
    else:
        ax.set_xlim(cf.xmin/cf.dx, cf.xmax/cf.dx)
        
    ax.axvline(cf.xmin     / cf.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(cf.xmax     / cf.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(cf.grid_mid / cf.dx, c='w', ls=':', alpha=0.75)   
        
    if save == True:
        fullpath = cf.anal_dir + 'B_REFLECTION_COEFFICIENT' + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('t-x Plot saved')
        plt.close('all')
    else:
        plt.show()
    return


def field_energy_vs_time(save=False, saveas='mag_energy_reflection', tmax=None):
    '''
    Arrays are time, space. Equatorial gridpoint is at NC//2
    
    Note: Calculating reflection coefficient as max vs. min isn't entirely 
    accurate, since energy continues to leave the simulation even after 
    the wave is no longer incident on the boundary. Better to specify a before
    vs. after point (i.e. right before incident vs. right after, and indicate 
    on graph that those are the two points being used).
    
    However, each frequency has different phase velocity - can't just use the 
    same calculation point for them all.
    '''
    t, by    = cf.get_array('by')
    t, bz    = cf.get_array('bz')
    bt       = np.sqrt(by ** 2 + bz ** 2) 
    
    dt       = t[1] - t[0] 
    eq       = cf.NC//2
    st       = cf.ND; en = cf.ND + cf.NX + 1
    x        = cf.B_nodes / cf.dx
    t_cut_idx= int(4.0 / dt)

    mag_energy_L = np.zeros(t.shape[0])
    mag_energy_R = np.zeros(t.shape[0])
    mag_energy   = np.zeros(t.shape[0])
    for ii in range(t.shape[0]):
        bt_sq_L          = (by[ii,   st:eq] ** 2 + bz[ii,   st:eq] ** 2)
        bt_sq_R          = (by[ii, eq+1:en] ** 2 + bz[ii, eq+1:en] ** 2)
        bt_sq            = (by[ii,   st:en] ** 2 + bz[ii,   st:en] ** 2)
        
        mag_energy_L[ii] = (0.5 / mu0) * bt_sq_L.sum() * (cf.NX //2) * cf.dx
        mag_energy_R[ii] = (0.5 / mu0) * bt_sq_R.sum() * (cf.NX //2) * cf.dx
        mag_energy[  ii] = (0.5 / mu0) * bt_sq.sum()   * (cf.NX //2) * cf.dx
    
    L_refl = 100. * mag_energy_L[t_cut_idx:].min() / mag_energy_L.max()
    R_refl = 100. * mag_energy_R[t_cut_idx:].min() / mag_energy_R.max()

    if not save:
        plt.ioff()
        
        ## PLOT IT
        fontsize = 18
        font     = 'monospace'
        
        tick_label_size = 14
        mpl.rcParams['xtick.labelsize'] = tick_label_size 
        mpl.rcParams['ytick.labelsize'] = tick_label_size 
        
        fig, [axl, ax, axr] = plt.subplots(nrows=1, ncols=3, figsize=(15, 10),
                                                 sharey=True, gridspec_kw={'width_ratios':[1,2,1]})
    
        im1 = ax.pcolormesh(x, t, bt, cmap='jet', vmin=0.0)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='5%', pad=0.40)
        cb  = fig.colorbar(im1, cax=cax, orientation='horizontal')
        cax.yaxis.set_ticks_position('right')
        cb.set_label('$B_\perp$ nT', rotation=0, family=font, fontsize=fontsize, labelpad=-70)
        
        axl.set_ylabel('t (s)', rotation=0, labelpad=30, fontsize=fontsize, family=font)
        ax.set_xlabel('x ($\Delta x$)', fontsize=fontsize, family=font)
        ax.set_ylim(0, tmax)
        
        ax.set_xlim(x[0], x[-1])
        ax.axvline(cf.xmin     / cf.dx, c='w', ls=':', alpha=1.0)
        ax.axvline(cf.xmax     / cf.dx, c='w', ls=':', alpha=1.0)
        ax.axvline(   0.0      / cf.dx, c='w', ls=':', alpha=0.75)   
        
        # Plot energy
        axl.plot(mag_energy_L/mag_energy_L.max(), t, c='b')
        axl.set_xlabel('$U_B$ Left\nNormalized', rotation=0, fontsize=fontsize, family=font)
        axl.set_xlim(0.01, 1)
        #axl.set_title('$\Gamma_L = %.2f$%%' % L_refl)
        axl.invert_xaxis()
        
        axr.plot(mag_energy_R/mag_energy_R.max(), t, c='b')
        axr.set_xlabel('$U_B$ Right\nNormalized', rotation=0, fontsize=fontsize, family=font)
        axr.set_xlim(0.01, 1)
        #axr.set_title('$\Gamma_R = %.2f$%%' % R_refl)
        #axl.invert_xaxis()
                
        fig.subplots_adjust(wspace=0)
        
        fullpath = cf.anal_dir + saveas + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
        print('t-x Plot saved')
        plt.close('all')

    return t, mag_energy


def plot_max_velocity(save=True):
    
    num_particle_steps = len(os.listdir(cf.particle_dir))
    
    # Load up species index and particle position, velocity for samples
    ptime  = np.zeros(num_particle_steps)
    max_vx = np.zeros(num_particle_steps)
    max_vy = np.zeros(num_particle_steps)
    max_vz = np.zeros(num_particle_steps)
    for ii in range(num_particle_steps):
        print('Loading sample particle data for particle file {}'.format(ii))
        pos, vel, idx, ptime[ii], id1, id2 = cf.load_particles(ii)
        
        max_vx[ii] = np.abs(vel[0]).max()
        max_vy[ii] = np.abs(vel[1]).max()
        max_vz[ii] = np.abs(vel[2]).max()
        
    plt.ioff()
    fig, ax = plt.subplots()
    ax.set_title('Maximum particle velocity in simulation')
    ax.plot(ptime, max_vx, label='$v_x$', c='k')
    ax.plot(ptime, max_vy, label='$v_y$', c='r', alpha=0.5, lw=0.5)
    ax.plot(ptime, max_vz, label='$v_z$', c='b', alpha=0.5, lw=0.5)
    ax.axhline(max_vx[0], color='k', ls='--', alpha=0.5)
    ax.legend()
    
    if save == True:
        fullpath = cf.anal_dir + 'max_velocity_check' + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
        print('t-x Plot saved')
        plt.close('all')
    else:
        plt.show()
    return


def plot_abs_with_boundary_parameters(tmax=None, saveas='tx_boundaries_plot', save=True, B0_lim=None):
    '''
    B0_lim is in multiples of B0
    '''
    plt.ioff()

    t, by    = cf.get_array('by')
    t, bz    = cf.get_array('bz')
    t, jx    = cf.get_array('jx')
    t, ex    = cf.get_array('ex')
    t, qdens = cf.get_array('qdens')
    x        = cf.B_nodes / cf.dx
    bt       = np.sqrt(by ** 2 + bz ** 2)*1e9
    
    fontsize = 18
    font     = 'monospace'
    
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    mpl.rcParams['ytick.labelsize'] = tick_label_size 
    
    ## PLOT IT
    fig, [axl, ax, axr] = plt.subplots(nrows=1, ncols=3, figsize=(15, 10),
                                             sharey=True, gridspec_kw={'width_ratios':[1,2,1]})
    
    axl2 = axl.twiny()    
    vmin = 0.0
    
    if B0_lim is None:
        vmax   = bt.max()
    else:
        vmax   = cf.B_eq * B0_lim * 1e9
        
    im1 = ax.pcolormesh(x, t, bt, cmap='nipy_spectral', vmin=vmin, vmax=vmax)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.40)
    cb  = fig.colorbar(im1, cax=cax, orientation='horizontal')
    cax.yaxis.set_ticks_position('right')
    cb.set_label('$B_\perp$ nT', rotation=0, family=font, fontsize=fontsize, labelpad=-70)
    
    axl.set_ylabel('t (s)', rotation=0, labelpad=30, fontsize=fontsize, family=font)
    ax.set_xlabel('x ($\Delta x$)', fontsize=fontsize, family=font)
    ax.set_ylim(0, tmax)
    
    ax.set_xlim(x[0], x[-1])
    ax.axvline(cf.xmin     / cf.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(cf.xmax     / cf.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(cf.grid_mid / cf.dx, c='w', ls=':', alpha=0.75)   
    
    # Plot fluxes
    LHS = cf.ND;    RHS = cf.ND + cf.NX - 1
    axl.plot(jx[:, LHS]*1e6, t, c='b')
    axl.set_xlabel('$J_x$\nLH Boundary', rotation=0, fontsize=fontsize, family=font)
    
    axr.plot(jx[:, RHS]*1e6, t, c='b')
    axr.set_xlabel('$J_x$\nRH Boundary', rotation=0, fontsize=fontsize, family=font)
    
    av_density = qdens.mean(axis=1)/(qi*cf.ne)
    
    # Plot average density and Ex
    axl2.plot(av_density, t, c='r')
    axl2.set_xlim(0, 1.2*av_density.max())
    axl2.set_title('Av. $\\rho_c$', fontsize=fontsize, family=font, color='r')
    axl2.xaxis.label.set_color('r')
    axl2.tick_params(axis='x', colors='r')
    
    fig.subplots_adjust(wspace=0)
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
        print('t-x Plot saved')
        plt.close('all')
    return


def plot_total_density_with_time(save=True):
    '''
    Plot total number of particles per species (from particle files)
    AND total charge density from field files.
    '''
    # Count active particles
    num_particle_steps = len(os.listdir(cf.particle_dir))
    ptime              = np.zeros(num_particle_steps)
    num_idx            = np.zeros((num_particle_steps, cf.Nj))
    
    for ii in range(num_particle_steps):
        print('Loading sample particle data for particle file {}'.format(ii))
        pos, vel, idx, ptime[ii], id1, id2 = cf.load_particles(ii)
        
        spidx, counts = np.unique(idx, return_counts=True)
        if cf.particle_open == 1:
            # +1 accounts for idx starting at -1 when sorted
            for jj in range(cf.Nj):
                num_idx[ii, jj] = counts[jj + 1]
        else:
            # +1 accounts for idx starting at -1 when sorted
            for jj in range(cf.Nj):
                num_idx[ii, jj] = counts[jj]
          
    # Get max density (real)
    ftime, qdens = cf.get_array('qdens')    
    max_dens = np.zeros(ftime.shape[0])
    for ii in range(ftime.shape[0]):
        max_dens[ii] = qdens[ii].sum()
        
    # Plot
    plt.ioff()
    fig, axes = plt.subplots(2, sharex=True, figsize=(16, 10))
    for jj in range(cf.Nj):
        axes[0].plot(ptime, num_idx[:, jj]/num_idx[0, jj], label=cf.species_lbl[jj], c=cf.temp_color[jj])
    axes[0].set_ylabel('Normalized Particle count')
    axes[0].legend()
    
    axes[1].plot(ftime, max_dens, c='k')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Total Density')
    
    axes[0].set_xlim(0, ptime[-1])
    
    if save == True:
        fullpath = cf.anal_dir + 'Particle_Density_Count' + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
        print('pc-t Plot saved')
        plt.close('all')
    else:
        plt.show()
    return


def multiplot_fluxes(series, save=True):
    '''
    Load outside loop: Fluxes and charge density of all runs in a series
    '''
    print('Plotting fluxes for multiple series...')
    series_dir  = '{}/runs//{}//'.format(drive, series)
    num_runs    = len([name for name in os.listdir(series_dir) if 'run_' in name])
    runs_to_do  = range(num_runs)
    
    clrs = ['k', 'b', 'g', 'r', 'c', 'm', 'y',
            'darkorange', 'peru', 'yellow']         
    
    plt.ioff()
    fig1, axes1 = plt.subplots(2, sharex=True, figsize=(15, 10))
    fig2, axes2 = plt.subplots(2, sharex=True, figsize=(15, 10))
    for run_num in runs_to_do:
        print('\nRun {}'.format(run_num))
        cf.load_run(drive, series, run_num, extract_arrays=True)
        
        # Quick & Easy way to do all runs: Plot function here
        ftime, bx, by, bz, ex, ey, ez, vex, vey, vez, te, jx, jy, jz, \
        qdens, field_sim_time, damping_array = cf.get_array(get_all=True)
        qn0 = cf.ne * qi
        LHS = cf.ND
        # Density plot at boundary cell
        axes1[0].set_title('Flux and density at LHS boundary')
        axes1[0].plot(ftime, qdens[:, LHS], c=clrs[run_num], label=str(cf.nsp_ppc[1])+' hppc')
        axes1[0].set_ylabel('$\\rho_c$', rotation=0, fontsize=14, labelpad=30)
        axes1[0].legend()
        axes1[0].axhline(qn0, c='k', ls=':', alpha=0.5)
        
        # Parallel current plot (proxy for total flux)
        axes1[1].plot(ftime, jx[:, LHS], c=clrs[run_num], label=str(cf.nsp_ppc[1])+' hppc')
        axes1[1].set_ylabel('$J_x$', rotation=0, fontsize=14, labelpad=30)
        axes1[1].set_xlabel('Time (s)', fontsize=12)
        axes1[1].set_xlim(0, None)
        axes1[1].axhline(0.0, c='k', ls=':', alpha=0.5)
        
        RHS = cf.ND + cf.NX - 1
        # Density plot at boundary cell
        axes2[0].set_title('Flux and density at RHS boundary')
        axes2[0].plot(ftime, qdens[:, RHS], c=clrs[run_num], label=str(cf.nsp_ppc[1])+' hppc')
        axes2[0].set_ylabel('$\\rho_c$', rotation=0, fontsize=14, labelpad=30)
        axes2[0].legend()
        axes2[0].axhline(qn0, c='k', ls=':', alpha=0.5)
        
        # Parallel current plot (proxy for total flux)
        axes2[1].plot(ftime, jx[:, RHS], c=clrs[run_num], label=str(cf.nsp_ppc[1])+' hppc')
        axes2[1].set_ylabel('$J_x$', rotation=0, fontsize=14, labelpad=30)
        axes2[1].set_xlabel('Time (s)', fontsize=12)
        axes2[1].set_xlim(0, None)
        axes2[1].axhline(0.0, c='k', ls=':', alpha=0.5)
        
        if save==True:
            fig1.savefig(series_dir + 'LHS_moments.png', facecolor=fig1.get_facecolor(), edgecolor='none', bbox_inches='tight')
            fig2.savefig(series_dir + 'RHS_moments.png', facecolor=fig1.get_facecolor(), edgecolor='none', bbox_inches='tight')
            print('Boundary moment plots saved')
            plt.close('all')
        else:
            plt.show()
    return


def multiplot_mag_energy(save=False):
    print('Plotting magnetic field energy for multiple series...')
    import matplotlib.cm as cm
    
    resis_fracs = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00]
    norm        = mpl.colors.Normalize(vmin=min(resis_fracs), vmax=max(resis_fracs), clip=False)
    mapper      = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    
    series = 'Fu_resist_test'
    series_dir  = '{}/runs//{}//'.format(drive, series)
    num_runs    = len([name for name in os.listdir(series_dir) if 'run_' in name])
    print('{} runs in series {}'.format(num_runs, series))
    
    # Plot the thing:
    plt.ioff()
    fig, ax = plt.subplots(figsize=(15, 10))
                
    for run_num, eta in zip(range(num_runs), resis_fracs):
        print('\nRun {}'.format(run_num))
        cf.load_run(drive, series, run_num, extract_arrays=True)
        time, mag_energy = field_energy_vs_time(save=False)

        ax.plot(time, mag_energy, c=mapper.to_rgba(eta))
        
        ax.set_xlim(0, time[-1])
        ax.set_ylim(0, 0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized $U_B$')
        ax.set_title('Total Wave Energy vs. Time for different Resistivities')
        #ax.set_yscale('log')
        
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="2%", pad=0.00)
    cbar    = fig.colorbar(mapper, cax=cax, label='Flux', orientation='vertical')
    cbar.set_label('Resistivity Fraction')
        
    if save == True:
        fig.savefig(series_dir + 'mag_energy.png', facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('Reflection coefficient plots saved')
        plt.close('all')
    else:
        print('Showing plot...')
        plt.show()
        sys.exit()
    return


def multiplot_parallel_scaling():
    # Only specific for this run, because nb.get_num_threads() didn't work
    print('Plotting parallel scaling for multiple series...')
    series  = 'parallel_time_test'
    series_dir  = '{}/runs//{}//'.format(drive, series)
    num_runs    = len([name for name in os.listdir(series_dir) if 'run_' in name])
    print('{} runs in series {}'.format(num_runs, series))
    
    n_threads   = np.zeros(num_runs)
    loop_times  = np.zeros(num_runs)
    total_times = np.zeros(num_runs)
    
    runs_to_do = range(num_runs)                
    for ii in runs_to_do:
        print('\nRun {}'.format(ii))
        cf.load_run(drive, series, ii, extract_arrays=True)
        n_particles = cf.N / 1e6
        num_inc     = int(round(cf.max_rev * cf.gyperiod / cf.dt_sim))               # Total runtime in seconds
        
        n_threads[ii]   = 2 ** ii         
        loop_times[ii]  = cf.loop_time
        total_times[ii] = cf.run_time
        
    plt.ioff()
    # Plot 1: threads vs. total time
    plt.figure()
    plt.plot(n_threads, total_times, c='k')
    plt.xlabel('Threads')
    plt.ylabel('Total Runtime (s)')
    plt.title('Parallel Scaling Test :: {} mil. particles :: {} iterations'.format(n_particles, num_inc))
    
        
    # Plot 2: threads vs. loop time
    plt.figure()
    plt.plot(n_threads, loop_times, c='k')
    plt.xlabel('Threads')
    plt.ylabel('Average Loop Time (s)')
    plt.title('Parallel Scaling Test :: {} mil. particles :: {} iterations'.format(n_particles, num_inc))
    plt.show()
    return


def multiplot_saturation_amplitudes_jan16(save=True):
    '''
    Plots the saturation (maximum) amplitude of each run series as a function of
    'time' (which probably has to be hardcoded)
    '''
    jan_times = ['2015-01-16T04:29:15.000000',
            '2015-01-16T04:31:45.000000',
            '2015-01-16T04:34:55.000000',
            '2015-01-16T04:38:45.000000',
            '2015-01-16T04:41:50.000000',
            '2015-01-16T04:43:30.000000',
            '2015-01-16T04:47:10.000000',
            '2015-01-16T04:50:35.000000',
            '2015-01-16T04:53:00.000000',
            '2015-01-16T05:00:30.000000',
            '2015-01-16T05:03:10.000000',
            '2015-01-16T05:04:30.000000',
            '2015-01-16T05:07:30.000000']
    
    savefile  = 'F://runs//JAN16_PKTS_30HE_PC//' + 'sat_amps_jan16.npz'
    fieldfile = 'F://runs//JAN16_PKTS_30HE_PC//' + 'field_vals_jan16.npz'
    all_series  = ['//JAN16_PKTS_5HE_PC//', '//JAN16_PKTS_15HE_PC//', '//JAN16_PKTS_30HE_PC//',]
    
    if os.path.exists(savefile) and False:
        print('Loading saturation amplitudes from file...')
        DR_file   = np.load(savefile)
        sat_amp   = DR_file['sat_amp']
    else:
        print('Plotting saturation amplitudes for multiple series...')
        # Get number of runs (assume same for each series) to set array
        num_runs  = 13 #len([name for name in os.listdir(test_dir) if 'run_' in name])
        sat_amp   = np.zeros((len(all_series), num_runs), dtype=np.float64)
        
        print('Collecting saturation amplitudes')
        for jj in range(len(all_series)):
            series      = all_series[jj]
            series_dir  = '{}/runs//{}//'.format(drive, series)
            num_runs    = len([name for name in os.listdir(series_dir) if 'run_' in name])
            print('{} runs in series {}'.format(num_runs, series))
        
            runs_to_do = range(num_runs)                
            for ii in runs_to_do:
                print('\nRun {}'.format(ii))
                #try:
                cf.load_run(drive, series, ii, extract_arrays=True)
                
                ty, by = cf.get_array('By') 
                tz, bz = cf.get_array('Bz')
                
                env_amps = np.sqrt(by ** 2 + bz ** 2)
                sat_amp[jj, ii] = env_amps.max() * 1e9
                #except:
                #    print(f'Something wrong with run {ii}')
                #    sat_amp[jj, ii] = np.nan
        print('Saving saturation amplitudes to file...')
        np.savez(savefile, sat_amp=sat_amp)
    
    timebase = np.array([np.datetime64(this_time) for this_time in jan_times])
    
    # Load comparison data
    data_scripts_dir = 'D://Google Drive//Uni//PhD 2017//Data//Scripts//'
    sys.path.append(data_scripts_dir)
    import rbsp_fields_loader as rfl
    import fast_scripts as fscr
    import analysis_scripts as ascr
    
    rbsp_path  = 'E://DATA//RBSP//'
    time_start = np.datetime64('2015-01-16T04:25:00.000000')
    time_end   = np.datetime64('2015-01-16T05:10:00.000000')
    probe       = 'a'
    
    if not os.path.exists(fieldfile):
        dat_times, pc1_mags, HM_mags, delt = \
             rfl.load_decomposed_magnetic_field(rbsp_path, time_start, time_end, probe, 
                                        pad=600, LP_B0=1.0, LP_HM=30.0, 
                                        get_gyfreqs=False, return_B0=False)
             
        spec_time, spec_freq, spec_power = fscr.get_pc1_tracepower_spectra(dat_times, pc1_mags,
                                 time_start, time_end, 
                                  _dt=delt, _olap=0.95, _res=15.0,
                                  window_data=True)
            
        fst, fen = ascr.boundary_idx64(spec_freq, 0.1, 0.3)
        pc1_int_power = spec_power[fst:fen, :].sum(axis=0)
        
        pc1_env = ascr.extract_envelope(dat_times, pc1_mags, delt, 100., 300., include_z=False)
        
        np.savez(fieldfile, spec_time=spec_time, spec_freq=spec_freq, spec_power=spec_power,
                            pc1_int_power=pc1_int_power, dat_times=dat_times, pc1_env=pc1_env)
    else:
        print('Loading field values from file...')
        FLD_file     = np.load(fieldfile)
        spec_time    = FLD_file['spec_time']
        spec_freq    = FLD_file['spec_freq']
        spec_power   = FLD_file['spec_power']
        pc1_int_power= FLD_file['pc1_int_power']
        dat_times    = FLD_file['dat_times']
        pc1_env      = FLD_file['pc1_env']
    
    # Plot result -- 3 plots: spectra, envelope (or integrated power? both?)
    #                then scatterplot with each run type in different color (for he concentration)
    plt.ioff()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.0, 0.7*3.71),
                             gridspec_kw={'width_ratios':[1, 0.3]})
    
# =============================================================================
#     # Pc1 Spectra
#     
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         im = axes[0, 0].pcolormesh(spec_time, spec_freq, spec_power,
#                        norm=colors.LogNorm(vmin=1e-5, vmax=1e0), cmap='jet',
#                        shading='auto')
#         fig.colorbar(im, cax=axes[0, 1], extend='max').set_label(
#                 '$Tr(P)$\n$nT^2/Hz$', fontsize=12, rotation=0, labelpad=30)
#         axes[0, 0].set_ylabel('f\nHz', rotation=0, labelpad=20)
#         axes[0, 0].set_ylim(0.0, 0.6)
#         axes[0, 0].set_xlim(time_start, time_end)
#         axes[0, 0].set_xticklabels([])
#             
#     
# =============================================================================
    
    # Hybrid Results
    #axes[0].set_title('Saturation Amplitudes :: Jan 16, 2015')
    axes[0].plot(dat_times, pc1_env, c='k', lw=0.75, alpha=0.75)
    
    clrs = ['blue', 'green', 'red']
    for ii, lbl in zip(range(len(all_series)), ['5% He', '15% He', '30% He']):
        axes[0].scatter(timebase, sat_amp[ii], c=clrs[ii], label=lbl)
    axes[0].set_xlim(time_start, time_end)
    axes[0].set_ylim(0.0, None)
    axes[0].set_xlabel('Time (UT)')
    axes[0].set_ylabel('$|B_w|$\n(nT)', rotation=0, labelpad=20)
    axes[0].legend(loc='upper right')
    leg = axes[0].legend(bbox_to_anchor=(1.0, 0.0), loc=3, borderaxespad=0., prop={'size': 12})
    leg.get_frame().set_linewidth(0.0)
    
    for xx in range(2):
        for this_time in timebase:
            axes[xx].axvline(this_time, c='k', ls='--', alpha=0.75)
    
    #axes[0, 0].set_xticklabels([])
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    axes[1].set_visible(False)
    
    fig.tight_layout()
    fig.align_ylabels()
    fig.subplots_adjust(hspace=0.0, wspace=0.01)
                            
    if save==True:  
        fig.savefig('F://runs//JAN16_PKTS_30HE_PC//' + 'data_model_saturation.png',
                    edgecolor='none', dpi=200)
        plt.close('all')
    else:
        plt.show()
    return


def multiplot_saturation_amplitudes_jul17(save=True):
    '''
    Plots the saturation (maximum) amplitude of each run series as a function of
    'time' (which probably has to be hardcoded)
    '''
    jul_times = ['1991-07-17T20:22:11.000000',
     '1991-07-17T20:26:30.000000',
     '1991-07-17T20:37:43.000000',
     '1991-07-17T20:42:16.000000',
     '1991-07-17T20:47:41.000000',
     '1991-07-17T20:50:55.000000',
     '1991-07-17T20:54:19.000000']
    
    tlim = [None, 220., 270., 360., 300., 280., 260.]
    savefile  = 'F://runs//JUL17_PC1PEAKS_VO_1pc//' + 'sat_amps_jul16.npz'
    fieldfile = 'F://runs//JUL17_PC1PEAKS_VO_1pc//' + 'field_vals_jul16.npz'
    all_series  = ['//JUL17_PC1PEAKS_VO_1pc//']
    
    if os.path.exists(savefile) and False:
        print('Loading saturation amplitudes from file...')
        DR_file   = np.load(savefile)
        sat_amp   = DR_file['sat_amp']
    else:
        print('Plotting saturation amplitudes for multiple series...')
        # Get number of runs (assume same for each series) to set array
        num_runs  = 7 #len([name for name in os.listdir(test_dir) if 'run_' in name])
        sat_amp   = np.zeros((len(all_series), num_runs), dtype=np.float64)
        
        print('Collecting saturation amplitudes')
        for jj in range(len(all_series)):
            series      = all_series[jj]
            series_dir  = '{}/runs//{}//'.format(drive, series)
            num_runs    = len([name for name in os.listdir(series_dir) if 'run_' in name])
            print('{} runs in series {}'.format(num_runs, series))
        
            runs_to_do = range(num_runs)                
            for ii in runs_to_do:
                    print('\nRun {}'.format(ii))
                #try:
                    cf.load_run(drive, series, ii, extract_arrays=True)
                    
                    ty, by = cf.get_array('By') 
                    tz, bz = cf.get_array('Bz')

                    if tlim[ii] is not None:
                        tmax_idx = np.where(np.abs(ty - tlim[ii]) == np.abs(ty - tlim[ii]).min())[0][0]
                    else:
                        tmax_idx = None
                    
                    env_amps = np.sqrt(by[:tmax_idx] ** 2 + bz[:tmax_idx] ** 2)
                    sat_amp[jj, ii] = env_amps.max() * 1e9
                #except:
                 #   print(f'Something wrong with run {ii}')
                  #  sat_amp[jj, ii] = np.nan
        print('Saving saturation amplitudes to file...')
        np.savez(savefile, sat_amp=sat_amp)
    
    timebase = np.array([np.datetime64(this_time) for this_time in jul_times])
    
    # Load comparison data
    data_scripts_dir = 'D://Google Drive//Uni//PhD 2017//Data//Scripts//'
    sys.path.append(data_scripts_dir)
    import crres_file_readers as cfr
    import fast_scripts as fscr
    import analysis_scripts as ascr
    
    crres_path  = 'E://DATA//CRRES//'
    time_start = np.datetime64('1991-07-17T20:15:00.000000')
    time_end   = np.datetime64('1991-07-17T21:00:00.000000')
    
    if not os.path.exists(fieldfile):
        dat_times, B0, HM_mags, pc1_mags, \
        E0, HM_elec, pc1_elec, S, B, E = cfr.get_crres_fields(crres_path, time_start, time_end,
                                        pad=1800, output_raw_B=True, Pc5_LP=30.0, B0_LP=1.0,
                                        Pc5_HP=None, dEx_LP=None, interpolate_nan=True)
        delt = 1/32.
        
        pc1_env = ascr.extract_envelope(dat_times, pc1_mags, delt, 100., 300., include_z=False)
        
        np.savez(fieldfile, dat_times=dat_times, pc1_env=pc1_env)
    else:
        print('Loading field values from file...')
        FLD_file     = np.load(fieldfile)
        dat_times    = FLD_file['dat_times']
        pc1_env      = FLD_file['pc1_env']
    
    # Plot result -- 3 plots: spectra, envelope (or integrated power? both?)
    #                then scatterplot with each run type in different color (for he concentration)
    plt.ioff()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.0, 0.7*3.71),
                             gridspec_kw={'width_ratios':[1, 0.3]})

    # Hybrid Results
    axes[0].set_title('Saturation Amplitudes :: Jan 16, 2015')
    axes[0].plot(dat_times, pc1_env, c='k', lw=0.75, alpha=0.75)
    
    clrs = ['blue', 'green', 'red']
    for ii, lbl in zip(range(len(all_series)), ['1% He']):
        axes[0].scatter(timebase, sat_amp[ii], c=clrs[ii], label=lbl)
    axes[0].set_xlim(time_start, time_end)
    axes[0].set_ylim(0.0, None)
    axes[0].set_xlabel('Time (UT)')
    axes[0].set_ylabel('$|B_w|$\n(nT)', rotation=0, labelpad=20)
    axes[0].legend(loc='upper right')
    leg = axes[0].legend(bbox_to_anchor=(1.0, 0.0), loc=3, borderaxespad=0., prop={'size': 12})
    leg.get_frame().set_linewidth(0.0)
    
    for xx in range(2):
        for this_time in timebase:
            axes[xx].axvline(this_time, c='k', ls='--', alpha=0.50, lw=0.75)
    
    #axes[0, 0].set_xticklabels([])
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    axes[1].set_visible(False)
    
    fig.tight_layout()
    fig.align_ylabels()
    fig.subplots_adjust(hspace=0.0, wspace=0.01)
                            
    if save==True:  
        fig.savefig('F://runs//JUL17_PC1PEAKS_VO_1pc//' + 'data_model_saturation.png',
                    edgecolor='none', dpi=200)
        plt.close('all')
    else:
        plt.show()
    return


def multiplot_RMS_jul25():
    '''    
    Collect maximum amplitude and RMS, just because maximum can be sullied by
    peaks in the waveform that are unphysical/numerical noise/issues
    
    TODO: Need some sort of spatial filter based on dx?
    
    Only collect for one series at first
    '''
    print('Plotting saturation amplitudes for multiple series...')

    all_series = ['JUL25_PROXYHONLY_30HE_PREDCORR']
    jul_times  = ['2013-07-25T21:25:29.000000',
                  '2013-07-25T21:26:15.000000',
                  '2013-07-25T21:27:27.000000',
                  '2013-07-25T21:29:38.000000',
                  '2013-07-25T21:30:50.000000',
                  '2013-07-25T21:31:55.000000',
                  '2013-07-25T21:32:41.000000',
                  '2013-07-25T21:33:53.000000',
                  '2013-07-25T21:36:30.000000',
                  '2013-07-25T21:37:42.000000',
                  '2013-07-25T21:39:07.000000',
                  '2013-07-25T21:40:13.000000',
                  '2013-07-25T21:41:05.000000',
                  '2013-07-25T21:42:11.000000',
                  '2013-07-25T21:43:16.000000',
                  '2013-07-25T21:44:02.000000',
                  '2013-07-25T21:45:33.000000',]
    timebase = np.array([np.datetime64(this_time) for this_time in jul_times])
    
    # Get number of runs (assume same for each series) to set array
    test_dir  = '{}/runs//{}//'.format(drive, all_series[0])
    num_runs  = len([name for name in os.listdir(test_dir) if 'run_' in name])
    max_rms   = np.zeros(num_runs, dtype=np.float64)
    max_env   = np.zeros(num_runs, dtype=np.float64)
    sat_amp   = np.zeros(num_runs, dtype=np.float64)
    
    print('Collecting RMS amplitudes')
    for jj in range(len(all_series)):
        series      = all_series[jj]
        series_dir  = '{}/runs//{}//'.format(drive, series)
        num_runs    = len([name for name in os.listdir(series_dir) if 'run_' in name])
        print('{} runs in series {}'.format(num_runs, series))
        
        runs_to_do = range(num_runs)                
        for ii in runs_to_do:
            print('\nRun {}'.format(ii))
            cf.load_run(drive, series, ii, extract_arrays=True)
            
            times, by   = cf.get_array('By') 
            times, bz   = cf.get_array('Bz')
            env_amps = np.sqrt(by ** 2 + bz ** 2)
            
            # For each time, get the RMS amplitude in space
            # Can we take the RMS values for each time and just pythag them?
            by_rms = np.zeros(times.shape[0], dtype=float)
            bz_rms = np.zeros(times.shape[0], dtype=float)
            for tt in range(times.shape[0]):
                by_rms[tt] = np.sqrt(np.square(by[tt]).mean())
                bz_rms[tt] = np.sqrt(np.square(bz[tt]).mean())
            
            rms_amps = np.sqrt(by_rms ** 2 + bz_rms ** 2)
            max_rms[ii] = rms_amps.max() * 1e9
            
            env_amps = np.sqrt(by**2 + bz**2)
            max_env[ii] = env_amps.max() * 1e9
    
    plt.figure()
    plt.scatter(timebase, max_env,            c='k', label='max env')
    plt.scatter(timebase, max_rms*np.sqrt(2), c='b', label='max RMS*sqrt(2)')
    plt.legend()
    plt.show()
    
    if False:
        # Load comparison data
        data_scripts_dir = 'D://Google Drive//Uni//PhD 2017//Data//Scripts//'
        sys.path.append(data_scripts_dir)
        import rbsp_fields_loader as rfl
        import fast_scripts as fscr
        import analysis_scripts as ascr
        
        rbsp_path  = 'E://DATA//RBSP//'
        time_start = np.datetime64('2013-07-25T21:25:00.000000')
        time_end   = np.datetime64('2013-07-25T21:47:00.000000')
        probe       = 'a'
        
        dat_times, pc1_mags, HM_mags, delt = \
             rfl.load_decomposed_magnetic_field(rbsp_path, time_start, time_end, probe, 
                                        pad=600, LP_B0=1.0, LP_HM=30.0, 
                                        get_gyfreqs=False, return_B0=False)
             
        spec_time, spec_freq, spec_power = fscr.get_pc1_tracepower_spectra(dat_times, pc1_mags,
                                 time_start, time_end, 
                                  _dt=delt, _olap=0.95, _res=35.0,
                                  window_data=True)
            
        fst, fen = ascr.boundary_idx64(spec_freq, 0.2, 0.8)
        pc1_int_power = spec_power[fst:fen, :].sum(axis=0)
        
        # Plot result -- 3 plots: spectra, envelope (or integrated power? both?)
        #                then scatterplot with each run type in different color (for he concentration)
        plt.ioff()
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 9),
                                 gridspec_kw={'width_ratios':[1, 0.01]})
        
        # Pc1 Spectra
        axes[0, 0].set_title('Data/Simulation Comparison :: July 25, 2013')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = axes[0, 0].pcolormesh(spec_time, spec_freq, spec_power,
                           norm=colors.LogNorm(vmin=1e-7, vmax=1e1), cmap='jet')
            fig.colorbar(im, cax=axes[0, 1], extend='max').set_label(
                    '$Tr(P)$\n$nT^2/Hz$', fontsize=12, rotation=0, labelpad=30)
            axes[0, 0].set_ylabel('Hz', rotation=0, labelpad=30)
            axes[0, 0].set_ylim(0.0, 1.1)
            axes[0, 0].set_xlim(time_start, time_end)
            axes[0, 0].set_xticklabels([])
            
        # Integrated Power
        axes[1, 1].set_visible(False)
        axes[1, 0].plot(spec_time, pc1_int_power)
        axes[1, 0].set_xlim(time_start, time_end)
        axes[1, 0].set_xticklabels([])
        axes[1, 0].set_ylim(0.0, None)
        axes[1, 0].set_ylabel('Integrated Power (nT^2/Hz)', rotation=0, labelpad=30)
        
        for xx in range(2):
            for this_time in timebase:
                axes[xx, 0].axvline(this_time, c='k', ls='--', alpha=0.75)
        
        # Hybrid Results
        clrs = ['blue', 'green', 'red']
        for ii, lbl in zip(range(len(all_series)), ['5% He', '15% He', '30% He']):
            axes[2, 0].scatter(timebase, sat_amp[ii], c=clrs[ii], label=lbl)
        axes[2, 1].set_visible(False)
        axes[2, 0].set_xlim(time_start, time_end)
        axes[2, 0].set_ylim(0.0, None)
        axes[2, 0].set_xlabel('Time (UT)')
        axes[2, 0].set_ylabel('Saturation Amplitude (nT)')
        axes[2, 0].legend(loc='upper right')
# =============================================================================
#         if save==True:  
#             fig.savefig(cf.base_dir + 'data_model_saturation.png', edgecolor='none')
#             plt.close('all')
#         else:
#             plt.show()
# =============================================================================
    return


def multiplot_AUG12_density(save=True, fmax=1.0):
    '''
    Create figure

    Saturation amplitudes vs. space to show control waveforms, 2x2 /w cbar
    Dotted line to show where ULF changes
    Do the 5x run because has the strongest saturation amplitude (for visual)
    '''
    plot_dir = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//REVISION_PLOTS//Model Output'
    
    series  = 'AUG12_DROP_HIGHER'
    ffamily = 'monospace'
    fsize   = 10
    lpad    = 15
    cpad    = 30
    
    Bmax = 5.0
    Pmax = 1e-1
    
    ## PLOT
    plt.ioff()
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6.0, 4.0), 
                             gridspec_kw={'width_ratios':[1, 1, 0.05]})
        
    col = 0
    for ii in [2, 10]:              
        print('\nRun {}'.format(ii))
        cf.load_run(drive, series, ii, extract_arrays=True)
            
        # Get wx data and axes
        ftime, by_wx = disp.get_wx('By', fac=1e9)
        ftime, bz_wx = disp.get_wx('Bz', fac=1e9)
        wx = by_wx + bz_wx
        
        freqs = np.fft.rfftfreq(ftime.shape[0], d=cf.dt_field)
        x_arr = cf.B_nodes * 1e-6
        x_lim = np.array([cf.xmin, cf.xmax]) * 1e-6
    
        # Get xt
        ftime, By_raw = cf.get_array('by')
        ftime, Bz_raw = cf.get_array('bz')
        B_wave = np.sqrt(By_raw**2 + Bz_raw**2)*1e9
        
        # XT WAVE PLOT
        im1 = axes[0, col].pcolormesh(x_arr, ftime, B_wave, cmap='jet', vmin=0.0, vmax=Bmax)
        
        axes[0, col].set_xlabel('x (Mm)', fontsize=fsize, family=ffamily)
        axes[0, col].set_ylim(0, ftime[-1])
        axes[0, col].set_xlim(x_lim[0], x_lim[1])    
        axes[0, col].set_xticklabels([])
        
        # WX DISPERSION PLOT
        im2 = axes[1, col].pcolormesh(x_arr, freqs, wx, cmap='nipy_spectral', vmin=0.0, vmax=Pmax)
        
        axes[1, col].set_xlabel('x (Mm)', fontsize=fsize, family=ffamily)
        axes[1, col].set_ylim(0, fmax)
        axes[1, col].set_xlim(x_lim[0], x_lim[1])
        col += 1
    
    axes[0, 0].set_ylabel('t\n(s)', rotation=0, labelpad=lpad, fontsize=fsize, family=ffamily)
    axes[0, 0].set_yticks([50, 150, 250])
    
    axes[1, 0].set_ylabel('f\n(Hz)', rotation=0, labelpad=lpad, fontsize=fsize, family=ffamily)
    axes[1, 0].set_yticks([0.0, 0.4, 0.8])
    
    axes[0, 1].set_yticklabels([])
    axes[1, 1].set_yticklabels([])
    
    axes[0, 0].tick_params(axis='y', labelsize=8)
    axes[1, 0].tick_params(axis='y', labelsize=8)
    axes[1, 0].tick_params(axis='x', labelsize=8)
    axes[1, 1].tick_params(axis='x', labelsize=8)
    
    # COLORBARS
    cbar1 = fig.colorbar(im1, cax=axes[0, 2], extend='max')
    cbar1.set_label('nT', rotation=0,
                    family=ffamily, fontsize=fsize, labelpad=20)
    
    cbar2 = fig.colorbar(im2, cax=axes[1, 2], extend='max')
    cbar2.set_label('$\\frac{nT^2}{Hz}$', rotation=0,
                    family=ffamily, fontsize=fsize+2, labelpad=20)
    
    cbar1.ax.tick_params(labelsize=8) 
    cbar2.ax.tick_params(labelsize=8) 
        
    fig.tight_layout()
    fig.align_ylabels()
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
                   
    if save:
        fullpath = plot_dir + 'aug12_fxt_plot.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(),
                    edgecolor='none', bbox_inches='tight', dpi=200)
        print('Dispersion stackplot saved')
        plt.close('all')
    else:
        plt.show()       
    return


def multiplot_AUG12_waveform(save=True, fmax=1.0):
    '''
    Create figure

    Saturation amplitudes vs. space to show control waveforms, 2x2 /w cbar
    Dotted line to show where ULF changes
    Do the 5x run because has the strongest saturation amplitude (for visual)
    '''
    plot_dir = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//REVISION_PLOTS//Model Output//'
    
    series  = 'AUG12_DROP_HIGHER'
    ffamily = 'monospace'
    fsize   = 10
    lpad    = 15
    cpad    = 30
    
    Bmax = 5.0
    Pmax = 1e-1
    
    B0 = 145.00e-9; gy0 = qi * B0 / (2*np.pi*mp)
    B1 =  95.00e-9; gy1 = qi * B1 / (2*np.pi*mp)
    
    ## PLOT
    plt.ioff()
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6.0, 4.0), 
                             gridspec_kw={'width_ratios':[1, 1, 0.05]})
        
    col = 0
    for ii in [2, 10]:              
        print('\nRun {}'.format(ii))
        cf.load_run(drive, series, ii, extract_arrays=True)
            
        # Get wx data and axes
        ftime, by_wx = disp.get_wx('By', fac=1e9)
        ftime, bz_wx = disp.get_wx('Bz', fac=1e9)
        wx = by_wx + bz_wx
        
        freqs = np.fft.rfftfreq(ftime.shape[0], d=cf.dt_field)
        x_arr = cf.B_nodes * 1e-6
        x_lim = np.array([cf.xmin, cf.xmax]) * 1e-6
    
        # Get xt
        ftime, By_raw = cf.get_array('by')
        ftime, Bz_raw = cf.get_array('bz')
        B_wave = np.sqrt(By_raw**2 + Bz_raw**2)*1e9
        
        # XT WAVE PLOT
        im1 = axes[0, col].pcolormesh(x_arr, ftime, B_wave, cmap='jet', vmin=0.0, vmax=Bmax)
        
        axes[0, col].set_xlabel('x (Mm)', fontsize=fsize, family=ffamily)
        axes[0, col].set_ylim(0, ftime[-1])
        axes[0, col].set_xlim(x_lim[0], x_lim[1])    
        axes[0, col].set_xticklabels([])
        
        # WX DISPERSION PLOT
        im2 = axes[1, col].pcolormesh(x_arr, freqs, wx, cmap='nipy_spectral', vmin=0.0, vmax=Pmax)
        
        axes[1, col].set_xlabel('x (Mm)', fontsize=fsize, family=ffamily)
        axes[1, col].set_ylim(0, fmax)
        axes[1, col].set_xlim(x_lim[0], x_lim[1])
        
        if ii == 2:
            axes[1, col].axhline(0.2500*gy0, lw=0.75, c='yellow', label='$f_{cHe^+}$')
            axes[1, col].axhline(0.0625*gy0, lw=0.75, c='r', label='$f_{cO^+}$')
           
        else:
            axes[1, col].axhline(0.2500*gy0, lw=0.75, ls='--', c='yellow', label='$f_{cHe^+}$')
            axes[1, col].axhline(0.0625*gy0, lw=0.75, ls='--', c='r', label='$f_{cO^+}$')
            
            axes[1, col].axhline(0.2500*gy1, lw=0.75, c='yellow', label='$f_{cHe^+}$')
            axes[1, col].axhline(0.0625*gy1, lw=0.75, c='r', label='$f_{cO^+}$')
        
        col += 1
    
    axes[1, 0].legend(loc='upper left', ncol=2)
    
    axes[0, 0].set_ylabel('t\n(s)', rotation=0, labelpad=lpad, fontsize=fsize, family=ffamily)
    axes[0, 0].set_yticks([50, 150, 250])
    
    axes[1, 0].set_ylabel('f\n(Hz)', rotation=0, labelpad=lpad, fontsize=fsize, family=ffamily)
    axes[1, 0].set_yticks([0.0, 0.4, 0.8])
    
    axes[0, 1].set_yticklabels([])
    axes[1, 1].set_yticklabels([])
    
    axes[0, 0].tick_params(axis='y', labelsize=8)
    axes[1, 0].tick_params(axis='y', labelsize=8)
    axes[1, 0].tick_params(axis='x', labelsize=8)
    axes[1, 1].tick_params(axis='x', labelsize=8)
    
    # COLORBARS
    cbar1 = fig.colorbar(im1, cax=axes[0, 2], extend='max')
    cbar1.set_label('nT', rotation=0,
                    family=ffamily, fontsize=fsize, labelpad=20)
    
    cbar2 = fig.colorbar(im2, cax=axes[1, 2], extend='max')
    cbar2.set_label('$\\frac{nT^2}{Hz}$', rotation=0,
                    family=ffamily, fontsize=fsize+2, labelpad=20)
    
    cbar1.ax.tick_params(labelsize=8) 
    cbar2.ax.tick_params(labelsize=8) 
        
    fig.tight_layout()
    fig.align_ylabels()
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
                   
    if save:
        fullpath = plot_dir + 'aug12_fxt_plot.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(),
                    edgecolor='none', bbox_inches='tight', dpi=200)
        print(f'AUG12 dispersion stackplot saved in {fullpath}')
        plt.close('all')
    else:
        plt.show()       
    return


def multiplot_AUG12_Benergy(save=True):
    '''
    Examine B energy vs. time for AUG12 runs
    AUG12_DROP :: Runs 0 (control, x1), 8 (drop, x1)
    AUG12_DROP_HIGHER :: Runs 0, 2 (controlx2, x5), 8, 10 (drop x2, x5)
    
    Load run
    Calculate energy vs. time
    Plot
    Second plot will have ULF (take from last run)
    
    Loop through control first
    Then through ULF
    Use same color for each RC concentration
    '''
    plot_dir = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//'
    
    ffamily = 'monospace'
    fsize   = 10
    lpad    = 20
        
    ## PLOT
    plt.ioff()
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.0, 4.0), sharex=True)
    
    rc_clr = ['blue', 'red']
    rc_lbl = ['$n_h = 0.2 cm^{-3}$', '$n_h = 0.4 cm^{-3}$']         # Change to cc later
    
    sA = 'AUG12_DROP'; sB = 'AUG12_DROP_HIGHER'
    all_series = [sA, sB]
    run_nums   = [0, 0]
    
    # Control ones: Make thicker?
    for series, run, clr, lbl in zip(all_series, run_nums, rc_clr, rc_lbl):              
        print('\nRun {}'.format(run))
        cf.load_run(drive, series, run, extract_arrays=True)

        # (time, space)
        ftime, By_raw = cf.get_array('by')
        ftime, Bz_raw = cf.get_array('bz')
        B_wave = np.sqrt(By_raw**2 + Bz_raw**2)*1e9
        
        mag_energy = np.zeros(ftime.shape[0])
        for ii in range(ftime.shape[0]):
            #mag_energy[ii] = (0.5 / mu0) * np.square(B_wave[ii]).sum() * cf.dx
            mag_energy[ii] = np.sqrt(np.square(B_wave[ii]).mean())
        
        # Energy lineplot
        axes[0].plot(ftime, mag_energy, c=clr, label=lbl, lw=0.75, ls='--')
    axes[0].legend()
    
    all_series = [sA, sB]
    run_nums   = [8, 8]
    # ULF ones: Can be thin?
    for series, run, clr, lbl in zip(all_series, run_nums, rc_clr, rc_lbl):              
        print('\nRun {}'.format(run))
        cf.load_run(drive, series, run, extract_arrays=True)
        
        # (time, space)
        ftime, By_raw = cf.get_array('by')
        ftime, Bz_raw = cf.get_array('bz')
        B_wave = np.sqrt(By_raw**2 + Bz_raw**2)*1e9
        
        mag_energy = np.zeros(ftime.shape[0])
        for ii in range(ftime.shape[0]):
            #mag_energy[ii] = (0.5 / mu0) * np.square(B_wave[ii]).sum() * cf.dx
            mag_energy[ii] = np.sqrt(np.square(B_wave[ii]).mean())
        
        # Energy lineplot
        axes[0].plot(ftime, mag_energy, c=clr, label=lbl, lw=0.75)
    
    # ULF plot
    t, bxc = cf.get_array(component='bxc', get_all=False, timebase=None)
    bxc = bxc[:, cf.NX//2]*1e9
    axes[1].plot(ftime, bxc, c='b', label=lbl)
    
    # Set labels and limits
    axes[0].set_ylabel('RMS($B_w$)\n(nT)', rotation=0, labelpad=lpad, fontsize=fsize, family=ffamily)
    axes[1].set_ylabel('$B_0$\n$(nT)$', rotation=0, labelpad=lpad, fontsize=fsize, family=ffamily)
    axes[1].set_xlabel('t (s)', fontsize=fsize, family=ffamily)
    
    axes[0].set_xlim(125, ftime[-1])
    axes[1].set_xlim(125, ftime[-1])
    #axes[0].set_xticklabels([])
    #axes[0].set_ylim(1e-8, 1e-5)
    axes[1].set_ylim(0.9*bxc.min(), 1.1*bxc.max())
        
    #axes[0, 0].tick_params(axis='y', labelsize=8)
    #axes[1, 0].tick_params(axis='y', labelsize=8)
    #axes[1, 0].tick_params(axis='x', labelsize=8)
    #axes[1, 1].tick_params(axis='x', labelsize=8)
        
    fig.tight_layout()
    fig.align_ylabels()
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
                   
    if True:
        fullpath = plot_dir + 'aug12_Benergy_plot.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(),
                    edgecolor='none', bbox_inches='tight', dpi=200)
        print('Dispersion stackplot saved')
        plt.close('all')
    else:
        plt.show()
            
    return


def multiplot_AUG12_Benergy_times(save=True):
    '''
    Examine B energy vs. time for AUG12 runs
    AUG12_DROP :: Runs 0 (control, x1), 8 (drop, x1)
    AUG12_DROP_HIGHER :: Runs 0, 2 (controlx2, x5), 8, 10 (drop x2, x5)
    
    Load run
    Calculate energy vs. time
    Plot
    Second plot will have ULF (take from last run)
    
    Loop through control first
    Then through ULF
    Use same color for each RC concentration
    '''
    plot_dir = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//'
    
    ffamily = 'monospace'
    fsize   = 10
    lpad    = 20
        
    ## PLOT
    plt.ioff()
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.0, 4.0), sharex=True)
    
    rc_clr = ['black', 'blue', 'red']
    rc_lbl = ['N/A', '100s', '200s']         # Change to cc later
    
    run_nums   = [0, 4, 8]
    
    # Control ones: Make thicker?
    for run, clr, lbl in zip(run_nums, rc_clr, rc_lbl):              
        print('\nRun {}'.format(run))
        cf.load_run(drive, 'AUG12_DROP_HIGHER', run, extract_arrays=True)

        # (time, space)
        ftime, By_raw = cf.get_array('by')
        ftime, Bz_raw = cf.get_array('bz')
        B_wave = np.sqrt(By_raw**2 + Bz_raw**2)*1e9
        
        mag_energy = np.zeros(ftime.shape[0])
        for ii in range(ftime.shape[0]):
            #mag_energy[ii] = (0.5 / mu0) * np.square(B_wave[ii]).sum() * cf.dx
            mag_energy[ii] = np.sqrt(np.square(B_wave[ii]).mean())
        
        # Energy lineplot
        axes[0].plot(ftime, mag_energy, c=clr, label=lbl, lw=0.75, ls='-')
        
        # ULF plot
        t, bxc = cf.get_array(component='bxc', get_all=False, timebase=None)
        bxc = bxc[:, cf.NX//2]*1e9
        axes[1].plot(ftime, bxc, c=clr, label=lbl)
    
    #axes[0].legend()
    
    # Set labels and limits
    axes[0].set_ylabel('RMS($B_w$)\n(nT)', rotation=0, labelpad=lpad, fontsize=fsize, family=ffamily)
    axes[1].set_ylabel('$B_0$\n$(nT)$', rotation=0, labelpad=lpad, fontsize=fsize, family=ffamily)
    axes[1].set_xlabel('t (s)', fontsize=fsize, family=ffamily)
    
    tlim = 0
    axes[0].set_xlim(tlim, ftime[-1])
    axes[1].set_xlim(tlim, ftime[-1])
    #axes[0].set_xticklabels([])
    #axes[0].set_ylim(1e-8, 1e-5)
    axes[1].set_ylim(0.9*bxc.min(), 1.1*bxc.max())
        
    #axes[0, 0].tick_params(axis='y', labelsize=8)
    #axes[1, 0].tick_params(axis='y', labelsize=8)
    #axes[1, 0].tick_params(axis='x', labelsize=8)
    #axes[1, 1].tick_params(axis='x', labelsize=8)
        
    fig.tight_layout()
    fig.align_ylabels()
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
                   
    if True:
        fullpath = plot_dir + 'aug12_Benergy_times_plot.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(),
                    edgecolor='none', bbox_inches='tight', dpi=200)
        print('Dispersion stackplot saved')
        plt.close('all')
    else:
        plt.show()
            
    return


def multiplot_wk_thesis_good(saveas='wk_plot_thesis', save=True):
    '''
    Just do magnetic field, make axes big enough to see.
    
    Also need to put CPDR on these
    '''
    from matplotlib.ticker import FormatStrFormatter
    
    fontsize = 10
    font     = 'monospace'
    mpl.rcParams['xtick.labelsize'] = 8 
    mpl.rcParams['ytick.labelsize'] = 8 
    
    plt.ioff()
    fig = plt.figure(figsize=(6.0, 4.0))
    gs  = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 0.05])
    cax = fig.add_subplot(gs[:, 2])
    axes = []
    
    # Load and Calculate dispersion relations for all runs
    run = 0
    for mm in range(2):
        for nn in range(2):
            cf.load_run('E:', 'CH4_tests_noQS_2048', run, extract_arrays=True); run += 1
            k, f, wky,     tf = disp.get_wk('by', linear_only=False, norm_z=True, centre_only=False)
            k, f, wkz,     tf = disp.get_wk('bz', linear_only=False, norm_z=True, centre_only=False)
            wk_perp = wky + wkz
            
            k_vals, CPDR_solns, WPDR_solns, HPDR_solns = disp.get_linear_dispersion_from_sim(k, zero_cold=True)

            ax = fig.add_subplot(gs[mm, nn]) 
            axes.append(ax)
    
            xfac = c / cf.wpi
            yfac = 2*np.pi / cf.gyfreq
            cyc  = 1.0 / np.array([1., 4., 16.])
        
            im = ax.pcolormesh(xfac*k[1:], yfac*f[1:], wk_perp[1:, 1:].real, cmap='jet',
                                norm=colors.LogNorm(vmin=1e-5, vmax=1e4))
            
            for ii in range(3):
                if cf.species_present[ii] == True:
                    ax.axhline(cyc[ii], linestyle=':', c='k', alpha=0.75, lw=0.75)

            for ii in range(CPDR_solns.shape[1]):
                ax.plot(xfac*k_vals, yfac*CPDR_solns[:, ii].real, c='k', linestyle='--',
                        label='CPDR' if ii == 0 else '', alpha=0.75, lw=0.75)
                
            ax.set_xlim(0, 0.8)
            ax.set_ylim(0, 1.05)
    
    axes[0].set_ylabel('$\\frac{\omega}{\Omega_H}$', fontsize=fontsize+4, family=font, rotation=0, labelpad=20)
    axes[2].set_ylabel('$\\frac{\omega}{\Omega_H}$', fontsize=fontsize+4, family=font, rotation=0, labelpad=20)
    axes[2].set_xlabel('$\mathtt{kc/\omega_{pi}}$' , fontsize=fontsize, family=font)
    axes[3].set_xlabel('$\mathtt{kc/\omega_{pi}}$' , fontsize=fontsize, family=font)
    
    axes[1].set_yticklabels([])
    axes[3].set_yticklabels([])
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[2].set_xticks([0.0, 0.25, 0.5, 0.75])
    axes[3].set_xticks([0.0, 0.25, 0.5, 0.75])
    
    cbar = fig.colorbar(im, cax=cax, extend='both')
    #cbar.set_label('Pwr\n$\left(\\frac{nT^2}{Hz}\\right)$',
    #    rotation=0, fontsize=fontsize, family=font, labelpad=30)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    if save == True:
        plot_dir = 'E://runs//CH4_tests_noQS_2048//'

        fullpath1  = plot_dir + 'hybrid_CPDR_validation.png'
            
        fig.savefig(fullpath1, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=200)

        print('w-k for B-field saved')
        plt.close('all')
    else:
        plt.show()
    return


def multiplot_ABCs_thesis():
    '''
    Plot the homogeneous above the parabolic, then plot each separately on a half page. Just
    to have the pick.
    Runs 5, 6 of CH4_tests_v2 on E:
    '''
    fontsize = 10
    font     = 'monospace'
    mpl.rcParams['xtick.labelsize'] = 8 
    mpl.rcParams['ytick.labelsize'] = 8 
    
    plt.ioff()
    fig_all  = plt.figure(figsize=(6.0, 4.0))
    gs_all   = GridSpec(2, 2, figure=fig_all, width_ratios=[1.0, 0.05])
    cax_all  = fig_all.add_subplot(gs_all[:, 1])
    
    # Load and Calculate dispersion relations for all runs
    _xx = 0; axes = []
    for run in [5, 6]:
        cf.load_run('E:', 'CH4_tests_v2', run, extract_arrays=True)
        
        t, by_arr = cf.get_array('by')
        t, bz_arr = cf.get_array('bz')
        bt_arr    = np.sqrt(by_arr ** 2 + bz_arr ** 2)*1e9

        x_arr = cf.B_nodes / cf.dx
        
        ## PLOT IT (on its own)
        fig, ax = plt.subplots(1, figsize=(6.0, 4.0))
        
        vmin = 0.0
        vmax = bt_arr.max()
        im1  = ax.pcolormesh(x_arr, t, bt_arr, cmap='jet', vmin=vmin, vmax=vmax)
        
        cb  = fig.colorbar(im1)        
        cb.set_label('nT', rotation=0, family=font, fontsize=fontsize, labelpad=20)

        ax.set_ylabel('t\n(s)', rotation=0, labelpad=20, fontsize=fontsize, family=font)
        ax.set_xlabel('$x/\Delta x$', fontsize=fontsize, family=font)
        ax.set_ylim(0, t.max())
        
        #ax.axvline(cf.xmin     / cf.dx, c='w', ls=':', alpha=1.0)
        #ax.axvline(cf.xmax     / cf.dx, c='w', ls=':', alpha=1.0)
        ax.axvline(0.0                , c='w', ls=':', alpha=0.75)   
        ax.set_xlim(x_arr[0], x_arr[-1]) 
        
        if True:
            fullpath = cf.anal_dir + 'bt_thesis' + '.png'
            plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=200)
            print('t-x Plot saved')
            
        ## ADD TO COMBINED PLOT
        this_ax = fig_all.add_subplot(gs_all[_xx, 0]); _xx += 1 
        
        im1a  = this_ax.pcolormesh(x_arr, t, bt_arr, cmap='jet', vmin=vmin, vmax=vmax)
        
        this_ax.set_ylabel('t (s)', rotation=0, labelpad=30, fontsize=fontsize, family=font)
        this_ax.set_ylim(0, t.max())
        this_ax.axvline(cf.xmin     / cf.dx, c='w', ls=':', alpha=1.0)
        this_ax.axvline(cf.xmax     / cf.dx, c='w', ls=':', alpha=1.0)
        this_ax.axvline(0.0                , c='w', ls=':', alpha=0.75)   
        this_ax.set_xlim(x_arr[0], x_arr[-1]) 
        
        axes.append(this_ax)
    
    this_ax.set_xlabel('$x/\Delta x$', fontsize=fontsize, family=font)
    
    cb = fig_all.colorbar(im1a, cax=cax_all)   
    cb.set_label('nT', rotation=0, family=font, fontsize=fontsize, labelpad=20)
    fig_all.subplots_adjust(wspace=0.05)
    
    plot_dir = 'E://runs//CH4_tests_v2//'
    fullpath = plot_dir + 'bt_thesis_combined' + '.png'
    fig_all.savefig(fullpath, facecolor=fig_all.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=200)
    print('t-x Plot saved')
    
    plt.close('all')
    return


def plot_FB_waves_winske(save=True):
    '''
    Routine that splits B-field up into its backwards/forwards propagating
    components and plots each one. Check is done by adding these two waves
    together and checking it against the original field. They should be
    the same.
    
    # Once that's proven, do a tx plot to look at the propagation of the waves
    '''
    skip  = 50
    
    # Winske version
    ftime, By_raw  = cf.get_array('By')
    ftime, Bz_raw  = cf.get_array('Bz')
    ftime, Bt_pos, Bt_neg = bk.get_helical_components(False)
    
    B_raw = By_raw[:, 1:-2] + 1j*Bz_raw[:, 1:-2]
    B_max = np.abs(B_raw).max()
    B_sum = Bt_pos + Bt_neg
    
    By_pos = Bt_pos.real
    By_neg = Bt_neg.real
    Bz_pos = Bt_pos.imag
    Bz_neg = Bt_neg.imag

    xarr = np.arange(By_pos.shape[1])

    plt.ioff()
    for ii in range(ftime.shape[0]):
        if ii%skip == 0:
            print('Plotting FB waves {} of {}'.format(ii, ftime.shape[0]))
            fig, axes = plt.subplots(2, sharex=True, figsize=(16,10))
            
            axes[0].set_title('+/- Waves :: t = {:.2f}'.format(ftime[ii]))
            
            axes[0].plot(xarr, By_pos[ii], c='b', label='Fwd')
            axes[0].plot(xarr, Bz_pos[ii], c='b', alpha=0.5)
            axes[0].plot(xarr, By_neg[ii], c='r', label='Bwd')
            axes[0].plot(xarr, Bz_neg[ii], c='r', alpha=0.5)
            axes[0].set_ylim(-B_max/2, B_max/2)
            axes[0].legend()
            
            axes[1].plot(xarr, np.abs(B_sum[ii]), c='r', label='Sum')
            axes[1].plot(xarr, np.abs(B_raw[ii]), c='k', label='Raw')
            axes[1].set_ylim(0, B_max)
            
            for ax in axes:
                ax.set_xlim(xarr[0], xarr[-1])
                ax.set_ylabel('nT' , rotation=0, labelpad=20)
                ax.legend()
                
            if save==True:  
                save_folder = cf.anal_dir + '/FB_waves_check/'
                if os.path.exists(save_folder) == False:
                    os.makedirs(save_folder)
            
                fig.savefig(save_folder + 'FB_wave_check_{:05}.png'.format(ii), edgecolor='none')
                plt.close('all')
    return



def plot_FB_waves_timeseries(save=True):
    '''
    DIAGNOSTIC
    Validated by plotting B_sum and B_raw on same plot, they were identical.
    '''
    ftime, B_fwd, B_bwd, B_raw = bk.get_FB_waves(overwrite=False, field='B', st=1, en=-2)
    B_fwd *= 1e9; B_bwd *= 1e9; B_raw *= 1e9

    xarr   = cf.B_nodes[1:-2]
    skip   = 50
    
    B_sum = B_fwd + B_bwd
    B_max = np.abs(B_raw).max()
    
    plt.ioff()
    for ii in range(ftime.shape[0]):
        if ii%skip == 0:
            print('Plotting FB waves {} of {}'.format(ii, ftime.shape[0]))
            fig, axes = plt.subplots(2, sharex=True, figsize=(16,10))
            
            axes[0].set_title('Fwd/Bwd Waves :: t = {:.2f}'.format(ftime[ii]))
            
            axes[0].plot(xarr, B_fwd[ii].real, c='b', label='Fwd')
            axes[0].plot(xarr, B_fwd[ii].imag, c='b', alpha=0.5)
            axes[0].plot(xarr, B_bwd[ii].real, c='r', label='Bwd')
            axes[0].plot(xarr, B_bwd[ii].imag, c='r', alpha=0.5)
            axes[0].set_ylim(-B_max/2, B_max/2)
            axes[0].legend()
            
            axes[1].plot(xarr, np.abs(B_sum[ii]), c='r', label='Sum')
            axes[1].plot(xarr, np.abs(B_raw[ii]), c='k', label='Raw')
            axes[1].set_ylim(0, B_max)
            
            for ax in axes:
                ax.set_xlim(xarr[0], xarr[-1])
                ax.set_ylabel('nT' , rotation=0, labelpad=20)
                ax.legend()
                
            if save==True:  
                save_folder = cf.anal_dir + '/FB_waves_check_shoji/'
                if os.path.exists(save_folder) == False:
                    os.makedirs(save_folder)
            
                fig.savefig(save_folder + 'FB_wave_check_{:05}.png'.format(ii), edgecolor='none')
                plt.close('all')
    return


def plot_particle_paths(it_max=None, nsamples=1000):    
    save_folder = cf.anal_dir + '//particle_trajectories//'
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
    
    indexes = np.random.randint(0, cf.N, nsamples)
    Np      = len(indexes)

    spos = np.zeros((it_max, Np))
    svel = np.zeros((it_max, Np, 3))
    
    for ii in range(it_max):
        print('Loading timestep', ii, 'of', it_max)
        pos, vel, idx, ptime, idx_start, idx_end = cf.load_particles(ii, shuffled_idx=False)
        
        for index, jj in zip(indexes, range(Np)):
            spos[ii, jj]    = pos[   index]
            svel[ii, jj, :] = vel[:, index]
    
    for ii in Np:
        prt = indexes[ii]
        print('Plotting trajectory for particle', ii)
        fig, ax = plt.subplots()
        ax.set_title('Particle {} Trajectory :: Bottle {:.1f} - {:.1f} nT'.format(prt, cf.B_eq*1e9, cf.B_xmax*1e9))
        ax.scatter(spos[:, prt], svel[:, prt, 0], label='$v_x$')
        ax.scatter(spos[:, prt], svel[:, prt, 1], label='$v_y$')
        ax.scatter(spos[:, prt], svel[:, prt, 2], label='$v_z$')
        
        fig.savefig(save_folder + 'particle_{:08}'.format(ii))
        plt.close('all')
    return


def plot_density_change(ppd=False, it_max=None, save=True, overwrite_moments=False):
    '''
    Use collected charge density to work out if a particle distribution has
    relaxed into the field or not.
    -- Calculate change (as percentage) for each grid point from previous time
    -- Sum absolute changes
    -- Plot with time
    '''
    if cf.particle_open == 1:
        shuffled_idx = True
    else:
        shuffled_idx = False
        
    if ppd == True and os.path.exists(cf.data_dir + '//equil_particles//') == False:
        print('No equilibrium data to plot. Aborting.')
        return
    
    if it_max is None:
        if ppd == False:
            it_max = len(os.listdir(cf.particle_dir))
        else:
            it_max = len(os.listdir(cf.data_dir + '//equil_particles//'))
    
    filename = 'number_density_time_variation.png'
    filepath = cf.anal_dir + filename

    save_file = cf.temp_dir + 'numberflux_densities.npz'
    if os.path.exists(save_file) and not overwrite_moments:
        print('Loading moments from file...')
        dat = np.load(save_file)
        number_densities = dat['number_densities']
        flux_densities = dat['flux_densities']
        ptimes = dat['ptimes']
        gtimes = dat['gtimes']
    else:
        print('Extracting number densities...')
        number_densities = np.zeros((it_max, cf.NC, cf.Nj))
        flux_densities   = np.zeros((it_max, cf.NC, cf.Nj, 3))
        
        ptimes = np.zeros(it_max)
        gtimes = np.zeros(it_max)
        for ii in range(it_max):
            print('Particle timestep', ii)
            pos, vel, idx, ptime, idx_start, idx_end =\
                cf.load_particles(ii, shuffled_idx=shuffled_idx, preparticledata=ppd)
            
            if cf.disable_waves:
                ptime = cf.dt_particle*ii
            gf_time = ptime * cf.gyfreq_xmax
            ptimes[ii] = ptime
            gtimes[ii] = gf_time
            
            ni, nu = bk.get_number_densities(pos, vel, idx)
            
            number_densities[ii] = ni
            flux_densities[ii]   = nu
        print('Saving to file...')
        np.savez(save_file, number_densities=number_densities,
                            flux_densities=flux_densities, 
                            ptimes=ptimes, gtimes=gtimes)
            
    # Calculate density change   
    diffs = np.zeros((it_max, cf.NC, cf.Nj), dtype=float)
    for ii in range(1, number_densities.shape[0]):
        diffs[ii] = np.abs(number_densities[ii] - number_densities[ii - 1])
    diffs = diffs.sum(axis=1)
    
    plt.ioff()
    fig, axes = plt.subplots(cf.Nj, figsize=(16, 10), sharex=True)
    for jj in range(cf.Nj):
        axes[jj].plot(gtimes[1:], diffs[1:, jj], c=cf.temp_color[jj])
        axes[jj].set_ylabel(cf.species_lbl[jj])
    axes[-1].set_xlabel('Time ($\Omega t$)')
    
    fig.savefig(filepath)
    return


def thesis_plot_dispersion(save=True, fmax=1.0, tmax=None, Bmax=None, Pmax=None):
    '''
    Stackplot of xt and wx plot. Try next to each other and stacked vertically.
     -- Total xt
     -- Total wx (add powers)
    '''
    # Get wx data and axes
    ftime, by_wx = disp.get_wx('By', fac=1e9)
    ftime, bz_wx = disp.get_wx('Bz', fac=1e9)
    wx = by_wx + bz_wx
    
    freqs = np.fft.rfftfreq(ftime.shape[0], d=cf.dt_field)
    x_arr = cf.B_nodes * 1e-6
    x_lim = np.array([cf.xmin, cf.xmax]) * 1e-6

    # Get xt
    ftime, By_raw = cf.get_array('by')
    ftime, Bz_raw = cf.get_array('bz')
    B_wave = np.sqrt(By_raw**2 + Bz_raw**2)*1e9
    
    ffamily = 'monospace'
    fsize   = 12
    lpad    = 15
    cpad    = 30
    
    ## PLOT
    plt.ioff()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.0, 4.0), 
                             gridspec_kw={'width_ratios':[1, 0.01]})
    
    # Gyrofrequencies
    gy_x  = np.linspace(cf.xmin, cf.xmax, 1000)
    gy_bx = cf.B_eq * (1. + cf.a * gy_x**2)
    
    gy_H  = qi * gy_bx / (2 * np.pi * mp) 
    gy_He = 0.2500 * gy_H
    gy_O  = 0.0625 * gy_H 
    print(gy_He.min(), gy_He.max())
    
    # XT WAVE PLOT
    im1   = axes[0, 0].pcolormesh(x_arr, ftime, B_wave, cmap='jet', vmin=0.0, vmax=Bmax)
    cbar1 = fig.colorbar(im1, cax=axes[0, 1], extend='max')
    cbar1.set_label('nT', rotation=0,
                    family=ffamily, fontsize=fsize, labelpad=cpad-10)

    axes[0, 0].set_ylabel('t\n(s)', rotation=0, labelpad=lpad, fontsize=fsize, family=ffamily)
    axes[0, 0].set_xlabel('x (Mm)', fontsize=fsize, family=ffamily)
    axes[0, 0].set_ylim(0, tmax)
    axes[0, 0].set_xlim(x_lim[0], x_lim[1])    
    axes[0, 0].set_xticklabels([])
    
    # WX DISPERSION PLOT
    im2 = axes[1, 0].pcolormesh(x_arr, freqs, wx, cmap='nipy_spectral', vmin=0.0, vmax=Pmax)
    cbar2 = fig.colorbar(im2, cax=axes[1, 1], extend='max') #, ticks=[0, 1, 2, 3, 4, 5]
    cbar2.set_label('$\\frac{nT^2}{Hz}$', rotation=0,
                    family=ffamily, fontsize=fsize+2, labelpad=cpad)
    
    axes[1, 0].set_ylabel('f\n(Hz)', rotation=0, labelpad=lpad, fontsize=fsize, family=ffamily)
    axes[1, 0].set_xlabel('x (Mm)', fontsize=fsize, family=ffamily)
    axes[1, 0].set_ylim(0.0, fmax)
    #axes[1, 0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axes[1, 0].set_xlim(x_lim[0], x_lim[1])
    
    #axes[1, 0].plot(gy_x*1e-6, gy_H , c='w',      label='$f_{cH^+}$')
    axes[1, 0].plot(gy_x*1e-6, gy_He, c='yellow', label='$f_{cHe^+}$')
    axes[1, 0].plot(gy_x*1e-6, gy_O , c='r',      label='$f_{cO^+}$')
    axes[1, 0].legend(loc='upper left', ncol=2)
    
    fig.tight_layout()
    fig.align_ylabels()
    fig.subplots_adjust(hspace=0.00, wspace=0.05)
                            
    if save:
        fullpath = cf.anal_dir + 'thesis_dispersion_stackplot_revised.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(),
                    edgecolor='none', bbox_inches='tight', dpi=200)
        print('Dispersion stackplot saved')
        plt.close('all')
    return


def plot_equilibrium_distribution(saveas='eqdist', histogram=True):
    '''
    Plots the initial and final 'relaxed' particle distributions to show how
    the particles were initially loaded and what their state is before the
    beginning of the simulation
    
    Plot x vs. vx, vy, vz and
    vx/vz, vy/vz, vx/vy, v_para/v_perp
    
    '''
    if cf.particle_open == 1:
        shuffled_idx = True
    else:
        shuffled_idx = False
        
    if os.path.exists(cf.data_dir + '//equil_particles//') == False:
        print('No equilibrium data to plot. Aborting.')
        return
    n_equil = len(os.listdir(cf.data_dir + '//equil_particles//'))
    
    filepath = cf.anal_dir + '//equilibrium_distributions//'
    filename = filepath + saveas
    if not os.path.exists(filepath): os.makedirs(filepath)
        
    print('Loading particles...')
    pos1, vel1, idx1, ptime1, idx_start1, idx_end1 =\
        cf.load_particles(0, shuffled_idx=shuffled_idx, preparticledata=True)
    
    pos2, vel2, idx2, ptime2, idx_start2, idx_end2 =\
        cf.load_particles(n_equil-1, shuffled_idx=shuffled_idx, preparticledata=True)
        
    vel1 /= cf.va; vel2 /= cf.va
    pos1 /= cf.dx; pos2 /= cf.dx
        
    v_perp1 = np.sqrt(vel1[1] ** 2 + vel1[2] ** 2) * np.sign(vel1[2])
    v_perp2 = np.sqrt(vel2[1] ** 2 + vel2[2] ** 2) * np.sign(vel2[2])
    
    xbins = np.linspace(cf.xmin/cf.dx, cf.xmax/cf.dx, cf.NX + 1, endpoint=True)
    
    plt.ioff()
    for jj in range(cf.Nj):
        
        # Do the calculations and plotting
        cfac  = 10  if cf.temp_type[jj] == 1 else 10
        vlim  = 5*cf.vth_perp[jj]/cf.va
        vbins = np.linspace(-vlim, vlim, 501, endpoint=True)
        print('\nGenerating plots for species', jj)
        
        if True:
            pfac = 3
            
            # Plot Spatial (vx, vy, vz, vperp) for before and after
            fig1, axes1 = plt.subplots(nrows=4, ncols=2, figsize=(16, 9))
            fig1.suptitle(cf.species_lbl[jj])
            axes1[0, 0].set_title('At initialization')
            st1 = idx_start1[jj]; en1 = idx_end1[jj]
            
            counts, xedges, yedges, im1a = axes1[0, 0].hist2d(pos1[st1:en1], vel1[0, st1:en1], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im1a, ax=axes1[0, 0], pad=0.015)
            cb.set_label('Counts')
            axes1[0, 0].set_ylabel('$v_x$', rotation=0)
            
            counts, xedges, yedges, im2a = axes1[1, 0].hist2d(pos1[st1:en1], vel1[1, st1:en1], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im2a, ax=axes1[1, 0], pad=0.015)
            cb.set_label('Counts')
            axes1[1, 0].set_ylabel('$v_y$', rotation=0)
            
            counts, xedges, yedges, im3a = axes1[2, 0].hist2d(pos1[st1:en1], vel1[2, st1:en1], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im3a, ax=axes1[2, 0], pad=0.015)
            cb.set_label('Counts')
            axes1[2, 0].set_ylabel('$v_z$', rotation=0)
            
            counts, xedges, yedges, im4a = axes1[3, 0].hist2d(pos1[st1:en1], v_perp1[st1:en1], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im4a, ax=axes1[3, 0], pad=0.015)
            cb.set_label('Counts')
            axes1[3, 0].set_ylabel('$v_\perp$', rotation=0)
            
            
            axes1[0, 1].set_title('After relaxation')
            st2 = idx_start2[jj]; en2 = idx_end2[jj]
            
            counts, xedges, yedges, im1b = axes1[0, 1].hist2d(pos2[st2:en2], vel2[0, st2:en2], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im1b, ax=axes1[0, 1], pad=0.015)
            cb.set_label('Counts')
            axes1[0, 1].set_ylabel('$v_x$', rotation=0)
            
            counts, xedges, yedges, im2b = axes1[1, 1].hist2d(pos2[st2:en2], vel2[1, st2:en2], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im2b, ax=axes1[1, 1], pad=0.015)
            cb.set_label('Counts')
            axes1[1, 1].set_ylabel('$v_y$', rotation=0)
            
            counts, xedges, yedges, im3b = axes1[2, 1].hist2d(pos2[st2:en2], vel2[2, st2:en2], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im3b, ax=axes1[2, 1], pad=0.015)
            cb.set_label('Counts')
            axes1[2, 1].set_ylabel('$v_z$', rotation=0)
            
            counts, xedges, yedges, im4b = axes1[3, 1].hist2d(pos2[st2:en2], v_perp2[st2:en2], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im4b, ax=axes1[3, 1], pad=0.015)
            cb.set_label('Counts')
            axes1[3, 1].set_ylabel('$v_\perp$', rotation=0)
           
            for mm in range(4):
                for nn in range(2):
                    axes1[mm, nn].set_xlim(cf.xmin/cf.dx, cf.xmax/cf.dx)
                    axes1[mm, nn].set_ylim(-vlim, vlim)
                    
            fig1.subplots_adjust()
            fig1.savefig(filename + f'_spatial_sp{jj}.png')
            print('Plot saved as', filename + f'_spatial_sp{jj}.png')
        
        
        if True:
            vfac=6
            
            # Plot phase space (vx/vy, vx/vz, vy/vz, v_perp/v_para)
            fig2, axes2 = plt.subplots(nrows=4, ncols=2, figsize=(16, 9))
            fig2.suptitle(cf.species_lbl[jj])
            
            axes2[0, 0].set_title('At initialization')
            st1 = idx_start1[jj]; en1 = idx_end1[jj]
            
            counts, xedges, yedges, im1c = axes2[0, 0].hist2d(vel1[0, st1:en1], vel1[1, st1:en1], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im1c, ax=axes2[0, 0], pad=0.015)
            cb.set_label('Counts')
            axes2[0, 0].set_ylabel('$v_x$', rotation=0)
            
            counts, xedges, yedges, im2c = axes2[1, 0].hist2d(vel1[0, st1:en1], vel1[2, st1:en1], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im2c, ax=axes2[1, 0], pad=0.015)
            cb.set_label('Counts')
            axes2[1, 0].set_ylabel('$v_y$', rotation=0)
            
            counts, xedges, yedges, im3c = axes2[2, 0].hist2d(vel1[1, st1:en1], vel1[2, st1:en1], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im3c, ax=axes2[2, 0], pad=0.015)
            cb.set_label('Counts')
            axes2[2, 0].set_ylabel('$v_z$', rotation=0)
            
            counts, xedges, yedges, im4c = axes2[3, 0].hist2d(vel1[0, st1:en1], v_perp1[st1:en1], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im4c, ax=axes2[3, 0], pad=0.015)
            cb.set_label('Counts')
            axes2[3, 0].set_ylabel('$v_\perp$', rotation=0)
            
            
            axes2[0, 1].set_title('After relaxation')
            st2 = idx_start2[jj]; en2 = idx_end2[jj]
            
            counts, xedges, yedges, im1d = axes2[0, 1].hist2d(vel2[0, st2:en2], vel2[1, st2:en2], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im1d, ax=axes2[0, 1], pad=0.015)
            cb.set_label('Counts')
            axes2[0, 1].set_ylabel('$v_x$', rotation=0)
            
            counts, xedges, yedges, im2d = axes2[1, 1].hist2d(vel2[0, st2:en2], vel2[2, st2:en2], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im2d, ax=axes2[1, 1], pad=0.015)
            cb.set_label('Counts')
            axes2[1, 1].set_ylabel('$v_y$', rotation=0)
            
            counts, xedges, yedges, im3d = axes2[2, 1].hist2d(vel2[1, st2:en2], vel2[2, st2:en2], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im3d, ax=axes2[2, 1], pad=0.015)
            cb.set_label('Counts')
            axes2[2, 1].set_ylabel('$v_z$', rotation=0)
            
            counts, xedges, yedges, im4d = axes2[3, 1].hist2d(vel2[0, st2:en2], v_perp2[st2:en2], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im4d, ax=axes2[3, 1], pad=0.015)
            cb.set_label('Counts')
            axes2[3, 1].set_ylabel('$v_\perp$', rotation=0)
            
            for mm in range(4):
                for nn in range(2):
                    axes2[mm, nn].axis('equal')
                    
            fig2.subplots_adjust()
            fig2.savefig(filename + f'_phase_sp{jj}.png')
            print('Plot saved as', filename + f'_phase_sp{jj}.png')
            plt.close('all')
        
    return
#%% MAIN
if __name__ == '__main__':
    drive       = 'E:'
    
    #############################
    ### MULTI-SERIES ROUTINES ###
    #############################
    if False:            
        #multiplot_mag_energy(save=True)
        #multiplot_fluxes(series)
        #multiplot_parallel_scaling()
        #multiplot_saturation_amplitudes(save=True)
        #multiplot_RMS_jul25()
        
        #multiplot_saturation_amplitudes_jan16(save=True)
        #multiplot_saturation_amplitudes_jul17(save=True)
        
        multiplot_AUG12_waveform(fmax=1.0)
        #multiplot_AUG12_Benergy(save=True)
        #multiplot_AUG12_Benergy_times(save=True)
        
        #multiplot_wk_thesis_good(saveas='wk_plot_thesis', save=True)
        #multiplot_ABCs_thesis()
        print('Exiting multiplot routines successfully.')
        sys.exit()
    
    ####################################
    ### SINGLE SERIES ANALYSIS ########
    ################################
    #'//_NEW_RUNS//AUG12_PC1_PEAKS_CAVRCVO_5HE_1pc//',
                  # '//_NEW_RUNS//AUG12_PC1_PEAKS_Vrc_5HE_1pc//',
                  # '//_NEW_RUNS//JUL17_PC1PEAKS_VO_1pc//',
                  # '//_NEW_RUNS//JUL25_CP_MULTIPOP_LONGER//',
                  # '//_NEW_RUNS//JUL25_CP_PBOLIC_LONGER//'
    for series in ['//energy_conservation_tests//']:
        series_dir = f'{drive}/runs//{series}//'
        num_runs   = len([name for name in os.listdir(series_dir) if 'run_' in name])
        print('{} runs in series {}'.format(num_runs, series))
        
        if True:
            runs_to_do = range(num_runs)
        else:
            runs_to_do = [8]

        # Extract all summary files and plot field stuff (quick)
        if True:
            for run_num in runs_to_do:
                print('\nRun {}'.format(run_num))
                #cf.delete_analysis_folders(drive, series, run_num)
                
                cf.load_run(drive, series, run_num, extract_arrays=True, overwrite_summary=True)
                #plot_stability_check()
                #thesis_plot_dispersion(save=True, fmax=1.1, tmax=None, Bmax=None, Pmax=None)
                    
                #single_point_spectra(nx=[2, 130, 258, 386, 514], overlap=0.95, f_res_mHz=25, fmax=1.0)
                    #single_point_FB(     nx=[2, 130, 258, 386, 514], overlap=0.95, f_res_mHz=25, fmax=1.0)
                    
                #single_point_spectra(nx=[2, 258, 514, 770, 1026], overlap=0.95, f_res_mHz=25, fmax=1.0)
                    #single_point_FB(     nx=[2, 258, 514, 770, 1026], overlap=0.95, f_res_mHz=25, fmax=1.0)
    
                    #standard_analysis_package(disp_overlay=False, pcyc_mult=1.1,
                    #              tx_only=False, tmax=None, remove_ND=False)
                
                #plot_equilibrium_distribution(saveas='eqdist', histogram=True)
                
                plot_energies(normalize=True, save=True)
                
# =============================================================================
#                 plot_abs_T(saveas='abs_plot', save=True, log=False, tmax=None, normalize=False,
#                            B0_lim=None, remove_ND=False)
#                 plot_abs_T(saveas='abs_plot', save=True, log=True, tmax=None, normalize=False,
#                            B0_lim=None, remove_ND=False)
#                 
#                 plot_abs_J(saveas='abs_plot', save=True, log=False, tmax=None, remove_ND=False)
#                 plot_abs_J(saveas='abs_plot', save=True, log=True, tmax=None, remove_ND=False)
#                 
#                 plot_tx(component='By', saveas='tx_plot', save=True, log=False, tmax=None, 
#                         remove_ND=False, normalize=False, bmax=None)
#                 plot_tx(component='By', saveas='tx_plot', save=True, log=True, tmax=None, 
#                         remove_ND=False, normalize=False, bmax=None)
#                 
#                 plot_wk_thesis_good(dispersion_overlay=True, save=True,
#                      pcyc_mult=1.1, xmax=0.8, zero_cold=True,
#                      linear_only=False, normalize_axes=True, centre_only=False)
#                 
#                 plot_vi_vs_x(it_max=None, sp=None, save=True, shuffled_idx=False, skip=1, ppd=False)
# =============================================================================
                
                #plot_abs_T_w_Bx(saveas='abs_plot_bx', save=True, tmax=None,
                #    B0_lim=None, remove_ND=False)

                #check_fields(save=True, ylim=False, skip=5)
                #plot_total_density_with_time()
                #plot_max_velocity()
                #check_fields(save=True, ylim=True, skip=25)
                

                #winske_summary_plots(save=True)
                #plot_helical_waterfall(title='', save=True, overwrite=False, it_max=None)
                #winske_magnetic_density_plot()
                #disp.plot_kt_winske()
                #disp.plot_fourier_mode_timeseries(it_max=None)
                
                #plot_kt(component='By', saveas='kt_plot_norm', save=True, normalize_x=True, xlim=1.0)
                
# =============================================================================
#                 for _nx in range(cf.ND, cf.NX+cf.NX+1):
#                     SWSP_dynamic_spectra(nx=_nx, overlap=0.95, f_res=50)
# =============================================================================
# =============================================================================
#                 # Find cell at x = 0, 1000km
#                 for x_pos in [5e5, 1e6, 2e6, 3e6, 4e6]:
#                     print('Plotting split-wave single point at x = {:.1f}km'.format(x_pos*1e-3))
#                     try:
#                         diff_xpos = abs(cf.B_nodes - x_pos)
#                         xidx      = np.where(diff_xpos == diff_xpos.min())[0][0]
#                         ggg.SWSP_timeseries(nx=xidx , save=True, log=True, normalize=True, tmax=45, LT_overlay=False)
#                     except:
#                         print('ABORT: SWSP ERROR')
#                         continue
# =============================================================================

                #field_energy_vs_time(save=True, saveas='mag_energy_reflection', tmax=None)
                #ggg.straight_line_fit(save=True, normfit_min=0.3, normfit_max=0.7, normalize_time=True,
                #                      plot_LT=True, plot_growth=True, klim=1.5)
                
                #try:
                #standard_analysis_package(thesis=False, tx_only=False, disp_overlay=True, remove_ND=False)
# =============================================================================
#                 except:
#                     pass            
# =============================================================================
                       
                
        #plot_phase_space_with_time()
            
        #plot_helical_waterfall(title='', save=True, overwrite=False, it_max=None)
            
        #do_all_dynamic_spectra(ymax=2.0)
        #do_all_dynamic_spectra(ymax=None)

        #plot_energies(normalize=False, save=True)
        
        #single_point_both_fields_AGU()
        
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
#             plot_energies(normalize=True, save=True)
#         except:
#             pass
# =============================================================================