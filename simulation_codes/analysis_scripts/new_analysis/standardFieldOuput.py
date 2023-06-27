# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:56:34 2016

@author: c3134027
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, sys, pdb

import doSpectra as ds
from _constants import UNIT_CHARGE, PROTON_MASS, ELECTRON_MASS, ELEC_PERMITTIVITY, MAGN_PERMEABILITY, BOLTZMANN_CONSTANT, LIGHT_SPEED


# Ignore common warnings. If shit goes funky, turn them back on by replacing 'ignore' with 'default'
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
'''
Contains standard diagnostic outputs for fields
Specifically contains plots that are only single output, no particle data
i.e. not multiple plots for each cell, etc. just stuff you see at a glance

Other functions TODO:
    -- Particle summary plots (Winske plots, Jackson plots, vi vs. x etc.)
    -- Detailed field plots (i.e. for each time check fields, or for each point)
    -- Deeper analysis involving fields (i.e. reflection, growth rates, poynting, helicity)
    -- Field dynamic spectra incl. with forward/backward or helical wave decomposition
    -- Other diagnostic plots (e.g. density conservation, mu for single particles, etc.)
    
TODO:
    -- Fetching should be done in calling functions, plotting by another function
NOTE: Don't treat attr calls like copies - they're mutable, and will change in future calls
'''

def plotFieldDiagnostics(Sim, log_tx=False, remove_ND=False, normalize_B0=False, bmax=None):
    '''
    Wrapper function to pass Sim instance to each plotting function and save the output
    Need to decide what flags go here (e.g. time, log, save, normalizations, max values, etc.)
    '''
    plot_tx(Sim, component='By', save=True, log=log_tx, tmax=None, remove_ND=remove_ND, normalize=normalize_B0,
                bmax=bmax)   
    plot_abs_T(Sim, save=True, log=False, tmax=None,
                   normalize=False, B0_lim=None, remove_ND=False)
    plot_abs_J(Sim, saveas='abs_plot', save=True, log=False, tmax=None,
                   remove_ND=False)
    plot_wk_thesis_good(Sim, dispersion_overlay=False, save=True,
                         pcyc_mult=None, xmax=None, zero_cold=True,
                         linear_only=False, normalize_axes=False,
                         centre_only=False)
    plot_wx(Sim, component='By', linear_overlay=False, 
                save=True, pcyc_mult=None, remove_ND=False)
    return


def plot_tx(Sim, component='By', save=False, log=False, tmax=None,
            remove_ND=False, normalize=False, bmax=None):
    plt.ioff()

    t   = getattr(Sim, 'field_sim_time')
    arr = getattr(Sim, component.lower())
    
    fontsize = 18
    font     = 'monospace'
    
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    mpl.rcParams['ytick.labelsize'] = tick_label_size 
    
    if component[0] == 'B':
        arr *= 1e9
        x    = Sim.B_nodes / Sim.dx
    else:
        arr *= 1e3
        x    = Sim.E_nodes / Sim.dx
        
    if normalize == True and component[0].lower() == 'b':
        arr /= (Sim.B_eq*1e9)
        
    if tmax is None:
        lbl = 'full'
    else:
        lbl = '{:04}'.format(tmax)
    
    if remove_ND == True:
        x_lim = [Sim.xmin / Sim.dx, Sim.xmax / Sim.dx]
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

    ax.set_title(f'{component} spatiotemporal :: {Sim.series_name}[{Sim.run_num}]', fontsize=fontsize, family=font)
    ax.set_ylabel('t (s)', rotation=0, labelpad=30, fontsize=fontsize, family=font)
    ax.set_xlabel('x ($\Delta x$)', fontsize=fontsize, family=font)
    ax.set_ylim(0, tmax)
    
    ax.axvline(Sim.xmin / Sim.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(Sim.xmax / Sim.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(0.0 / Sim.dx, c='w', ls=':', alpha=0.75)   
    ax.set_xlim(x_lim[0], x_lim[1])
        
    if save == True:
        fullpath = Sim.anal_dir + 'tx_plot' + '_{}_{}'.format(component.lower(), lbl) + suffix + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('t-x Plot saved')
        plt.close('all')

    return


def plot_wx(Sim, component='By', saveas='wx_plot', linear_overlay=False, 
            save=False, pcyc_mult=None, remove_ND=False):
    plt.ioff()
    wx = ds.get_wx(Sim, component)
    
    if component[0] == 'B':
        x = Sim.B_nodes
    else:
        x = Sim.E_nodes
        
    f  = np.fft.rfftfreq(Sim.num_field_steps, d=Sim.dt_field)
    
    if remove_ND == True:
        x_lim = [Sim.xmin, Sim.xmax]
    else:
        x_lim = [x[0], x[-1]]
    
    ## PLOT IT
    fig, ax = plt.subplots(1, figsize=(15, 10))
    im1 = ax.pcolormesh(x, f, wx, cmap='nipy_spectral')      # Remove f[0] since FFT[0] >> FFT[1, 2, ... , k]
    fig.colorbar(im1)
    
    lbl  = [r'$\Omega_{H^+}$', r'$\Omega_{He^+}$', r'$\Omega_{O^+}$']
    clr  = ['white', 'yellow', 'red']    
    M    = np.array([1., 4., 16.]) * PROTON_MASS
    
    try:
        pcyc = UNIT_CHARGE * Sim.Bc[:, 0] / (2 * np.pi * PROTON_MASS)
    except:
        pcyc = UNIT_CHARGE * Sim.B_eq * np.ones(Sim.B_nodes.shape[0]) / (2 * np.pi)
    
    for ii in range(3):
        if Sim.species_present[ii] == True:
            try:
                cyc = UNIT_CHARGE * Sim.Bc[:, 0] / (2 * np.pi * PROTON_MASS * M[ii])
            except:
                cyc = UNIT_CHARGE * Sim.B_eq * np.ones(Sim.B_nodes.shape[0]) / (2 * np.pi * M[ii])
                
            ax.plot(Sim.B_nodes, cyc, linestyle='--', c=clr[ii], label=lbl[ii])
    
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

    ax.set_title(f'{component} spatiofrequency :: {Sim.series_name}[{Sim.run_num}]', fontsize=14, family='monospace')
    ax.set_ylabel(r'f (Hz)', rotation=0, labelpad=15)
    ax.set_xlabel('x (m)')
    
    if pcyc_mult is not None:
        ax.set_ylim(0, pcyc_mult*pcyc.max())
    else:
        ax.set_ylim(0, None)

    ax.axvline(Sim.xmin, c='w', ls=':', alpha=1.0)
    ax.axvline(Sim.xmax, c='w', ls=':', alpha=1.0)
    ax.axvline(0.0, c='w', ls=':', alpha=0.75)   
    ax.set_xlim(x_lim[0], x_lim[1])

    if save == True:
        fullpath = Sim.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('w-x Plot saved')
        plt.close('all')
    return


def plot_kt(Sim, component='By', saveas='kt_plot', save=False, normalize_x=False, xlim=None):
    '''
    Note: k values from fftfreq() are their linear counterparts. Need to multiply by 2pi
    to compare with theory
    '''
    plt.ioff()
    k, kt, st, en = ds.get_kt(Sim, component)
    ftime = getattr(Sim, 'field_sim_time')
    
    fig = plt.figure(1, figsize=(15, 10))
    ax  = fig.add_subplot(111)
    
    if normalize_x == False:
        im1 = ax.pcolormesh(k*1e6, ftime, kt, cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k] antialiased=True
        fig.colorbar(im1)
        ax.set_ylabel(r'$\Omega_i t$', rotation=0)
        ax.set_xlabel(r'$k (m^{-1}) \times 10^6$')
        #ax.set_ylim(0, 15)
    else:
        k_plot = 2*np.pi*k * LIGHT_SPEED / Sim.wpi
        
        im1 = ax.pcolormesh(k_plot, ftime, kt, cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k] antialiased=True
        fig.colorbar(im1)
        ax.set_ylabel(r'$\Omega_i t$', rotation=0)
        ax.set_xlabel(r'$kc/\omega_{pi}$')
        #ax.set_ylim(0, 15)
        ax.set_xlim(0, xlim)
    ax.set_title(f'{component} Wavenumber-Time :: {Sim.series_name}[{Sim.run_num}]', fontsize=14, family='monospace')
    
    if save == True:
        fullpath = Sim.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close('all')
        print('k-t Plot saved')
    return



def plot_wk(Sim, saveas='wk_plot', dispersion_overlay=False, save=True,
                     pcyc_mult=None, xmax=None, zero_cold=True,
                     linear_only=False, normalize_axes=False,
                     centre_only=False):
    '''
    21/02/2021 rewriting this to just take the w/k of the perp/parallel fields
    and do all fields in the one call.
    '''
    if normalize_axes == True:
        xfac = LIGHT_SPEED / Sim.wpi
        yfac = 2*np.pi / Sim.gyfreq
        xlab = '$\mathtt{kc/\omega_{pi}}$'
        ylab = 'f\n$(\omega / \Omega_H)$'
        cyc  = 1.0 / np.array([1., 4., 16.])
    else:
        xfac = 1e6
        yfac = 1.0
        xlab = '$\mathtt{k (\\times 10^{-6}m^{-1})}$'
        ylab = 'f\n(Hz)'
        cyc  = UNIT_CHARGE * Sim.B_eq / (2 * np.pi * PROTON_MASS * np.array([1., 4., 16.]))
        
        
    for field in ['B', 'E']:
    
        # Calculate dispersion relations from model data
        k, f, wk_para, tf = ds.get_wk(field+'x', linear_only=linear_only, norm_z=normalize_axes, centre_only=centre_only)
        k, f, wky,     tf = ds.get_wk(field+'y', linear_only=linear_only, norm_z=normalize_axes, centre_only=centre_only)
        k, f, wkz,     tf = ds.get_wk(field+'z', linear_only=linear_only, norm_z=normalize_axes, centre_only=centre_only)
        
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

        ax1.set_title(r'$\omega/k$ Plot :: {}_\perp :: Linear Theory up to {:.3f}s :: {}[{}]'.format(field, tf, Sim.series_name, Sim.run_num),
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
                if Sim.species_present[ii] == True:
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
                k_vals, CPDR_solns, WPDR_solns, HPDR_solns = ds.get_linear_dispersion_from_sim(k, zero_cold=zero_cold)
                for ii in range(CPDR_solns.shape[1]):
                    ax.plot(xfac*k_vals, yfac*CPDR_solns[:, ii].real, c='k', linestyle='-' , label='CPDR' if ii == 0 else '', alpha=alpha)
                    ax.plot(xfac*k_vals, yfac*WPDR_solns[:, ii].real, c='k', linestyle='--', label='WPDR' if ii == 0 else '', alpha=alpha)
                    ax.plot(xfac*k_vals, yfac*HPDR_solns[:, ii].real, c='k', linestyle=':' , label='HPDR' if ii == 0 else '', alpha=alpha)
                ax.legend(loc='upper right', facecolor='white', prop={'size': fontsize-2, 'family':font})


            
        if save == True:
            zero_suff = '' if zero_cold is False else 'zero'
            fullpath1  = Sim.anal_dir + saveas + '_{}perp'.format(field.upper()) + '_{}'.format(zero_suff)
            #fullpath2  = cf.anal_dir + saveas + '_{}para'.format(field.upper()) + '_{}'.format(zero_suff)
                
            fig1.savefig(fullpath1, facecolor=fig1.get_facecolor(), edgecolor='none', bbox_inches='tight')
            #fig2.savefig(fullpath2, facecolor=fig1.get_facecolor(), edgecolor='none', bbox_inches='tight')

            print('w-k for {} field saved'.format(field.upper()))
            plt.close('all')
        else:
            plt.show()
    return


def plot_wk_thesis_good(Sim, saveas='wk_plot_thesis', dispersion_overlay=False, save=True,
                     pcyc_mult=None, xmax=None, zero_cold=True,
                     linear_only=False, normalize_axes=False,
                     centre_only=False):
    '''
    Just do magnetic field, make axes big enough to see.
    
    Also need to put CPDR on these
    
    Combhine with other wk function
    '''
    if normalize_axes == True:
        xfac = LIGHT_SPEED / Sim.wpi
        yfac = 2*np.pi / Sim.gyfreq
        xlab = '$\mathtt{kc/\omega_{pi}}$'
        ylab = 'f\n$(\omega / \Omega_H)$'
        cyc  = 1.0 / np.array([1., 4., 16.])
    else:
        xfac = 1e6
        yfac = 1.0
        xlab = '$\mathtt{k (\\times 10^{-6}m^{-1})}$'
        ylab = 'f\n(Hz)'
        cyc  = UNIT_CHARGE * Sim.B_eq / (2 * np.pi * PROTON_MASS * np.array([1., 4., 16.]))
            
    # Calculate dispersion relations from model data
    k, f, wk_para, tf = ds.get_wk(Sim, 'bx', linear_only=linear_only, norm_z=normalize_axes, centre_only=centre_only)
    k, f, wky,     tf = ds.get_wk(Sim, 'by', linear_only=linear_only, norm_z=normalize_axes, centre_only=centre_only)
    k, f, wkz,     tf = ds.get_wk(Sim, 'bz', linear_only=linear_only, norm_z=normalize_axes, centre_only=centre_only)
    
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
        if Sim.species_present[ii] == True:
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
        k_vals, CPDR_solns, WPDR_solns, HPDR_solns = ds.get_linear_dispersion_from_sim(k, zero_cold=zero_cold)
        for ii in range(CPDR_solns.shape[1]):
            ax.plot(xfac*k_vals, yfac*CPDR_solns[:, ii].real, c='k', linestyle='--', label='CPDR' if ii == 0 else '', alpha=alpha)

    ax.set_title(r'$\omega/k$ Plot :: $B_\perp$ :: Linear Theory up to {:.3f}s :: {}[{}]'.format(tf, Sim.series_name, Sim.run_num),
                 fontsize=fontsize, family=font)
    
    fig1.tight_layout()
    fig1.subplots_adjust(wspace=0.05)
    
    if save == True:
        zero_suff = '' if zero_cold is False else 'zero'
        fullpath1  = Sim.anal_dir + saveas + '_Bperp' + '_{}'.format(zero_suff)
            
        fig1.savefig(fullpath1, facecolor=fig1.get_facecolor(), edgecolor='none', bbox_inches='tight')

        print('w-k for B-field saved')
        plt.close('all')
    else:
        plt.show()
    return



def plot_abs_T(Sim, saveas='abs_plot', save=False, log=False, tmax=None,
               normalize=False, B0_lim=None, remove_ND=False):
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

    t  = getattr(Sim, 'field_sim_time')
    by = getattr(Sim, 'by')
    bz = getattr(Sim, 'bz')
    
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
        bt     = np.sqrt(by ** 2 + bz ** 2) / Sim.B_eq
        clabel = '$\\frac{|B_\perp|}{B_{eq}}$'
        suff   = '_norm' 
        
    x = Sim.B_nodes / Sim.dx
        
    if tmax is None:
        lbl = 'full'
    else:
        lbl = '{:04}'.format(tmax)
        
    if remove_ND == True:
        xlim = [Sim.xmin/Sim.dx, Sim.xmax/Sim.dx]
    else:
        xlim = [x[0], x[-1]]
    
    ## PLOT IT
    fig, ax = plt.subplots(1, figsize=(6.0, 4.0))
    ax.set_title(f'|B| spatiotemporal :: {Sim.series_name}[{Sim.run_num}]')

    vmin = 0.0
    
    if B0_lim is None:
        vmax = bt.max()
    else:
        if normalize == True:
            vmax = Sim.B_eq * B0_lim * 1e9
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
    
    ax.axvline(Sim.xmin / Sim.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(Sim.xmax / Sim.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(0.0              , c='w', ls=':', alpha=0.75)   
    ax.set_xlim(xlim[0], xlim[1])    
    
    if save == True:
        fullpath = Sim.anal_dir + saveas + '_BPERP_{}{}{}'.format(lbl, suff, logsuff) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=200)
        print('B abs(t-x) Plot saved')
        plt.close('all')
    return


def plot_abs_T_w_Bx(Sim, saveas='abs_plot', save=False, tmax=None,
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

    t   = getattr(Sim, 'field_sim_time')
    by  = getattr(Sim, 'by')
    bz  = getattr(Sim, 'bz')
    bxc = getattr(Sim, 'bxc')
    bxc = bxc[:, Sim.NX//2]*1e9
    
    fontsize = 18
    font     = 'monospace'
    
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    mpl.rcParams['ytick.labelsize'] = tick_label_size 
    
    bt = np.sqrt(by ** 2 + bz ** 2) * 1e9
    x  = Sim.B_nodes / Sim.dx
        
    if tmax is None:
        lbl = 'full'
        tmax = t[-1]
    else:
        lbl = '{:04}'.format(tmax)
        
    if remove_ND == True:
        xlim = [Sim.xmin/Sim.dx, Sim.xmax/Sim.dx]
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
    axes[0].set_title(f'|B| spatiotemporal :: {Sim.series_name}[{Sim.run_num}]')

    
    axes[1].set_xlabel('x ($\Delta x$)', fontsize=fontsize, family=font)
    axes[1].set_ylim(0, tmax)
    axes[1].set_yticklabels([])
    
    axes[1].axvline(Sim.xmin / Sim.dx, c='w', ls=':', alpha=1.0)
    axes[1].axvline(Sim.xmax / Sim.dx, c='w', ls=':', alpha=1.0)
    axes[1].axvline(0.0                , c='w', ls=':', alpha=0.75)   
    axes[1].set_xlim(xlim[0], xlim[1])    
    
    fig.subplots_adjust(wspace=0.0)
    if save == True:
        fullpath = Sim.anal_dir + saveas + '_BPERP_{}'.format(lbl) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('B abs(t-x) Plot saved')
        plt.close('all')
    else:
        plt.show()
    return


def plot_abs_J(Sim, saveas='abs_plot', save=False, log=False, tmax=None,
               remove_ND=False):
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

    t  = getattr(Sim, 'field_sim_time')
    jx = getattr(Sim, 'jx')
    jy = getattr(Sim, 'jy')
    jz = getattr(Sim, 'jz')
    
    fontsize = 18
    font     = 'monospace'
    
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    mpl.rcParams['ytick.labelsize'] = tick_label_size 
    
    Jt     = np.sqrt(jx ** 2 + jy ** 2 + jz ** 2) * 1e3
    clabel = '$|J_T|$\nmA/m'
    suff   = '' 
        
    x    = Sim.E_nodes / Sim.dx
        
    if tmax is None:
        lbl = 'full'
    else:
        lbl = '{:04}'.format(tmax)
        
    if remove_ND == True:
        xlim = [Sim.xmin/Sim.dx, Sim.xmax/Sim.dx]
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
    ax.set_title(f'|J| spatiotemporal :: {Sim.series_name}[{Sim.run_num}]')
    
    ax.axvline(Sim.xmin     / Sim.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(Sim.xmax     / Sim.dx, c='w', ls=':', alpha=1.0)
    ax.axvline(0.0                , c='w', ls=':', alpha=0.75)   
    ax.set_xlim(xlim[0], xlim[1])    
    
    if save == True:
        fullpath = Sim.anal_dir + saveas + '_JTOT_{}{}{}'.format(lbl, suff, logsuff) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('J abs(t-x) Plot saved')
        plt.close('all')
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


def plot_fourier_mode_timeseries(Sim, it_max=None):
    '''
    Load helical components Bt pos/neg, extract By_pos
    For each snapshot in time, take spatial FFT of By_pos (similar to how helicity is done)
    Save these snapshots in array
    Plot single mode timeseries from this 2D array
    
    Test run: Seems relatively close qualitatively, with a smaller growth rate
                and a few of the modes not quite as large. This could be any 
                number of reasons - from the simulation method to the helicity.
                Will be interesting to compare direct to linear theory via the
                method outlined in Munoz et al. (2018).
    '''        
    ftime, Bt_pos, Bt_neg = ds.get_helical_components()
    radperiods     = ftime * Sim.gyfreq
    if it_max is None:
        it_max = ftime.shape[0]

    By_pos  = Bt_pos.real
    x       = np.linspace(0, Sim.NX*Sim.dx, Sim.NX)
    k_modes = np.fft.rfftfreq(x.shape[0], d=Sim.dx)
    Byk_2   = np.zeros((ftime.shape[0], k_modes.shape[0]), dtype=np.float64) 
    
    # Do time loop here. Could also put normalization flag
    for ii in range(ftime.shape[0]):
        Byk          = (1 / k_modes.shape[0]) * np.fft.rfft(By_pos[ii])
        Byk_2[ii, :] = (Byk * np.conj(Byk)).real / Sim.B_eq ** 2

    plt.ioff()
    fig, axes = plt.subplots(ncols=2, nrows=3, sharex=True, figsize=(15, 10))
    
    axes[0, 0].semilogy(radperiods, Byk_2[:, 1])
    axes[0, 0].set_title('m = 1')
    axes[0, 0].set_xlim(0, 100)
    axes[0, 0].set_ylim(1e-7, 1e-3) 
    
    axes[1, 0].semilogy(radperiods, Byk_2[:, 2])
    axes[1, 0].set_title('m = 2')
    axes[1, 0].set_xlim(0, 100)
    axes[1, 0].set_ylim(1e-6, 1e-1) 
    
    axes[2, 0].semilogy(radperiods, Byk_2[:, 3])
    axes[2, 0].set_title('m = 3')
    axes[2, 0].set_xlim(0, 100)
    axes[2, 0].set_ylim(1e-6, 1e-1) 
    
    axes[0, 1].semilogy(radperiods, Byk_2[:, 4])
    axes[0, 1].set_title('m = 4')
    axes[0, 1].set_xlim(0, 100)
    axes[0, 1].set_ylim(1e-6, 1e-0) 
    
    axes[1, 1].semilogy(radperiods, Byk_2[:, 5])
    axes[1, 1].set_title('m = 5')
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_ylim(1e-6, 1e-0) 
    
    axes[2, 1].semilogy(radperiods, Byk_2[:, 6])
    axes[2, 1].set_title('m = 6')
    axes[2, 1].set_xlim(0, 100)
    axes[2, 1].set_ylim(1e-6, 1e-0) 
    
    fig.savefig(Sim.anal_dir + 'fourier_modes.png')
    plt.close('all')
    return