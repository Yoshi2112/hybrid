# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:56:34 2016

@author: c3134027
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
'''
def plot_tx(component='By', saveas='tx_plot', save=False, tmax=None, yunits='seconds', add_ND=False, ND=None):
    plt.ioff()    
    
    tx = cf.get_array(component)

    fontsize = 18
    font     = 'monospace'
    
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    mpl.rcParams['ytick.labelsize'] = tick_label_size 
    
    x  = np.arange(cf.NX)
    
    if yunits == 'seconds':
        t    = cf.time_seconds_field
        ylab = 't (s)'
    elif yunits == 'radperiods':
        t    = cf.time_radperiods_field
        ylab = 'Time\n$(\Omega^{-1})$'
    else:
        t    = cf.time_gperiods_field
        ylab = 'Time\n$(\f^{-1})$'
    
    if component[0] == 'B':
        tx *= 1e9
    else:
        tx *= 1e3

    if add_ND == True:
        
        if ND is None:
            ND = x.shape[0]
            
        new_tx = np.zeros((tx.shape[0], cf.NX + 2*ND), dtype=tx.dtype)
        new_tx[:, cf.NX:cf.NX + ND] = tx
        
        new_x = np.arange(cf.NX + 2*ND)
    else:
        new_tx = tx
        new_x  = x

    
    ## PLOT IT
    fig, ax = plt.subplots(1, figsize=(15, 10))

    vmin = tx.min()
    vmax = tx.max()
    im1 = ax.pcolormesh(new_x, t, new_tx, cmap='nipy_spectral', vmin=vmin, vmax=vmax)
    cb  = fig.colorbar(im1)
    
    if component[0] == 'B':
        cb.set_label('nT', rotation=0,   family=font, fontsize=fontsize)
    else:
        cb.set_label('mV/m', rotation=0, family=font, fontsize=fontsize, labelpad=20)

    ax.set_title('Field Plot :: {} Component'.format(component), fontsize=fontsize, family=font)
    ax.set_ylabel(ylab   , fontsize=fontsize, family=font, labelpad=15, rotation=0)
    ax.set_xlabel('x ($\Delta x$)', fontsize=fontsize, family=font)
    ax.set_ylim(0, tmax)
    
    if add_ND == True:
        ax.axvline(ND,         c='w', ls=':', alpha=1.0)
        ax.axvline(ND + cf.NX, c='w', ls=':', alpha=1.0)
        
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('t-x Plot saved')
        plt.close('all')
    else:
        plt.show()
    return


def plot_wx(component='By', saveas='wx_plot', linear_overlay=False, save=False, pcyc_mult=None):
    plt.ioff()
    wx = disp.get_wx(component)
    
    x  = np.arange(cf.NX)
    f  = np.fft.rfftfreq(cf.time_seconds_field.shape[0], d=cf.dt_field)
    
    ## PLOT IT
    fig = plt.figure(1, figsize=(15, 10))
    ax  = fig.add_subplot(111)

    im1 = ax.pcolormesh(x, f, wx, cmap='nipy_spectral')      # Remove f[0] since FFT[0] >> FFT[1, 2, ... , k]
    fig.colorbar(im1)
    
    lbl  = [r'$\Omega_{H^+}$', r'$\Omega_{He^+}$', r'$\Omega_{O^+}$']
    clr  = ['white', 'yellow', 'red']    
    M    = np.array([1., 4., 16.])
    cyc  = q * cf.B0 / (2 * np.pi * mp * M)
    for ii in range(3):
        if cf.species_present[ii] == True:
            ax.axhline(cyc[ii], linestyle='--', c=clr[ii], label=lbl[ii])
    
    if linear_overlay == True:
        try:
            freqs, cgr, stop = disp.get_cgr_from_sim()
            max_idx          = np.where(cgr == cgr.max())
            max_lin_freq     = freqs[max_idx]
            plt.axhline(max_lin_freq, c='green', linestyle='--', label='CGR')
        except:
            pass

    ax.set_title(r'w-x Plot', fontsize=14)
    ax.set_ylabel(r'f (Hz)', rotation=0, labelpad=15)
    ax.set_xlabel('x (cell)')
    
    if pcyc_mult is not None:
        ax.set_ylim(0, pcyc_mult*cyc[0])
    else:
        ax.set_ylim(0, None)
        
    ax.legend(loc=2, facecolor='grey')
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('w-x Plot saved')
        plt.close('all')
    return


def plot_kt(component='By', saveas='kt_plot', save=False):
    plt.ioff()
    kt = disp.get_kt(component)
    
    k   = np.fft.fftfreq(cf.NX, cf.dx)
    k   = k[k>=0] * 1e6
    
    fig = plt.figure(1, figsize=(15, 10))
    ax  = fig.add_subplot(111)
    
    im1 = ax.pcolormesh(k[:k.shape[0]], cf.time_gperiods_field, kt, cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k] antialiased=True
    fig.colorbar(im1)
    ax.set_title(r'k-t Plot', fontsize=14)
    ax.set_ylabel(r'$\Omega_i t$', rotation=0)
    ax.set_xlabel(r'$k (m^{-1}) \times 10^6$')
    #ax.set_ylim(0, 15)
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close('all')
        print('k-t Plot saved')
    return


def plot_wk(component='By', saveas='wk_plot' , dispersion_overlay=False, save=False, pcyc_mult=None, zero_cold=False):
    plt.ioff()
    
    k, f, wk = disp.get_wk(component)

    xfac = 1e6
    xlab = r'$k (\times 10^{-6}m^{-1})$'
    ylab = r'f (Hz)'

    fig = plt.figure(1, figsize=(15, 10))
    ax  = fig.add_subplot(111)
    
    im1 = ax.pcolormesh(xfac*k[1:], f[1:], np.log10(wk[1:, 1:].real), cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k]
    fig.colorbar(im1)
    ax.set_title(r'$\omega/k$ Dispersion Plot for {}'.format(component), fontsize=14)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    
    clr  = ['white', 'yellow', 'red'] 
    lbl  = [r'$\Omega_{H^+}$', r'$\Omega_{He^+}$', r'$\Omega_{O^+}$']
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
        # k (actually beta) from FFT linear. LT requires angular k (k = 2pi*beta)
        k_vals, CPDR_solns, warm_solns = disp.get_linear_dispersion_from_sim(k, zero_cold=zero_cold)
        for ii in range(CPDR_solns.shape[1]):
            ax.plot(xfac*k_vals, CPDR_solns[:, ii],      c='k', linestyle='--', label='CPDR')
            ax.plot(xfac*k_vals, warm_solns[:, ii].real, c='k', linestyle='-',  label='WPDR')
        
    if True:
        # Plot Alfven velocity on here just to see
        alfven_line = k_vals * cf.va
        ax.plot(xfac*k_vals, alfven_line, c='blue', linestyle=':', label='$v_A$')
        
    ax.legend(loc=2, facecolor='grey')
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('w-k saved')
        plt.close('all')
    else:
        plt.show()
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

    fig     = plt.figure(figsize=(15, 7))
    ax      = plt.subplot2grid((7, 7), (0, 0), colspan=6, rowspan=7)

    ax.plot(cf.time_seconds_field, mag_energy      / mag_energy[0],      label = r'$U_B$', c='green')
    ax.plot(cf.time_seconds_field, electron_energy / electron_energy[0], label = r'$U_e$', c='orange')
    ax.plot(cf.time_seconds_field, total_energy    / total_energy[0],    label = r'$Total$', c='k')
    
    for jj in range(cf.Nj):
        ax.plot(cf.time_seconds_particle, particle_energy[:, jj, 0] / particle_energy[0, jj, 0],
                 label=r'$K_{E\parallel}$ %s' % cf.species_lbl[jj], c=cf.temp_color[jj], linestyle=':')
        
        ax.plot(cf.time_seconds_particle, particle_energy[:, jj, 1] / particle_energy[0, jj, 1],
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
    ax.set_xlim(0, cf.time_seconds_field[-1])

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
    By_raw         = cf.get_array('By')
    Bz_raw         = cf.get_array('Bz')
    
    if it_max is None:
        it_max = cf.num_field_steps
    
    Bt_pos, Bt_neg = bk.get_helical_components(overwrite)

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
    
    bx, by, bz, ex, ey, ez, vex, vey, vez, te, jx, jy, jz, qdens = cf.get_array(get_all=True)
    
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

    bx, by, bz, ex, ey, ez, vex, vey, vez, te, jx, jy, jz, qdens = cf.get_array(get_all=True)
    
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


def interpolate_fields_to_particle_time():
    '''
    For each particle timestep, interpolate field values
    
    RECODE THIS TO USE NP.INTERPOLATE()
    '''
    bx, by, bz, ex, ey, ez, vex, vey, vez, te, jx, jy, jz, qdens = cf.get_array(get_all=True)

    time_particles = cf.time_seconds_particle
    time_fields    = cf.time_seconds_field
    
    pbx, pby, pbz, pex, pey, pez, pvex, pvey, pvez, pte, pjx, pjy, pjz, pqdens = \
    [np.zeros((time_particles.shape[0], cf.NX)) for _ in range(14)]
    
    for ii in range(time_particles.shape[0]):
        this_time    = time_particles[ii]                   # Target interpolant
        diff         = abs(this_time - time_fields)         # Difference matrix
        nearest_idx  = np.where(diff == diff.min())[0][0]   # Index of nearest value
        
        if time_fields[nearest_idx] < this_time:
            case = 1
            lidx = nearest_idx
            uidx = nearest_idx + 1
        elif time_fields[nearest_idx] > this_time:
            case = 2
            uidx = nearest_idx
            lidx = nearest_idx - 1
        else:
            case    = 3
            for arr_out, arr_in in zip([pbx, pby, pbz, pex, pey, pez, pvex, pvey, pvez, pte, pjx, pjy, pjz, pqdens], 
                                   [bx,  by,  bz,  ex,  ey,  ez,  vex,  vey,  vez,  te,  jx,  jy,  jz,  qdens]):
                arr_out[ii] = arr_in[nearest_idx]
            continue
        
        if not time_fields[lidx] <= this_time <= time_fields[uidx]:
            print('WARNING: Interpolation issue :: {}'.format(case))
        
        ufac = (this_time - time_fields[lidx]) / cf.dt_field
        lfac = 1.0 - ufac
        
        # Now do the actual interpolation: Example here, extend (or loop?) it to the other ones later.
        for arr_out, arr_in in zip([pbx, pby, pbz, pex, pey, pez, pvex, pvey, pvez, pte, pjx, pjy, pjz, pqdens], 
                                   [bx,  by,  bz,  ex,  ey,  ez,  vex,  vey,  vez,  te,  jx,  jy,  jz,  qdens]):
            arr_out[ii] = lfac*arr_in[lidx] + ufac*arr_in[uidx]

    return pbx, pby, pbz, pex, pey, pez, pvex, pvey, pvez, pte, pjx, pjy, pjz, pqdens


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


def summary_plots(save=True):
    '''
    Plot summary plot of raw values for each particle timestep
    Field values are interpolated to this point
    '''    
    path = cf.anal_dir + '/summary_plots/'
        
    if os.path.exists(path) == False:                                   # Create data directory
        os.makedirs(path)
            
    plt.ioff()
    pbx, pby, pbz, pex, pey, pez, pvex, pvey, pvez, pte, pjx, pjy, pjz, pqdens = interpolate_fields_to_particle_time()
    qdens_norm = pqdens / (cf.density*cf.charge).sum()                          # Normalized change density
    for ii in range(dumb_offset, cf.num_particle_steps+dumb_offset):
        filename = 'summ%05d.png' % ii
        fullpath = path + filename
        
        if os.path.exists(fullpath):
            print('Summary plot already present for timestep [{}]{}'.format(run_num, ii))
            continue
        
        print('Creating summary plot for particle timestep [{}]{}'.format(run_num, ii))
        fig_size = 4, 7                                                             # Set figure grid dimensions
        fig = plt.figure(figsize=(20,10))                                           # Initialize Figure Space
        fig.patch.set_facecolor('w')   
        xp, vp = cf.load_particles(ii + cf.missing_t0_offset)
        
        pos       = xp / cf.dx                                                   # Cell particle position
        vel       = vp / cf.va                                                      # Normalized velocity
    
        ax_vx   = plt.subplot2grid(fig_size, (0, 0), rowspan=2, colspan=3)
        ax_vy   = plt.subplot2grid(fig_size, (2, 0), rowspan=2, colspan=3)
        for jj in range(cf.Nj):
            ax_vx.scatter(pos[cf.idx_bounds[jj, 0]: cf.idx_bounds[jj, 1]], vel[0, cf.idx_bounds[jj, 0]: cf.idx_bounds[jj, 1]], s=3, c=cf.temp_color[jj], lw=0, label=cf.species_lbl[jj])
            ax_vy.scatter(pos[cf.idx_bounds[jj, 0]: cf.idx_bounds[jj, 1]], vel[1, cf.idx_bounds[jj, 0]: cf.idx_bounds[jj, 1]], s=3, c=cf.temp_color[jj], lw=0)
    
        ax_vx.legend()
        ax_vx.set_title(r'Particle velocities vs. Position (x)')
        ax_vy.set_xlabel(r'Cell', labelpad=10)
        ax_vx.set_ylabel(r'$\frac{v_x}{v_A}$', rotation=0)
        ax_vy.set_ylabel(r'$\frac{v_y}{v_A}$', rotation=0)
        
        plt.setp(ax_vx.get_xticklabels(), visible=False)
        ax_vx.set_yticks(ax_vx.get_yticks()[1:])
    
        for ax in [ax_vy, ax_vx]:
            ax.set_xlim(0, cf.NX)
            ax.set_ylim(-10, 10)
    
        ax_den  = plt.subplot2grid(fig_size, (0, 3), colspan=3)
        ax_den.plot(qdens_norm[ii], color='green')
                
        ax_den.set_title('Charge Density and Fields')
        ax_den.set_ylabel(r'$\frac{\rho_c}{\rho_{c0}}$', fontsize=14, rotation=0, labelpad=20)
        ax_den.set_ylim(0.5, 1.5)


        ax_Ex   = plt.subplot2grid(fig_size, (1, 3), colspan=3, sharex=ax_den)
        ax_Ex.plot(pex[ii]*1e3, color='red',   label=r'$E_x$')
        ax_Ex.plot(pey[ii]*1e3, color='cyan',  label=r'$E_y$')
        ax_Ex.plot(pez[ii]*1e3, color='black', label=r'$E_z$')
        ax_Ex.set_ylabel(r'$E (mV/m)$', labelpad=25, rotation=0, fontsize=14)
        ax_Ex.set_ylim(-30, 30)
        ax_Ex.legend(loc=4, ncol=3)
        
        ax_By  = plt.subplot2grid(fig_size, (2, 3), colspan=3, sharex=ax_den)
        ax_B   = plt.subplot2grid(fig_size, (3, 3), colspan=3, sharex=ax_den)
        mag_B  = np.sqrt(pby[ii] ** 2 + pbz[ii] ** 2)
    
        ax_B.plot( mag_B*1e9, color='g')
        ax_By.plot(pby[ii]*1e9, color='g',   label=r'$B_y$') 
        ax_By.plot(pbz[ii]*1e9, color='b',   label=r'$B_z$') 
        ax_By.legend(loc=4, ncol=2)
        
        ax_B.set_ylim(0, cf.B0*1e9)
        ax_By.set_ylim(-cf.B0*1e9, cf.B0*1e9)
    
        ax_B.set_ylabel( r'$B_\perp (nT)$', rotation=0, labelpad=30, fontsize=14)
        ax_By.set_ylabel(r'$B_{y,z} (nT)$', rotation=0, labelpad=20, fontsize=14)
        ax_B.set_xlabel('Cell Number')
        
        
        ax_HM    = ax_B.twinx()
        
        ax_HM.plot(pbx[ii]*1e9, c='red')
        ax_HM.set_ylabel(r'$B_x (nT)$', rotation=0, labelpad=30, fontsize=14, color='r')
        
        if cf.HM_amplitude != 0:
            ax_HM.set_ylim((cf.B0 - cf.HM_amplitude)*1e9, (cf.B0 + cf.HM_amplitude)*1e9)
        else:
            ax_HM.set_ylim(cf.B0*1e9 - 1, cf.B0*1e9 + 1)
    
        for ax in [ax_den, ax_Ex, ax_By]:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_yticks(ax.get_yticks()[1:])
            
        for ax in [ax_den, ax_Ex, ax_By, ax_B]:
            grad = cf.NX / (4.)
            ax.set_xlim(0,  cf.NX)
            ax.set_xticks(np.arange(0, cf.NX + grad, grad))
            ax.grid()
        
        plt.tight_layout(pad=1.0, w_pad=1.8)
        fig.subplots_adjust(hspace=0)
        
        ###################
        ### FIGURE TEXT ###
        ###################
        anisotropy = (cf.Tper / cf.Tpar - 1).round(1)
        beta_per   = (2*(4e-7*np.pi)*(1.381e-23)*cf.Tper*cf.ne / (cf.B0**2)).round(1)
        beta_e     = round((2*(4e-7*np.pi)*(1.381e-23)*cf.Te0*cf.ne  / (cf.B0**2)), 2)
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
        plt.figtext(0.855, top - 3*gap, '{} particles/cell'.format(cf.cellpart), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 4*gap, '{}'.format(estring), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 5*gap, '', fontsize=fontsize, family='monospace')
        
        plt.figtext(0.855, top - 6*gap, 'B0      : {}nT'.format(cf.B0*1e9), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 7*gap, 'n0      : {}cc'.format(cf.ne/1e6), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 8*gap, 'HM_amp  : {}nT'.format(cf.HM_amplitude*1e9), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 9*gap, 'HM_freq : {}mHz'.format(cf.HM_frequency*1e3), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 10*gap, '', fontsize=fontsize, family='monospace')
        
        plt.figtext(0.855, top - 11*gap, r'$\theta$       : %d deg' % cf.theta, fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 12*gap, r'$\beta_e$      : %.2f' % beta_e, fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 13*gap, 'dx      : {}km'.format(round(cf.dx/1e3, 2)), fontsize=fontsize, family='monospace')
        plt.figtext(0.855, top - 14*gap, '', fontsize=fontsize, family='monospace')
        
        ptop = top - 15*gap
        plt.figtext(0.855, ptop, 'Particle Parameters', fontsize=fontsize, family='monospace', fontweight='bold')
        plt.figtext(0.855, ptop - gap, ' SPECIES  ANI  XBET   VDR  RDNS', fontsize=fontsize-2, family='monospace')
        for jj in range(cf.Nj):
            plt.figtext(0.855       , ptop - (jj + 2)*gap, '{:>10}  {:>3}  {:>4}  {:>4}  {:<5}'.format(
                    cf.species_lbl[jj], anisotropy[jj], beta_per[jj], vdrift[jj], rdens[jj]),
                    fontsize=fontsize-2, family='monospace')
 
        time_top = 0.1
        plt.figtext(0.88, time_top - 0*gap, 't_seconds   : {:>10}'.format(round(cf.time_seconds_particle[ii], 3))   , fontsize=fontsize, family='monospace')
        plt.figtext(0.88, time_top - 1*gap, 't_gperiod   : {:>10}'.format(round(cf.time_gperiods_particle[ii], 3))  , fontsize=fontsize, family='monospace')
        plt.figtext(0.88, time_top - 2*gap, 't_radperiod : {:>10}'.format(round(cf.time_radperiods_particle[ii], 3)), fontsize=fontsize, family='monospace')
    
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
        plot_tx(component=comp, saveas=disp_folder + 'tx_plot', save=True)
        plot_wx(component=comp, saveas=disp_folder + 'wx_plot', save=True, linear_overlay=False,    pcyc_mult=1.25)
        plot_wk(component=comp, saveas=disp_folder + 'wk_plot', save=True, dispersion_overlay=True, pcyc_mult=1.25)
        plot_kt(component=comp, saveas=disp_folder + 'kt_plot', save=True)
        for zero_cold in [True, False]:
            plot_wk_polished(component=comp, dispersion_overlay=True, save=True,
                             pcyc_mult=1.25, zero_cold=zero_cold, xmax=20)
    
    if False:    
        plot_energies(normalize=True, save=True)
        plot_ion_energy_components(save=True, tmax=1./cf.HM_frequency)
        plot_helical_waterfall(title='{}: Run {}'.format(series, run_num), save=True)
        single_point_helicity_timeseries()
        plot_spatially_averaged_fields()
        single_point_field_timeseries(tmax=1./cf.HM_frequency)
    return


def do_all_dynamic_spectra(ymax=None):
    for component in ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']:
        for cell in np.arange(cf.NX):
            plot_dynamic_spectra(component=component, ymax=ymax, save=True, cell=cell)
    return


def plot_wk_polished(component='By', saveas='wk_plot_thesis', dispersion_overlay=False, save=False,
                     pcyc_mult=None, xmax=None, plot_alfven=False, zero_cold=False, overwrite=True):
    disp_folder = 'dispersion_plots_thesis/'
        
    if os.path.exists(cf.anal_dir + disp_folder) == False:
        os.makedirs(cf.anal_dir + disp_folder)
    
    plt.ioff()
    
    fontsize = 18
    font     = 'monospace'
    
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    mpl.rcParams['ytick.labelsize'] = tick_label_size 
    
    k, f, wk = disp.get_wk(component)

    xfac = 1e6
    xlab = '$\mathtt{k (\\times 10^{-6}m^{-1})}$'
    ylab = 'f\n(Hz)'
    
    if component[0].upper() == 'B':
        clab = 'Pwr\n$\left(\\frac{nT^2}{Hz}\\right)$'
    else:
        clab = 'Pwr\n$\left(\\frac{mV^2}{m^2Hz}\\right)$'

    fig = plt.figure(1, figsize=(15, 10))
    ax  = fig.add_subplot(111)
    
    im1 = ax.pcolormesh(xfac*k[1:], f[1:], wk[1:, 1:].real, cmap='jet',
                        norm=colors.LogNorm(vmin=wk[1:, 1:].real.min(),
                                            vmax=wk[1:, 1:].real.max()))      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k]
    
    fig.colorbar(im1, extend='both', fraction=0.05).set_label(clab, rotation=0, fontsize=fontsize, family=font, labelpad=30)
    ax.set_title(r'$\omega/k$ Plot :: {} Component :: Linear Theory Overlay'.format(component.upper()),
                 fontsize=fontsize, family=font)
    ax.set_ylabel(ylab, fontsize=fontsize, family=font, rotation=0, labelpad=30)
    ax.set_xlabel(xlab, fontsize=fontsize, family=font)
    
    clr  = ['black', 'green', 'red'] 
    lbl  = [r'$f_{H^+}$', r'$f_{He^+}$', r'$f_{O^+}$']
    M    = np.array([1., 4., 16.])
    cyc  = q * cf.B0 / (2 * np.pi * mp * M)
    
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transAxes, ax.transData) # the x coords of this transformation are axes, and the
            # y coord are data
    
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
    
    alpha=0.5
    if dispersion_overlay == True:
        k_vals, CPDR_solns, warm_solns = disp.get_linear_dispersion_from_sim(k, zero_cold=zero_cold)
        for ii in range(CPDR_solns.shape[1]):
            ax.plot(xfac*k_vals, CPDR_solns[:, ii],      c='k', linestyle='--', label='CPDR' if ii == 0 else '', alpha=alpha)
            ax.plot(xfac*k_vals, warm_solns[:, ii].real, c='k', linestyle='-',  label='WPDR' if ii == 0 else '', alpha=alpha)
      
    if plot_alfven == True:
        # Plot Alfven velocity on here just to see
        alfven_line = k * cf.va
        ax.plot(xfac*k, alfven_line, c='blue', linestyle=':', label='$v_A$')
        
    ax.legend(loc='upper right', facecolor='white', prop={'size': fontsize-2, 'family':font})
        
    if save == True:
        zero_suff = '' if zero_cold is False else 'zero'
        fullpath  = cf.anal_dir + disp_folder + saveas + '_{}'.format(component.lower()) + '_{}'.format(zero_suff)
        save_path = fullpath + '.png'
        
        if overwrite == False:
            count = 1
            while os.path.exists(save_path) == True:
                print('Save file exists, incrementing...')
                save_path = fullpath + '_{}.png'.format(count)
                count += 1
            
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('w-k for component {} saved'.format(component.lower()))
        plt.close('all')
    else:
        plt.show()
    return



#%%
if __name__ == '__main__':
    drive       = 'F:'
    series      = 'compare_old_new_LTcheck_homogenous'
    series_dir  = '{}/runs//{}//'.format(drive, series)
    num_runs    = len([name for name in os.listdir(series_dir) if 'run_' in name])
    dumb_offset = 0
    
    for run_num in [2]:
        print('Run {}'.format(run_num))
        cf.load_run(drive, series, run_num)

        #standard_analysis_package()
        
        #single_point_both_fields_AGU()
        
        #do_all_dynamic_spectra(ymax=1.0)
        #do_all_dynamic_spectra(ymax=None)
        
        #By_raw         = cf.get_array('By') * 1e9
        #Bz_raw         = cf.get_array('Bz') * 1e9
        #ggg.get_linear_growth(By_raw, Bz_raw)

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
        

