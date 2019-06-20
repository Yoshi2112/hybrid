# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:56:34 2016

@author: c3134027
"""
import numpy as np
import matplotlib.pyplot as plt
import os

import analysis_backend as bk
import analysis_config  as cf
import dispersions      as disp


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
def plot_tx(component='By', saveas='tx_plot', plot=False, save=False, tmax=600):
    plt.ioff()

    tx = cf.get_array(component)
    x  = np.arange(cf.NX) * cf.dx
    t  = cf.time_radperiods_field
    
    if component[0] == 'B':
        tx *= 1e9
    else:
        tx *= 1e3
    
    ## PLOT IT
    fig = plt.figure(1, figsize=(15, 10))
    ax  = fig.add_subplot(111)

    im1 = ax.pcolormesh(x, t, tx, cmap='nipy_spectral')      # Remove f[0] since FFT[0] >> FFT[1, 2, ... , k]
    cb  = fig.colorbar(im1)
    
    if component[0] == 'B':
        cb.set_label('nT', rotation=0)
    else:
        cb.set_label('mV/m', rotation=0)

    ax.set_title('t-x Plot for {}'.format(component), fontsize=14)
    ax.set_ylabel(r't $(\Omega^{-1})$', rotation=0, labelpad=15)
    ax.set_xlabel('x (m)')
    
    if plot == True:
        plt.show()
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('t-x Plot saved')
        plt.close()
    return


def plot_wx(component='By', saveas='wx_plot', linear_overlay=False, plot=False, save=False, pcyc_mult=None):
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
    
    if plot == True:
        plt.show()
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('w-x Plot saved')
        plt.close()
    return


def plot_kt(component='By', saveas='kt_plot', plot=False, save=False):
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
    
    if plot == True:
        plt.show()
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close(fig)
        print('k-t Plot saved')
    return


def plot_wk(component='By', saveas='wk_plot' , dispersion_overlay=False, plot=False, save=False, pcyc_mult=None):
    plt.ioff()
    
    k, f, wk = disp.get_wk(component)

    xlab = r'$k (\times 10^{-6}m^{-1})$'
    ylab = r'f (Hz)'

    fig = plt.figure(1, figsize=(15, 10))
    ax  = fig.add_subplot(111)
    
    im1 = ax.pcolormesh(k[1:], f[1:], np.log10(wk[1:, 1:].real), cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k]
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
        
    ax.legend(loc=2, facecolor='grey')
    
    if plot == True:
        plt.show()
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('w-k saved')
        plt.close(fig)
    return


def waterfall_plot(arr, component_label):
    plt.ioff()

    amp   = 10.                 # Amplitude multiplier of waves:
    cells = np.arange(cf.NX)

    plt.figure()
    for ii in np.arange(cf.num_field_steps):
        plt.plot(cells, amp*(arr[ii] / arr.max()) + ii, c='k', alpha=0.25)

    plt.title('Run %s : %s Waterfall plot' % (run_num, component_label))
    plt.xlim(0, cf.NX)
    plt.ylim(0, None)
    plt.xlabel('Cell Number')
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


def plot_ion_energy_components(normalize=True, save=False, tmax=600):
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
    
    return

def plot_helical_waterfall(title='', save=False, show=False, overwrite=False, it_max=None):
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
        
    if show == True:
        plt.show()
    else:
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


def plot_spatially_averaged_fields(plot=False, save=True, tmax=600):
    '''
    Recreates Omidi et al. (2010) Figure 2
    
    Field arrays are shaped like (time, space)
    '''
    Bx_raw         = 1e9 * (cf.get_array('Bx')  - cf.B0)
    By_raw         = 1e9 *  cf.get_array('By')
    Bz_raw         = 1e9 *  cf.get_array('Bz')
      
    lpad   = 20
    
    plt.ioff()
    fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(figsize=(18, 10), nrows=3, ncols=2)
    fig.subplots_adjust(wspace=0, hspace=0)
    
    
    ax1.plot(cf.time_radperiods_field, abs(Bz_raw).mean(axis=1))
    ax3.plot(cf.time_radperiods_field, abs(By_raw).mean(axis=1))
    ax5.plot(cf.time_radperiods_field, abs(Bx_raw).mean(axis=1))
    
    ax2.plot(cf.time_radperiods_field, abs(Bz_raw).mean(axis=1))
    ax4.plot(cf.time_radperiods_field, abs(By_raw).mean(axis=1))
    ax6.plot(cf.time_radperiods_field, abs(Bx_raw).mean(axis=1))
    
    ax1.set_ylabel(r'$\overline{|\delta B_z|}$ (nT)', rotation=0, labelpad=lpad)
    ax3.set_ylabel(r'$\overline{|\delta B_y|}$ (nT)', rotation=0, labelpad=lpad)
    ax5.set_ylabel(r'$\overline{|\delta B_x|}$ (nT)', rotation=0, labelpad=lpad)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticklabels([])
                
    for ax in [ax1, ax3, ax5]:
        ax.set_xlim(0, tmax)
        ax.set_ylim(0, 1.7)
        
    for ax in [ax2, ax4, ax6]:
        ax.set_xlim(0, cf.time_radperiods_field[-1])
        ax.set_ylim(0, 1.7)
        ax.set_yticklabels([])
            
    for ax in [ax5, ax6]:
        ax.set_xlabel(r'Time $(\Omega^{-1})$')
      
    ax1.set_title('{}_1D : Omidi et al. (2010) parameters.'.format(cf.method_type))
    plt.figtext(0.125, 0.05, 'Total time: {:.{p}g}s'.format(cf.time_seconds_field[-1], p=6), fontweight='bold')
    
    if save == True:
        fig.savefig(cf.anal_dir + 'sp_av_fields.png', facecolor=fig.get_facecolor(), edgecolor='none')
        
    if plot == True:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    return


def plot_gyrophase():
    '''
    Either do just for particles or just for fields. To look at relationship between the two,
    need to interpolate and/or find times that the two coincide.
    '''
    
    
    return


def single_point_timeseries(cells=None, overwrite=False, save=True):
    '''
    Plot timeseries for raw, +ve, -ve helicities at single point
    
    Maybe do phases here too? (Although is that trivial for the helical components
    since they're separated based on a phase relationship between By,z ?)
    '''
    if cells is None:
        cells = np.arange(cf.NX)
    
    ts_folder = cf.anal_dir + '//single_point//'
    
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
            
            ax1.plot(cf.time_seconds_field, 1e9*By_raw[:, x_idx], label='Raw B{}'.format(component), c='blue')
            ax2.plot(cf.time_seconds_field, 1e9*By_pos[:, x_idx], label='B{}+'.format(component), c='green')
            ax2.plot(cf.time_seconds_field, 1e9*By_neg[:, x_idx], label='B{}-'.format(component), c='orange')
            
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
    return


def interpolate_fields_to_particle_time():
    '''
    For each particle timestep, interpolate field values
    '''
    bx, by, bz, ex, ey, ez, vex, vey, vez, te, jx, jy, jz, qdens = cf.get_array(get_all=True)

    
    time_particles = cf.time_seconds_particle
    time_fields    = cf.time_seconds_field
    
    pbx, pby, pbz, pex, pey, pez, pvex, pvey, pvez, pte, pjx, pjy, pjz, pqdens = [np.zeros((time_particles.shape[0], cf.NX))]*14
    
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
    
    test_cell = cf.NX//2
    plt.plot(time_fields, by[:, test_cell], c='b', marker='o')
    plt.scatter(time_particles, pby[:, test_cell], c='r', marker='x')
    return pbx, pby, pbz, pex, pey, pez, pvex, pvey, pvez, pte, pjx, pjy, pjz, pqdens


def analyse_helicity(overwrite=False, plot=False, save=True):
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


def summary_plots():
    '''
    Plot summary plot of raw values for each particle timestep
    Field values are interpolated to this point
    '''
    import pdb
    plt.ioff()
    pbx, pby, pbz, pex, pey, pez, pvex, pvey, pvez, pte, pjx, pjy, pjz, pqdens = interpolate_fields_to_particle_time()
    pdb.set_trace()
    va         = cf.B0 / np.sqrt(mu0*cf.ne*mp)                                  # Alfven speed: Assuming pure proton plasma
    qdens_norm = pqdens / (cf.density*cf.charge).sum()                          # Normalized change density
    for ii in range(cf.num_particle_steps):
        print('Creating summary plot for particle timestep {}'.format(ii))
        fig_size = 4, 7                                                             # Set figure grid dimensions
        fig = plt.figure(figsize=(20,10))                                           # Initialize Figure Space
        fig.patch.set_facecolor('w')  

        xp, vp = cf.load_particles(ii + cf.missing_t0_offset)
        
        pos       = xp / cf.dx                                                   # Cell particle position
        vel       = vp / va                                                      # Normalized velocity
    
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
        ax_den.set_ylabel(r'$\rho_c$', fontsize=14, rotation=0, labelpad=5)
        ax_den.set_ylim(0.5, 1.5)


        ax_Ex   = plt.subplot2grid(fig_size, (1, 3), colspan=3, sharex=ax_den)
        ax_Ex.plot(pex[ii]*1e6, color='red',   label=r'$E_x$')
        ax_Ex.plot(pey[ii]*1e6, color='cyan',  label=r'$E_y$')
        ax_Ex.plot(pez[ii]*1e6, color='black', label=r'$E_z$')
        ax_Ex.set_ylabel(r'$E (\mu V/m)$', labelpad=25, rotation=0, fontsize=14)
        ax_Ex.set_ylim(-200, 200)
        
        ax_By  = plt.subplot2grid(fig_size, (2, 3), colspan=3, sharex=ax_den)
        ax_B   = plt.subplot2grid(fig_size, (3, 3), colspan=3, sharex=ax_den)
        mag_B  = np.sqrt(pby[ii] ** 2 + pbz[ii] ** 2)
    
        ax_B.plot( mag_B*1e9, color='g')
        ax_By.plot(pby[ii]*1e9, color='g') 
        #ax_By.plot(pbz[ii]*1e9, color='b') 
        
        ax_B.set_ylim(0, cf.B0*1e9)
        ax_By.set_ylim(-cf.B0*1e9, cf.B0*1e9)
    
        ax_B.set_ylabel( r'$B_\perp (nT)$', rotation=0, labelpad=30, fontsize=14)
        ax_By.set_ylabel(r'$B_{y,z} (nT)$', rotation=0, labelpad=20, fontsize=14)
        ax_B.set_xlabel('Cell Number')
    
        for ax in [ax_den, ax_Ex, ax_By]:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_yticks(ax.get_yticks()[1:])
            
        for ax in [ax_den, ax_Ex, ax_By, ax_B]:
            qrt = cf.NX / (4.)
            ax.set_xlim(0,  cf.NX)
            ax.set_xticks(np.arange(0, cf.NX + qrt, qrt))
            ax.grid()
        
        plt.tight_layout(pad=1.0, w_pad=1.8)
        fig.subplots_adjust(hspace=0)
    
        filename = 'summ%05d.png' % ii
        path     = cf.anal_dir + '/summary_plots/'
        
        if os.path.exists(path) == False:                                   # Create data directory
            os.makedirs(path)
    
        fullpath = path + filename
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
        print('Plot saved'.format(ii))
        plt.close('all')
        break
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
        plot_wx(component=comp, saveas=disp_folder + 'wx_plot', save=True, linear_overlay=False,     pcyc_mult=1.1)
        plot_wk(component=comp, saveas=disp_folder + 'wk_plot', save=True, dispersion_overlay=True, pcyc_mult=1.1)
        plot_kt(component=comp, saveas=disp_folder + 'kt_plot', save=True)
        
    plot_energies(normalize=True, save=True)
    plot_ion_energy_components(save=True)
    plot_helical_waterfall(title='{}: Run {}'.format(series, run_num), save=True)
    single_point_timeseries()
    return

#%%

if __name__ == '__main__':
    drive      = 'G://MODEL_RUNS//Josh_Runs//'
    series     = 'uniform_time_varying'
    series_dir = '{}/runs//{}//'.format(drive, series)
    num_runs   = len([name for name in os.listdir(series_dir) if 'run_' in name])
    
    for run_num in [0]:
        print('Run {}'.format(run_num))
        cf.load_run(drive, series, run_num)
        #standard_analysis_package()
        #summary_plots()
        
# =============================================================================
#     bx = np.load('G://MODEL_RUNS//Josh_Runs//runs//uniform_time_varying//run_0//extracted//' + 'bx' +'_array.npy')
#     by = np.load('G://MODEL_RUNS//Josh_Runs//runs//uniform_time_varying//run_0//extracted//' + 'by' +'_array.npy')
#     print(bx)
#     print((bx - by).max())
# =============================================================================

