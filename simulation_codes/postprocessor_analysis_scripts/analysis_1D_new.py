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
def plot_wx(component='By', saveas='wx_plot', linear_overlay=False):
    plt.ioff()
    wx = disp.get_wx(component)
    
    x  = np.arange(cf.NX)
    f  = np.fft.fftfreq(cf.time_seconds_field.shape[0], d=cf.dt_field)
    
    ## PLOT IT
    fig = plt.figure(1, figsize=(15, 10))
    ax  = fig.add_subplot(111)
    
    ax.pcolormesh(x, f[1:wx.shape[0] // 2], wx, cmap='nipy_spectral')      # Remove f[0] since FFT[0] >> FFT[1, 2, ... , k]
    
    proton_gyrofrequency = 1 / cf.gyperiod
    helium_gyrofrequency = 0.25  * proton_gyrofrequency
    oxygen_gyrofrequency = 0.125 * proton_gyrofrequency
    
    plt.axhline(proton_gyrofrequency, c='white')
    plt.axhline(helium_gyrofrequency, c='yellow')
    plt.axhline(oxygen_gyrofrequency, c='red')
    
    if linear_overlay == True:
        freqs, cgr, stop = disp.get_cgr_from_sim()
        max_idx          = np.where(cgr == cgr.max())
        max_lin_freq     = freqs[max_idx]
        plt.axhline(max_lin_freq, c='green', linestyle='--')

    ax.set_title(r'w-x Plot', fontsize=14)
    ax.set_ylabel(r'f (Hz)', rotation=0, labelpad=15)
    ax.set_xlabel('x (cell)')

    plt.xlim(None, 32)
    fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print('w-x Plot saved')
    return


def plot_kt(component='By', saveas='kt_plot'):
    plt.ioff()
    kt = disp.get_kt(component)
    
    k   = np.fft.fftfreq(cf.NX, cf.dx)
    k   = k[k>=0] * 1e6
    
    fig = plt.figure(1, figsize=(12, 8))
    ax  = fig.add_subplot(111)
    
    im1 = ax.pcolormesh(k[:k.shape[0]], cf.time_gperiods, kt, cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k] antialiased=True
    fig.colorbar(im1)
    ax.set_title(r'k-t Plot', fontsize=14)
    ax.set_ylabel(r'$\Omega_i t$', rotation=0)
    ax.set_xlabel(r'$k (m^{-1}) \times 10^6$')
    #ax.set_ylim(0, 15)
    
    fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print('K-T Plot saved')
    return


def plot_wk(component='By', saveas='wk_plot' , dispersion_overlay=False, plot=False, save=False):
    plt.ioff()
    
    k, f, wk = disp.get_wk(component)

    xlab = r'$k (\times 10^{-6}m^{-1})$'
    ylab = r'f (Hz)'

    fig = plt.figure(1, figsize=(12, 8))
    ax  = fig.add_subplot(111)

    ax.pcolormesh(k[1:], f[1:], np.log10(wk[1:, 1:].real), cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k]

    ax.set_title(r'$\omega/k$ Dispersion Plot for {}'.format(component), fontsize=14)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)

    M    = np.array([1., 4., 16.])
    cyc  = q * cf.B0 / (2 * np.pi * mp * M)
    for ii in range(3):
        if cf.species_present[ii] == True:
            ax.axhline(cyc[ii], linestyle='--', c='k')
    
    if dispersion_overlay == True:
        '''
        Some weird factor of about 2pi inaccuracy? Is this inherent to the sim? Or a problem
        with linear theory? Or problem with the analysis?
        '''
        k_vals, CPDR_solns, warm_solns = disp.get_dispersion_from_sim(k)
        for ii in range(CPDR_solns.shape[1]):
            ax.plot(k_vals, CPDR_solns[:, ii]*2*np.pi,      c='k', linestyle='--')
            ax.plot(k_vals, warm_solns[:, ii].real*2*np.pi, c='k', linestyle='-')
    
    if plot == True:
        plt.show()
    
    if save == True:
        fullpath = cf.anal_dir + saveas + '_{}'.format(component.lower()) + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('Dispersion Plot saved')
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


def plot_energies(normalize=True):
    mag_energy, electron_energy, particle_energy, total_energy = bk.get_energies()

    fig     = plt.figure(figsize=(15, 7))
    ax      = plt.subplot2grid((7, 7), (0, 0), colspan=6, rowspan=7)
    time_ax = cf.time_gperiods

    ax.plot(time_ax, mag_energy      / mag_energy[0],      label = r'$U_B$', c='green')
    ax.plot(time_ax, electron_energy / electron_energy[0], label = r'$U_e$', c='orange')
    ax.plot(time_ax, total_energy    / total_energy[0],    label = r'$Total$', c='k')
    
    for jj in range(cf.Nj):
        ax.plot(time_ax, particle_energy[:, jj] / particle_energy[0, jj],
                 label='$K_E$ {}'.format(cf.species_lbl[jj]), c=cf.temp_color[jj])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.2))
    fig.tight_layout()

    percent_ion = np.zeros(cf.Nj)
    for jj in range(cf.Nj):
        percent_ion[jj] = round(100.*(particle_energy[-1, jj] - particle_energy[0, jj]) / particle_energy[0, jj], 2)

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

    ax.set_xlabel('Time (Gyroperiods)')
    ax.set_xlim(0, cf.time_gperiods[-1])

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

    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close('all')
    
    print('Energy plot saved')
    return



def plot_helical_waterfall(title='', save=False, show=False, overwrite=False, it_max=None):
    By_raw         = cf.get_array('By')
    Bz_raw         = cf.get_array('Bz')
    
    if it_max is None:
        it_max = By_raw.shape[0]
    
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
    
    for ii in np.arange(cf.num_field_steps, it_max):
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

    for ii in np.arange(cf.num_field_steps, it_max):
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

    for ii in np.arange(cf.num_field_steps, it_max):
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
        fig2.savefig(cf.anal_dir + 'bz_helicity.png_t{}'.format(it_max), facecolor=fig2.get_facecolor(), edgecolor='none')
        
        fig3.subplots_adjust(bottom=0.07, top=0.96, left=0.04)
        fig3.subplots_adjust(wspace=0.05)
        ax6.set_yticklabels([])
        fig3.savefig(cf.anal_dir + 'raw_fields.png_t{}'.format(it_max), facecolor=fig3.get_facecolor(), edgecolor='none')
        
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


def plot_spatially_averaged_fields():
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
    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()
    
    ax1.plot(cf.time_radperiods, abs(Bz_raw).mean(axis=1))
    ax3.plot(cf.time_radperiods, abs(By_raw).mean(axis=1))
    ax5.plot(cf.time_radperiods, abs(Bx_raw).mean(axis=1))
    
    ax2.plot(cf.time_radperiods, abs(Bz_raw).mean(axis=1))
    ax4.plot(cf.time_radperiods, abs(By_raw).mean(axis=1))
    ax6.plot(cf.time_radperiods, abs(Bx_raw).mean(axis=1))
    
    ax1.set_ylabel(r'$\overline{|\delta B_z|}$ (nT)', rotation=0, labelpad=lpad)
    ax3.set_ylabel(r'$\overline{|\delta B_y|}$ (nT)', rotation=0, labelpad=lpad)
    ax5.set_ylabel(r'$\overline{|\delta B_x|}$ (nT)', rotation=0, labelpad=lpad)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticklabels([])
                
    for ax in [ax1, ax3, ax5]:
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 1.7)
        
    for ax in [ax2, ax4, ax6]:
        ax.set_xlim(0, cf.time_radperiods[-1])
        ax.set_ylim(0, 1.7)
        ax.set_yticklabels([])
            
    for ax in [ax5, ax6]:
        ax.set_label(r'Time $(\Omega^{-1})$')
            
    ax1.set_title('CAM_CL_1D : Omidi et al. (2010) parameters.')
    plt.figtext(0.125, 0.05, 'Total time: {:.{p}g}s'.format(cf.time_seconds_field[-1], p=6), fontweight='bold')
    fig.savefig(cf.anal_dir + 'sp_av_fields.png', facecolor=fig.get_facecolor(), edgecolor='none')
    return


def analyse_helicity(overwrite=False):
    By_raw         = cf.get_array('By')
    Bz_raw         = cf.get_array('Bz')
    Bt_pos, Bt_neg = bk.get_helical_components(overwrite)

    By_pos = Bt_pos.real
    By_neg = Bt_neg.real
    Bz_pos = Bt_pos.imag
    Bz_neg = Bt_neg.imag
    
    t_idx1 = 200
    t_idx2 = 205
    x_idx  = 768
    
    
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

    if True:
        '''
        Plot timeseries for raw, +ve, -ve helicities at single point
        '''
        plt.figure()
        plt.plot(cf.time_seconds_field, 1e9*By_raw[:, x_idx], label='Raw By')
        plt.plot(cf.time_seconds_field, 1e9*By_pos[:, x_idx], label='By+')
        plt.plot(cf.time_seconds_field, 1e9*By_neg[:, x_idx], label='By-')
        
        plt.title('Time-series at cell {}'.format(x_idx))
        plt.xlabel('Time (s)')
        plt.ylabel('By (nT)')
        plt.xlim(0, cf.time_seconds_field[-1])
        
        plt.legend()
    plt.show()
    return


def standard_analysis_package():
    for comp in ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']:
        plot_wx(component=comp, linear_overlay=False)
        plot_kt(component=comp, saveas='kt_plot')
        plot_wk(component=comp, dispersion_overlay=False, plot=False, save=False)
    
    plot_energies(normalize=True)
    plot_helical_waterfall(title='', save=False, show=False, overwrite=False, it_max=None)
    plot_spatially_averaged_fields()
    return

#%%

if __name__ == '__main__':
    #drive      = 'G://MODEL_RUNS//Josh_Runs//'
    drive      = 'F://'
    series     = 'omidi_1D'
    series_dir = '{}/runs//{}//'.format(drive, series)
    num_runs   = len([name for name in os.listdir(series_dir) if 'run_' in name])
    
    for run_num in [0]:
        print('Run {}'.format(run_num))
        cf.load_run(drive, series, run_num)
        #analyse_helicity()
        #plot_helical_waterfall(title='Series \'long_large_run\' : Run {}'.format(run_num), save=False)
        plot_spatially_averaged_fields()
        
        
