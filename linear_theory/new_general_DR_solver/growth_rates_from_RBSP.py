# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:29:15 2019

@author: Yoshi
"""
import sys
sys.path.append('F://Google Drive//Uni//PhD 2017//Data//Scripts//')

import pdb
import os
import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates

import extract_parameters_from_data   as data
from   dispersion_solver_multispecies import get_dispersion_relations, plot_dispersion
from   rbsp_file_readers              import get_pearl_times

mp  = 1.673e-27
qi  = 1.602e-19
c   = 3.000e+08
mu0 = 4.000e-07*np.pi 
'''
 - Load plasma moments and magnetic field from file
 - Sort into arrays, call DR solver for each time
 - From result, calculate group velocity and CGR
 - Plot as timeseries using max values (or maybe some sort of 2D thing for each band
'''
def get_raw_data(rbsp_path):
    save_file = save_dir + 'extracted_data_new{}.npz'.format(save_string)
    
    if os.path.exists(save_file) == False:
        times, B0, cold_dens, hope_dens, hope_temp, hope_anis, spice_dens, spice_temp, spice_anis\
            = data.load_and_interpolate_plasma_params(time_start, time_end, probe, pad, rbsp_path=rbsp_path,
                                                      HM_filter_mhz=50)

        print('Saving raw data...')
        np.savez(save_file, times=times, B0=B0, cold_dens=cold_dens,
                 hope_dens=hope_dens,   hope_temp=hope_temp,   hope_anis=hope_anis,
                 spice_dens=spice_dens, spice_temp=spice_temp, spice_anis=spice_anis)
    else:
        print('Save data found, loading...')
        dp = np.load(save_file)
        
        times     = dp['times']
        B0        = dp['B0']
        cold_dens = dp['cold_dens']
        
        hope_dens = dp['hope_dens']
        hope_temp = dp['hope_temp']
        hope_anis = dp['hope_anis']
        
        spice_dens = dp['spice_dens']
        spice_temp = dp['spice_temp']
        spice_anis = dp['spice_anis']
    return times, B0, cold_dens, hope_dens, hope_temp, hope_anis, spice_dens, spice_temp, spice_anis


def extract_species_arrays(rbsp_path, time_start, time_end, probe, pad,
                           cmp, return_raw_ne=False, HM_filter_mhz=50):
    '''
    Data module only extracts the 3 component species dictionary from HOPE and RBSPICE 
    energetic measurements. This function creates the single axis arrays required to 
    go into the dispersion solver.
    
    All output values are in SI except temperature, which is in eV
    '''
    times, B0, cold_dens, hope_dens, hope_temp, hope_anis, spice_dens, spice_temp, spice_anis\
        = data.load_and_interpolate_plasma_params(time_start, time_end, probe, pad, rbsp_path=rbsp_path,
                                                  HM_filter_mhz=HM_filter_mhz)

    Nt       = times.shape[0]
    _density = np.zeros((9, Nt), dtype=float)
    _tper    = np.zeros((9, Nt), dtype=float)
    _ani     = np.zeros((9, Nt), dtype=float)

    _name    = np.array([   'cold $H^{+}$',    'cold $He^{+}$',    'cold $O^{+}$',
                            'HOPE $H^{+}$',    'HOPE $He^{+}$',    'HOPE $O^{+}$',
                         'RBSPICE $H^{+}$', 'RBSPICE $He^{+}$', 'RBSPICE $O^{+}$'])
    
    _mass    = np.array([1.0, 4.0, 16.0, 1.0, 4.0, 16.0, 1.0, 4.0, 16.0]) * mp
    _charge  = np.array([1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0]) * qi
    
    _density[0] = cold_dens * cmp[0];  _density[3] = hope_dens[0];  _density[6] = spice_dens[0];
    _density[1] = cold_dens * cmp[1];  _density[4] = hope_dens[1];  _density[7] = spice_dens[1];
    _density[2] = cold_dens * cmp[2];  _density[5] = hope_dens[2];  _density[8] = spice_dens[2];

    _tper[0] = 5.0; _tper[3] = hope_temp[0]; _tper[6] = spice_temp[0]
    _tper[1] = 5.0; _tper[4] = hope_temp[1]; _tper[7] = spice_temp[1]
    _tper[2] = 5.0; _tper[5] = hope_temp[2]; _tper[8] = spice_temp[2]

    _ani[0]  = 0.0; _ani[3]  = hope_anis[0]; _ani[6]  = spice_anis[0];
    _ani[1]  = 0.0; _ani[4]  = hope_anis[1]; _ani[7]  = spice_anis[1];
    _ani[2]  = 0.0; _ani[5]  = hope_anis[2]; _ani[8]  = spice_anis[2];

    if return_raw_ne == False:
        return times, B0, _name, _mass, _charge, _density, _tper, _ani
    else:
        return times, B0, _name, _mass, _charge, _density, _tper, _ani, cold_dens


def get_all_DRs(time_start, time_end, probe, pad, cmp, Nk=1000):

    times, B0, _name, _mass, _charge, _density, _tper, _ani, cold_dens = \
    extract_species_arrays(_rbsp_path, time_start, time_end, probe, pad, cmp, return_raw_ne=True)
    
    # Do O concentrations from 1-30 percent
    # Do He concentrations from 1-30 percent
    # Need a special save for 0 percent (2 species) runs
    for O_rat in [0.1]:#np.arange(0.01, 0.11, 0.01):
        for He_rat in [0.2]:#np.arange(0.01, 0.31, 0.01):
            H_rat = (1.0 - O_rat - He_rat)
            
            print('Cold composition {:.0f}/{:.0f}/{:.0f}'.format(H_rat*100, He_rat*100, O_rat*100))
            
            _density[0] = cold_dens * H_rat
            _density[1] = cold_dens * He_rat
            _density[2] = cold_dens * O_rat
            
            comp        = np.array([H_rat, He_rat, O_rat])
            data_path   = save_dir + 'DR_results_coldcomp_{:03}_{:03}_{:03}_{}.npz'.format(int(100*H_rat), int(100*He_rat), int(100*O_rat), save_string)

            if os.path.exists(data_path) == False:
                Nt         = times.shape[0]
                all_CPDR   = np.zeros((Nt, Nk, 3), dtype=np.float64)
                all_WPDR   = np.zeros((Nt, Nk, 3), dtype=np.complex128)
                all_k      = np.zeros((Nt, Nk)   , dtype=np.float64)
                for ii in range(Nt):
                    print('Calculating dispersion/growth relation for {}'.format(times[ii]))
                    
                    try:
                        k, CPDR, warm_solns = get_dispersion_relations(B0[ii], _name, _mass, _charge,
                                                          _density[:, ii], _tper[:, ii], _ani[:, ii],
                                                          kmin=0.0, kmax=1.0, Nk=Nk, norm_k_in=True)    
            
                        all_CPDR[ii, :, :] = CPDR 
                        all_WPDR[ii, :, :] = warm_solns
                        all_k[ii, :]       = k
                    except:
                        print('ERROR: Skipping to next time...')
                        all_CPDR[ii, :, :] = np.ones((Nk, 3), dtype=np.float64   ) * np.nan 
                        all_WPDR[ii, :, :] = np.ones((Nk, 3), dtype=np.complex128) * np.nan
                        all_k[ii, :]       = np.ones(Nk     , dtype=np.float64   ) * np.nan
                        
                print('Saving dispersion history...')
                np.savez(data_path, all_CPDR=all_CPDR, all_WPDR=all_WPDR, all_k=all_k, comp=comp)
            else:
                print('Dispersion results already exist, skipping...')
    return


def plot_growth_rate_with_time(times, k_vals, all_WPDR, save=False, short=False,
                               norm_w=False, B0=None, ccomp=[70, 20, 10]):
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    
    species_colors = ['r', 'b', 'g']
    band_labels    = [r'$H^+$', r'$He^+$', r'$O^+$']
    
    fontsize = 18
    
    Nt    = times.shape[0]
    max_k = np.zeros((Nt, 3))
    max_g = np.zeros((Nt, 3))
    
    dispersion  = all_WPDR.real
    growth_rate = all_WPDR.imag
    
    # Extract max k and max growth rate for each time, band
    for ii in range(Nt):
        for jj in range(3):
            #try:
                if any(np.isnan(dispersion[ii, :, jj]) == True):
                    max_k[ii, jj] = np.nan
                    max_g[ii, jj] = np.nan
                else:
                    max_idx       = np.where(growth_rate[ii, :, jj] == growth_rate[ii, :, jj].max())[0][0]
                    max_k[ii, jj] = k_vals[ii, max_idx]
                    max_g[ii, jj] = growth_rate[ii, max_idx, jj]
                
                if norm_w == True:
                    max_g[ii, jj] /= qi * B0[ii] / mp

    plt.ioff()
    fig, ax1 = plt.subplots(figsize=(13, 6))
    
    for ii in range(3):
        ax1.plot(times, 1e3*max_g[:, ii], color=species_colors[ii], label=band_labels[ii], marker='o')
    
    ax1.set_xlabel('Time (UT)', fontsize=fontsize)
    ax1.set_ylabel(r'Temporal Growth Rate ($\times 10^{-3} s^{-1}$)', fontsize=fontsize)
    ax1.set_title('EMIC Temporal Growth Rate :: RBSP-A Instruments :: {}/{}/{}'.format(*ccomp), fontsize=fontsize+4)
    ax1.legend(loc='upper left', prop={'size': fontsize}) 
    
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    if False:
        pearl_idx, pearl_times, pex = get_pearl_times(time_start, gdrive=gdrive)
        for ii in range(pearl_times.shape[0]):
            ax1.axvline(pearl_times[ii], c='k', linestyle='--', alpha=0.4)
    
    # Set xlim to show either just pearls, or whole event
    if short == True:
        ax1.set_xlim(np.datetime64('2013-07-25T21:25:00'), np.datetime64('2013-07-25T21:45:00'))
        figsave_path = save_dir + '_LT_timeseries_CC_{:03}_{:03}_{:03}_{}_short.png'.format(ccomp[0], ccomp[1], ccomp[2], save_string)
    else:
        ax1.set_xlim(time_start, time_end)
        figsave_path = save_dir + '_LT_timeseries_CC_{:03}_{:03}_{:03}_{}.png'.format(ccomp[0], ccomp[1], ccomp[2], save_string)

    for ii in range(len(timee)):
        mark = dayy + timee[ii]
        ax1.axvline(np.datetime64(mark), color='k', ls=':', alpha=0.4)

    if save == True:
        print('Saving {}'.format(figsave_path))
        fig.savefig(figsave_path, bbox_inches='tight')
        plt.close('all')
    else:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
        fig.autofmt_xdate()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

    return max_k, max_g


def load_and_plot_timeseries(ccomp=[70,20,10]):
    
    
    times, B0, name, mass, charge, density, tper, ani, cold_dens = \
        extract_species_arrays(time_start, time_end, probe, pad, ccomp, return_raw_ne=True)
    
    this_file = 'DR_results_coldcomp_098_001_001_20130725_2100_2200.npz'
    
    files = [this_file]#os.listdir(save_dir)
    for file in files:
        if file[-4:] == '.npz':
            cstring = file[20:31].split('_')
            cH      = int(cstring[0])
            cHe     = int(cstring[1])
            cO      = int(cstring[2])
            ccomp   = [cH, cHe, cO]
            
            # DRs stored as (Nt, Nk, solns)
            data_pointer = np.load(save_dir + file)
            all_WPDR     = data_pointer['all_WPDR']
            all_k        = data_pointer['all_k']
            
            plot_growth_rate_with_time(times, all_k, all_WPDR, save=False,
                                       short=True, norm_w=True, B0=B0, ccomp=ccomp)
    return


def load_and_overlay_multiple_timeseries(save=True, short=False, pearls=True):
    '''
    Maybe just do He/O? Use linestyles this time rather than colour. Save colour for 
    composition
    
    We have: 
        He from 1-30% in 1% increments
        O  from 1-10% in 1% increments
        
    Up to 30 values per O ratio
    Up to 10 values per He ratio
    
    Maybe do a plot overlaying GR for each He ratio
    
    WPDR     stored as complex128 with dimensions (Nt, Nk, band_solns=3)
    k values stored as float64    with dimensions (Nt, Nk)
    '''
    times, B0, cold_dens, hope_dens, hope_temp, hope_anis,\
        spice_dens, spice_temp, spice_anis = get_raw_data()
    
    files = os.listdir(save_dir)

    # Get Nt, Nk from dummy
    for file in files:
        if file[-4:] == '.npz': 
            data_pointer = np.load(save_dir + file)
            this_WPDR    = data_pointer['all_WPDR']
            this_k       = data_pointer['all_k']
            Nt           = this_k.shape[0]
            break
    
    tg_max = 60.0;  wn_max = 15.0; t_int = 5
    nO = 10; O_min = 1.0; O_max = 10.0; O_rats = np.arange(O_min, O_max)
    for He_rat in range(1, 31, 1):
        all_max_g = np.zeros((Nt, 3, nO), dtype=np.float)
        all_max_k = np.zeros((Nt, 3, nO), dtype=np.float)
        
        # Load all files with this He concentration
        for O_rat, ii in zip(O_rats, range(nO)):
            H_rat = 100 - He_rat - O_rat
            fname = 'DR_results_coldcomp_{:03}_{:03}_{:03}_20130725_2100_2200.npz'.format(int(H_rat), He_rat, int(O_rat))
            
            if os.path.exists(save_dir + fname):
                data_pointer = np.load(save_dir + fname)
                this_WPDR    = data_pointer['all_WPDR']
                this_k       = data_pointer['all_k']
                
                # Extract maximum growth rate/wavenumber for each band
                for mm in range(Nt):
                    for nn in range(3):
                        dispersion  = this_WPDR.real
                        growth_rate = this_WPDR.imag
                        
                        if any(np.isnan(dispersion[mm, :, nn]) == True):
                            all_max_k[mm, nn, ii] = np.nan
                            all_max_g[mm, nn, ii] = np.nan
                        else:
                            max_idx               = np.where(growth_rate[mm, :, nn] ==
                                                             growth_rate[mm, :, nn].max())[0][0]
                            all_max_k[mm, nn, ii] = this_k[mm, max_idx]
                            all_max_g[mm, nn, ii] = growth_rate[mm, max_idx, nn]
            else:
                print('No file: {}'.format(fname))

        
        #### PLOTTING ### (No normalizations) -- For possible O value
               
        ## GROWTH RATE ##
        plt.ioff()
        fig, axes = plt.subplots(2, figsize=(15, 10), sharex=True)
        
        tick_label_size = 12
        mpl.rcParams['xtick.labelsize'] = tick_label_size 
        
        band_labels = [r'$H^+$', r'$He^+$', r'$O^+$']
        fontsize    = 16
        
        # Note: The call to cm works as a fraction of the max because the argument has to be normalized to 
        # between 0-1. To have a non-zero minimum, you'd probably have to pass a norm instance.
        for O_rat, ii in zip(O_rats, range(nO)):
            for kk in range(1,3):
                axes[kk-1].plot(times, 1e3*all_max_g[:, kk, ii], color=plt.cm.plasma(O_rat/O_max))
                axes[kk-1].set_ylabel('$\gamma (\\times 10^{-3} s^{-1})$\n%s band' % band_labels[kk], fontsize=fontsize)
                axes[kk-1].set_ylim(0.0, tg_max)
        
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=O_max))
        plt.colorbar(sm, ax=axes[0]).set_label('$O^{+}$ percent', fontsize=fontsize)
        plt.colorbar(sm, ax=axes[1]).set_label('$O^{+}$ percent', fontsize=fontsize)
        
        axes[0].set_title('EMIC Temporal Growth Rate :: RBSP-A Instruments :: {}% He'.format(He_rat), fontsize=fontsize)
        axes[1].set_xlabel('Time (UT)', fontsize=fontsize)
        axes[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=t_int))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        if pearls == True:
            pearl_idx, pearl_times, pex = get_pearl_times(time_start, gdrive=gdrive)
            for ii in range(pearl_times.shape[0]):
                for ax in axes:
                    ax.axvline(pearl_times[ii], c='k', linestyle='--', alpha=0.4)
        
        # Set xlim to show either just pearls, or whole event
        if short == True:
            axes[1].set_xlim(np.datetime64('2013-07-25T21:27:00'), np.datetime64('2013-07-25T21:45:00'))
            figsave_path = save_dir + 'LT_compilation_overlay_He_{:02}_{}_GR_short.png'.format(He_rat, save_string)
        else:
            axes[1].set_xlim(time_start, time_end)
            figsave_path = save_dir + 'LT_compilation_overlay_He_{:02}_{}_GR.png'.format(He_rat, save_string)
        
        fig.subplots_adjust(hspace=0.03, wspace=0)
        if save==True:
            print('Saving {}'.format(figsave_path))
            fig.savefig(figsave_path, bbox_inches='tight')
            plt.close('all')
            
        ## WAVENUMBER ## 
        plt.ioff()
        fig, axes = plt.subplots(2, figsize=(15, 10), sharex=True)
        
        tick_label_size = 12
        mpl.rcParams['xtick.labelsize'] = tick_label_size 
        
        band_labels = [r'$H^+$', r'$He^+$', r'$O^+$']
        fontsize    = 16
        
        # Note: The call to cm works as a fraction of the max because the argument has to be normalized to 
        # between 0-1. To have a non-zero minimum, you'd probably have to pass a norm instance.
        for O_rat, ii in zip(O_rats, range(nO)):
            for kk in range(1,3):
                axes[kk-1].plot(times, 1e6*all_max_k[:, kk, ii], color=plt.cm.plasma(O_rat/O_max))
                axes[kk-1].set_ylabel('$\gamma (\\times 10^{-6} s^{-1})$\n%s band' % band_labels[kk], fontsize=fontsize)
                axes[kk-1].set_ylim(0.0, wn_max)
        
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=O_max))
        plt.colorbar(sm, ax=axes[0]).set_label('$O^{+}$ percent', fontsize=fontsize)
        plt.colorbar(sm, ax=axes[1]).set_label('$O^{+}$ percent', fontsize=fontsize)
        
        axes[0].set_title('EMIC Wavenumber :: RBSP-A Instruments :: {}% He'.format(He_rat), fontsize=fontsize)
        axes[1].set_xlabel('Time (UT)', fontsize=fontsize)
        axes[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=t_int))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        if pearls == True:
            pearl_idx, pearl_times, pex = get_pearl_times(time_start, gdrive=gdrive)
            for ii in range(pearl_times.shape[0]):
                for ax in axes:
                    ax.axvline(pearl_times[ii], c='k', linestyle='--', alpha=0.4)
        
        # Set xlim to show either just pearls, or whole event
        if short == True:
            axes[1].set_xlim(np.datetime64('2013-07-25T21:27:00'), np.datetime64('2013-07-25T21:45:00'))
            figsave_path = save_dir + 'LT_compilation_overlay_He_{:02}_{}_WN_short.png'.format(He_rat, save_string)
        else:
            axes[1].set_xlim(time_start, time_end)
            figsave_path = save_dir + 'LT_compilation_overlay_He_{:02}_{}_WN.png'.format(He_rat, save_string)
        
        fig.subplots_adjust(hspace=0.03, wspace=0)
        if save==True:
            print('Saving {}'.format(figsave_path))
            fig.savefig(figsave_path, bbox_inches='tight')
            plt.close('all')
    return


def plot_all_DRs(ccomp=[70, 20, 10]):
    '''
    Values loaded in order of Nt, Nk, solns
    '''
    times, B0, name, mass, charge, density, tper, ani, cold_dens = \
        extract_species_arrays(time_start, time_end, probe, pad, ccomp, return_raw_ne=True)
            
    file = 'DR_results_coldcomp_{:03}_{:03}_{:03}_{}.npz'.format(ccomp[0], ccomp[1], ccomp[2], save_string)
    
    data_pointer = np.load(save_dir + file)
    all_CPDR     = data_pointer['all_CPDR']
    all_WPDR     = data_pointer['all_WPDR']
    all_k        = data_pointer['all_k']
    
    DR_save_path = save_dir + '//ALL_{:03}_{:03}_{:03}_{}//'.format(ccomp[0], ccomp[1], ccomp[2], save_string)

    if os.path.exists(DR_save_path) == False:
        os.makedirs(DR_save_path)
    
    for ii in range(times.shape[0]):
        print('Plotting DR for {}'.format(times[ii]))
        PlasParams = {}
        PlasParams['va']    = B0[ii] / np.sqrt(mu0*(density[:, ii] * mass).sum())  # Alfven speed
        PlasParams['ne']    = density.sum()                                        # Electron number density
        PlasParams['p_cyc'] = qi*B0[ii] / mp                                       # Proton cyclotron frequency
    
        savename = DR_save_path + 'DR_{:04}.png'.format(ii)

        plot_dispersion(all_k[ii], all_CPDR[ii], all_WPDR[ii], PlasParams,
                        norm_k=True, norm_w=True, save=True, savename=savename,
                        title=times[ii], growth_only=True, glims=0.01)
    
    return


def check_if_exists():
    all_files = os.listdir(save_dir)
    
    for O_rat in range(1, 11, 1):
        for He_rat in range(1, 31, 1):
            H_rat = 100 - O_rat - He_rat
            fname = 'DR_results_coldcomp_{:03}_{:03}_{:03}_20130725_2100_2200.npz'.format(H_rat, He_rat, O_rat)
            if fname not in all_files:
                print('No file :: {:02}/{:02}/{:02}'.format(H_rat, He_rat, O_rat))
    return


if __name__ == '__main__':
    gdrive    = 'F://Google Drive//'
    _rbsp_path = 'G://DATA//RBSP//'
    dump_drive= 'G://'
    
    _Nk       = 500
    output    = 'save'
    overwrite = True
    figtext   = True
    
    time_start  = np.datetime64('2013-07-25T21:00:00')
    time_end    = np.datetime64('2013-07-25T22:00:00')
    probe       = 'a'
    pad         = 0
    
    date_string = time_start.astype(object).strftime('%Y%m%d')
    save_string = time_start.astype(object).strftime('%Y%m%d_%H%M_') + time_end.astype(object).strftime('%H%M')
    save_dir    = '{}NEW_LT//EVENT_{}//NEW_FIXED_DISPERSION_RESULTS//'.format(dump_drive, date_string)
    
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    
    dayy  = '2013-07-25T'
    timee = ['21:30:04.105',
            '21:30:50.105',
            '21:32:21.605',
            '21:32:48.105',
            '21:33:07.605',
            '21:34:06.605',
            '21:37:03.105',
            '21:39:07.605',
            '21:40:26.105',
            '21:41:05.605']
    
    get_all_DRs(time_start, time_end, probe, pad, cmp=[0.7, 0.2, 0.1], Nk=_Nk)
    
    #plot_all_DRs()
    #load_and_plot_timeseries()
    
    #check_if_exists()
        
    #load_and_overlay_multiple_timeseries()
    
    #get_raw_data()

        