# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:29:15 2019

@author: Yoshi
"""
import sys
sys.path.append('D://Google Drive//Uni//PhD 2017//Data//Scripts//')

#import pdb
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

def extract_species_arrays(time_start, time_end, probe, pad, cmp, return_raw_ne=False):
    '''
    Data module only extracts the 3 component species dictionary from HOPE and RBSPICE 
    energetic measurements. This function creates the single axis arrays required to 
    go into the dispersion solver.
    
    All output values are in SI except temperature, which is in eV
    '''
    
    
    times, B0, cold_dens, hope_dens, hope_temp, hope_anis, spice_dens, spice_temp, spice_anis\
        = data.load_and_interpolate_plasma_params(time_start, time_end, probe, pad, rbsp_path=rbsp_path)

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
    extract_species_arrays(time_start, time_end, probe, pad, cmp, return_raw_ne=True)
    
    # Do O concentrations from 1-30 percent
    # Do He concentrations from 1-30 percent
    # Need a special save for 0 percent (2 species) runs
    for O_rat in np.arange(0.01, 0.11, 0.01):
        for He_rat in np.arange(0.01, 0.31, 0.01):
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

        
    if output.lower() == 'save':
        print('Saving {}'.format(figsave_path))
        fig.savefig(figsave_path, bbox_inches='tight')
        plt.close('all')
    elif output.lower() == 'show':
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
        fig.autofmt_xdate()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
    else:
        plt.close('all')
    return max_k, max_g


def load_and_plot_timeseries(ccomp=[70,20,10]):
    files = os.listdir(save_dir)
    
    times, B0, name, mass, charge, density, tper, ani, cold_dens = \
        extract_species_arrays(time_start, time_end, probe, pad, ccomp, return_raw_ne=True)
    
    #this_file = 'DR_results_coldcomp_070_020_010_20130725_2100_2200.npz'
    
    for file in files:
        if file[-4:] == '.npz':
            try:
                cstring = file[20:31].split('_')
                cH      = int(cstring[0])
                cHe     = int(cstring[1])
                cO      = int(cstring[2])
                ccomp   = [cH, cHe, cO]
                
                # DRs stored as (Nt, Nk, solns)
                data_pointer = np.load(save_dir + file)
                all_WPDR     = data_pointer['all_WPDR']
                all_k        = data_pointer['all_k']
                
                plot_growth_rate_with_time(times, all_k, all_WPDR, save=True,
                                           short=True, norm_w=True, B0=B0, ccomp=ccomp)
            except:
                print('Error with file', file)
                continue
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


if __name__ == '__main__':
    gdrive    = 'D://Google Drive//'
    rbsp_path = 'E://DATA//RBSP//'
    dump_drive= 'D://'
    
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
    save_dir    = '{}NEW_LT//EVENT_{}//LINEAR_DISPERSION_RESULTS//'.format(dump_drive, date_string)
    
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    
    #get_all_DRs(time_start, time_end, probe, pad, cmp, Nk=_Nk)
    
    #plot_all_DRs()
    load_and_plot_timeseries()

        