# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:29:15 2019

@author: Yoshi
"""
import pdb
import os
import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt

import extract_parameters_from_data as data
from dispersion_solver_multispecies import get_dispersion_relations

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
    
    To Do:
        Could make some way to alter the cold composition just before calling
        the DR solver - so that the values doesn't need to keep being read
        from the files anymore. Maybe save the time-based data in an array
        since that'll be quicker?
        
        -- Output raw time-based ne as well. Can overwrite the arrays then on my own in another function
    '''
    mp = 1.673e-27
    qi = 1.602e-19
    
    times, B0, cold_dens, hope_dens, hope_temp, hope_anis, spice_dens, spice_temp, spice_anis\
        = data.load_and_interpolate_plasma_params(time_start, time_end, probe, pad)
    
    Nt       = times.shape[0]
    _density = np.zeros((9, Nt), dtype=float)
    _tper    = np.zeros((9, Nt), dtype=float)
    _ani     = np.zeros((9, Nt), dtype=float)

    _name    = np.array(['cold H'   , 'cold He'   , 'cold O'   ,
                         'HOPE H'   , 'HOPE He'   , 'HOPE O'   ,
                         'RBSPICE H', 'RBSPICE He', 'RBSPICE O'])
    
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


def get_all_DRs(data_path, time_start, time_end, probe, pad, cmp, Nk=1000):

    times, B0, _name, _mass, _charge, _density, _tper, _ani, cold_dens = \
    extract_species_arrays(time_start, time_end, probe, pad, cmp, return_raw_ne=True)
    
    # Do O concentrations from 1-30 percent
    # Do He concentrations from 1-30 percent
    # Need a special save for 0 percent (2 species) runs
    for O_rat in np.arange(0.05, 0.35, 0.05):
        for He_rat in np.arange(0.05, 0.35, 0.05):
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
    return all_CPDR, all_WPDR, all_k


def plot_growth_rate_with_time(times, k_vals, growth_rate, per_tol=50, output='none', short=False):
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    
    fontsize = 18
    
    Nt    = times.shape[0]
    max_k = np.zeros((Nt, 3))
    max_g = np.zeros((Nt, 3))
    
    # Extract max k and max growth rate for each time, band
    for ii in range(Nt):
        for jj in range(3):
            max_idx       = np.where(growth_rate[ii, :, jj] == growth_rate[ii, :, jj].max())[0][0]
            max_k[ii, jj] = k_vals[ii, max_idx]
            max_g[ii, jj] = growth_rate[ii, max_idx, jj]
    
    #pearl_idx, pearl_times, pex = get_pearl_times(time_start, gdrive=gdrive)
    
    plt.ioff()
    fig    = plt.figure(figsize=(13, 6))
    grid   = gs.GridSpec(1, 1)
    ax1    = fig.add_subplot(grid[0, 0])
    
    for ii in range(3):
        ax1.plot(times, 1e3*max_g[:, ii], color=species_colors[ii], label=band_labels[ii], marker='o')
    
    ax1.set_xlabel('Time (UT)', fontsize=fontsize)
    ax1.set_ylabel(r'Temporal Growth Rate ($\times 10^{-3} s^{-1}$)', fontsize=fontsize)
    ax1.set_title('EMIC Temporal Growth Rate :: RBSP-A Instruments'.format(*cmp), fontsize=fontsize+4)
    ax1.legend(loc='upper left', prop={'size': fontsize}) 
    
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    for ii in range(pearl_times.shape[0]):
        ax1.axvline(pearl_times[ii], c='k', linestyle='--', alpha=0.4)
    
    if short == True:
        ax1.set_xlim(np.datetime64('2013-07-25T21:25:00'), np.datetime64('2013-07-25T21:45:00'))
        figsave_path = save_dir + '_LT_timeseries_CC_{:03}_{:03}_{:03}_{}_short.png'.format(cmp[0], cmp[1], cmp[2], save_string)
    else:
        ax1.set_xlim(time_start, time_end)
        figsave_path = save_dir + '_LT_timeseries_CC_{:03}_{:03}_{:03}_{}.png'.format(cmp[0], cmp[1], cmp[2], save_string)

        
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


if __name__ == '__main__':
    rbsp_path = 'G://DATA//RBSP//'
    
    _Nk       = 500
    output    = 'save'
    overwrite = True
    figtext   = True
    
    time_start  = np.datetime64('2013-07-25T21:00:00')
    time_end    = np.datetime64('2013-07-25T22:00:00')
    probe       = 'a'
    pad         = 0
    
    cmp         = np.array([0.70, 0.20, 0.10])
    
    date_string = time_start.astype(object).strftime('%Y%m%d')
    save_string = time_start.astype(object).strftime('%Y%m%d_%H%M_') + time_end.astype(object).strftime('%H%M')
    save_dir    = 'G://NEW_LT//EVENT_{}//LINEAR_DISPERSION_RESULTS//'.format(date_string, cmp[0], cmp[1], cmp[2])
    data_path   = save_dir + 'DR_results_coldcomp_{:03}_{:03}_{:03}_{}.npz'.format(cmp[0], cmp[1], cmp[2], save_string)
    
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    
    plot_start = np.datetime64('2013-07-25T21:00:00')
    plot_end   = np.datetime64('2013-07-25T22:00:00')
    
    _all_CPDR, _all_WPDR, _all_k = get_all_DRs(data_path, time_start, time_end, probe, pad, cmp, Nk=_Nk)
    #plot_all_DRs(_param_dict, _all_k, _all_CPDR, _all_WPDR)
    
# =============================================================================
#     if os.path.exists(data_path) == True:
#         print('Save file found: Loading...')
#         data_pointer = np.load(data_path)
#         all_CPDR     = data_pointer['all_CPDR']
#         all_WPDR     = data_pointer['all_WPDR']
#         all_k        = data_pointer['all_k']
#     else:
# =============================================================================