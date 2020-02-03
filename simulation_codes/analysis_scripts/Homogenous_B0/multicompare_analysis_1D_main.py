# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:56:34 2016

@author: c3134027
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

import analysis_backend              as bk
import multicompare_analysis_config  as cf

q   = 1.602e-19               # Elementary charge (C)
c   = 3e8                     # Speed of light (m/s)
me  = 9.11e-31                # Mass of electron (kg)
mp  = 1.67e-27                # Mass of proton (kg)
e   = -q                      # Electron charge (C)
mu0 = (4e-7) * np.pi          # Magnetic Permeability of Free Space (SI units)
kB  = 1.38065e-23             # Boltzmann's Constant (J/K)
e0  = 8.854e-12               # Epsilon naught - permittivity of free space

'''
Multi-run version for comparison. Some functions removed (like dispersion)
since it doesn't make sense to have this for multiple runs (use single run
instance instead, loop through run_num)

Aim: To populate this script with plotting routines ONLY. Separate out the 
processing/loading/calculation steps into other modules that can be called.

Assumes the existence of extracted array files. Don't run if these don't exist.
'''

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
        fig.savefig(series_dir + 'ion_energy_species_{}.png'.format(jj), facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close('all')
    return


def plot_spatially_averaged_fields(save=True, tmax=None):
    '''
    Field arrays are shaped like (time, space)
    '''
    plt.ioff()
    fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(figsize=(18, 10), nrows=3, ncols=2)
    fig.subplots_adjust(wspace=0, hspace=0)
        
    for ii in range(num_runs):
        cf.initialize_simulation_variables(series_dir  + 'run_{}/data/'.format(ii))
        arr_dir = series_dir  + 'run_{}/extracted/'.format(ii)

        Bx_raw = 1e9 * (cf.get_array(arr_dir, component='Bx')  - cf.B0)
        By_raw = 1e9 *  cf.get_array(arr_dir, component='By')
        Bz_raw = 1e9 *  cf.get_array(arr_dir, component='Bz')
          
        lpad   = 20
        
        ax1.plot(cf.time_seconds_field, abs(Bz_raw).mean(axis=1), label='Run {}'.format(ii), c=run_colors[ii])
        ax3.plot(cf.time_seconds_field, abs(By_raw).mean(axis=1), label='Run {}'.format(ii), c=run_colors[ii])
        ax5.plot(cf.time_seconds_field, abs(Bx_raw).mean(axis=1), label='Run {}'.format(ii), c=run_colors[ii])
        
        ax2.plot(cf.time_seconds_field, abs(Bz_raw).mean(axis=1), label='Run {}'.format(ii), c=run_colors[ii])
        ax4.plot(cf.time_seconds_field, abs(By_raw).mean(axis=1), label='Run {}'.format(ii), c=run_colors[ii])
        ax6.plot(cf.time_seconds_field, abs(Bx_raw).mean(axis=1), label='Run {}'.format(ii), c=run_colors[ii])
        
        ax1.set_ylabel(r'$\overline{|\delta B_z|}$ (nT)', rotation=0, labelpad=lpad)
        ax3.set_ylabel(r'$\overline{|\delta B_y|}$ (nT)', rotation=0, labelpad=lpad)
        ax5.set_ylabel(r'$\overline{|\delta B_x|}$ (nT)', rotation=0, labelpad=lpad)
        
        ax1.legend()
        
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
        
        if save == True and ii == (num_runs - 1):
            fig.savefig(series_dir + 'sp_av_fields.png', facecolor=fig.get_facecolor(), edgecolor='none')
            print('Spatially averaged B-field plot saved')
    return


def AGU_plot_spatially_averaged_fields(save=True, tmax=None):
    '''
    Field arrays are shaped like (time, space)
    '''
    tick_label_size = 14
    mpl.rcParams['xtick.labelsize'] = tick_label_size 
    
    fontsize = 18
    
    plt.ioff()
    fig, [ax1, ax2] = plt.subplots(2, figsize=(13, 6))
    fig.subplots_adjust(wspace=0, hspace=0)
        
    for ii, lbl in zip(range(num_runs), ['High Growth Case', 'Low Growth Case']):
        cf.initialize_simulation_variables(series_dir  + 'run_{}/data/'.format(ii))
        arr_dir = series_dir  + 'run_{}/extracted/'.format(ii)

        By_raw = 1e9 *  cf.get_array(arr_dir, component='By')
        Bz_raw = 1e9 *  cf.get_array(arr_dir, component='Bz')
          
        lpad   = 24
        
        ax1.plot(cf.time_seconds_field, abs(By_raw).mean(axis=1), label=lbl, c=run_colors[ii])
        ax2.plot(cf.time_seconds_field, abs(Bz_raw).mean(axis=1), label=lbl, c=run_colors[ii])

        ax1.set_ylabel('$\overline{|\delta B_y|}$\n (nT)', rotation=0, labelpad=lpad, fontsize=fontsize)
        ax2.set_ylabel('$\overline{|\delta B_z|}$\n (nT)', rotation=0, labelpad=lpad, fontsize=fontsize)
        
        ax1.legend(loc='lower right', prop={'size': fontsize}) 
        
        for ax in [ax1]:
            ax.set_xticklabels([])
        
        for ax in [ax1, ax2]:
            ax.set_xlim(0, cf.time_seconds_field[-1])
            ax.set_ylim(0, None)
        
        ax1.set_title('Spatially Averaged Fields :: Low/High Growth Parameters', fontsize=fontsize+4)
        ax2.set_xlabel(r'Time (s)', fontsize=fontsize)
          
        if save == True and ii == (num_runs - 1):
            fig.savefig(series_dir + 'AGU_sp_av_fields.png', facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
            print('Spatially averaged B-field plot saved')
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




#%%
if __name__ == '__main__':
    drive      = 'E://MODEL_RUNS//Josh_Runs//'
    #drive       = 'F://'
    series      = 'july_25_lingrowth'
    series_dir  = '{}/runs//{}//'.format(drive, series)
    num_runs    = len([name for name in os.listdir(series_dir) if 'run_' in name])
    dumb_offset = 0

    run_colors  = ['blue', 'red']

    AGU_plot_spatially_averaged_fields()
        
# =============================================================================
#         try:
#             single_point_field_timeseries()
#         except:
#             pass
#         
# =============================================================================
# =============================================================================
#         try:
#             plot_energies(normalize=True, save=True)
#         except:
#             pass
# =============================================================================
        

