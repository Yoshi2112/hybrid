# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:29:15 2019

@author: Yoshi
"""
import sys
sys.path.append('F://Google Drive//Uni//PhD 2017//Data//Scripts//')

import pdb
import os
import numpy                        as np
import matplotlib.pyplot            as plt
import matplotlib.gridspec          as gs
import extract_parameters_from_data as data
import calculate_DR_chen_data       as cdr
from   matplotlib.lines         import Line2D
from   statistics               import mode
from   rbsp_file_readers        import get_pearl_times

def create_band_legend(fn_ax, labels, colors):
    legend_elements = []
    for label, color in zip(labels, colors):
        legend_elements.append(Line2D([0], [0], color=color, lw=1, label=label))
        
    new_legend = fn_ax.legend(handles=legend_elements, loc='upper right')#, bbox_to_anchor=(1, 0.6))
    return new_legend


def create_type_legend(fn_ax, labels, linestyles):
    legend_elements = []
    for label, style in zip(labels, linestyles):
        legend_elements.append(Line2D([0], [0], color='k', lw=1, label=label, linestyle=style))
        
    new_legend = fn_ax.legend(handles=legend_elements, loc='upper left')#, bbox_to_anchor=(1, 0.6))
    return new_legend


def set_frequencies_and_variables(field, ndensc, ndensw, t_perp, A, ndensw2, t_perp2, A2):
    global ndens, w_pc2, w_ps2, w_pe2, w_cyc, e_cyc, alfven
    global w_pw2,  alpha_par
    global w_pw2b, alpha_par2
    global c

    c     = 3E8                                 # m/s
    mp    = 1.673E-27                           # kg
    me    = 9.109E-31                           # kg
    q     = 1.602e-19                           # C
    qe    = -q
    e0    = 8.854e-12                           # F/m
    mu0   = (4e-7) * np.pi                      # units
    
    mi    = np.zeros(3)
    mi[0] = 1.  * mp
    mi[1] = 4.  * mp
    mi[2] = 16. * mp

    qi    = np.zeros(3)
    qi[0] = 1.0*q
    qi[1] = 1.0*q
    qi[2] = 1.0*q

    t_par   = t_perp  / (A + 1)                     # Tpar (in eV, converted to K in v_th calculation)
    t_par2  = t_perp2 / (A2 + 1)
    ndens   = ndensc + ndensw + ndensw2
    
    w_pc2   = ndensc      * qi ** 2 / (mi * e0)     # Cold      ion plasma frequencies (rad/s)
    w_pw2   = ndensw      * qi ** 2 / (mi * e0)     # Warm      ion plasma frequencies
    w_pw2b  = ndensw2     * qi ** 2 / (mi * e0)     # Warm2     ion plasma frequencies
    w_ps2   = ndens       * qi ** 2 / (mi * e0)     # Total     ion plasma frequencies
    w_pe2   = ndens.sum() * qe ** 2 / (me * e0)     # Electron  ion plasma frequencies
    
    w_cyc   =  q * field / (mi * 2 * np.pi)         # Ion      cyclotron frequencies (Hz)
    e_cyc   =  qe* field / (me * 2 * np.pi)         # Electron cyclotron frequency
    
    rho       = (ndens * mi).sum()                  # Mass density (kg/m3)
    alfven    = field / np.sqrt(mu0 * rho)          # Alfven speed (m/s)
   
    alpha_par  = np.sqrt(2.0 * q * t_par  / mi)     # Thermal velocity in m/s (make relativistic?)  
    alpha_par2 = np.sqrt(2.0 * q * t_par2 / mi)     # Thermal velocity in m/s (make relativistic?)
    return


def plot_dispersion_multiple(ax_disp, ax_growth, k_vals, CPDR_solns, warm_solns, k_isnormalized=False,
                             w_isnormalized=False, save=False, savepath=None, alpha=1.0):
    '''
    Plots the CPDR and WPDR nicely as per Wang et al 2016. Can plot multiple dispersion/growth curves for varying parameters.
    
    INPUT:
        k_vals     -- Wavenumber values in /m or normalized to p_cyc/v_A
        CPDR_solns -- Cold-plasma frequencies in Hz or normalized to p_cyc
        WPDR_solns -- Warm-plasma frequencies in Hz or normalized to p_cyc. 
                   -- .real is dispersion relation, .imag is growth rate vs. k
    '''
    
    
    # Plot dispersion #
    for ii in range(3):
        ax_disp.plot(k_vals[1:]*1e6, CPDR_solns[1:, ii],      c=species_colors[ii], linestyle='--', label='Cold', alpha=alpha)
        ax_disp.plot(k_vals[1:]*1e6, warm_solns[1:, ii].real, c=species_colors[ii], linestyle='-',  label='Warm', alpha=alpha)
        ax_disp.axhline(w_cyc[ii], c='k', linestyle=':')
    
    type_label = ['Cold Plasma Approx.', 'Hot Plasma Approx.', 'Cyclotron Frequencies']
    type_style = ['--', '-', ':']
    type_legend = create_type_legend(ax_disp, type_label, type_style)
    ax_disp.add_artist(type_legend)
    
    # Plot growth #
    
    band_legend = create_band_legend(ax_growth, band_labels, species_colors)
    ax_growth.add_artist(band_legend)
    
    for ii in range(3):
        ax_growth.plot(k_vals[1:]*1e6, warm_solns[1:, ii].imag, c=species_colors[ii], linestyle='-',  label='Growth', alpha=alpha)
    ax_growth.axhline(0, c='k', linestyle=':')
    return


def set_figure_text(ax, ii, param_dict):
    field     = param_dict['field'][ii]
    ndensc    = param_dict['ndensc'][:, ii]
    ndensw    = param_dict['ndensw'][:, ii]
    temp_perp = param_dict['temp_perp'][:, ii]
    A         = param_dict['A'][:, ii]
    ndensw2   = param_dict['ndensw2'][:, ii]
    temp_perp2= param_dict['temp_perp2'][:, ii]
    A2        = param_dict['A2'][:, ii]
    
    n0         = (ndensc + ndensw + ndensw2).sum()
    c_percent  = ndensc.sum() / n0 * 100.
    w_percent  = ndensw.sum() / n0 * 100.
    w2_percent = ndensw2.sum() / n0 * 100.
    
    TPER_kev  = temp_perp  * 1e-3
    TPER_kev2 = temp_perp2 * 1e-3
    
    font    = 'monospace'
    fsize   = 10
    top     = 1.0               # Top of text block
    left    = 1.15              # Left boundary of text block
    
    ax.text(left, top - 0.02, '$B_0 = ${:5.2f}nT'.format(field),           transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, top - 0.05, '$n_0 = $%.1f$cm^{-3}$' % n0, transform=ax.transAxes, fontsize=fsize, fontname=font)

    v_space = 0.03              # Vertical spacing between lines
    
    c_top   = top - 0.15         # 'Cold' top
    w_top   = top - 0.35         # 'Warm' top
    h_top   = top - 0.55         # 'Hot' (Warmer) top
    
    
    # Cold Table
    ax.text(left, c_top + 1*v_space, 'Cold Population ({:.3f}%)'.format(c_percent), transform=ax.transAxes, fontsize=fsize+2, fontname=font, fontweight='bold')
    ax.text(left + 0.05, c_top, r'    $n_c (cm^{-3})$  Cold Composition', transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, c_top - 1*v_space, ' H+:   {:>7.2f}     {:>3.0f}%'.format(round(ndensc[0], 2), cmp[0]), transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, c_top - 2*v_space, 'He+:   {:>7.2f}     {:>3.0f}%'.format(round(ndensc[1], 2), cmp[1]), transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, c_top - 3*v_space, ' O+:   {:>7.2f}     {:>3.0f}%'.format(round(ndensc[2], 2), cmp[2]), transform=ax.transAxes, fontsize=fsize, fontname=font)

    
    # Warm Table
    ax.text(left, w_top + 1*v_space, 'Warm Population 1 ({:.3f}%)'.format(w_percent), transform=ax.transAxes, fontsize=fsize+2, fontname=font, fontweight='bold')
    ax.text(left + 0.05, w_top, r'    $n_i (cm^{-3})$    $T_{\perp} (keV)$    $A_i$ ', transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, w_top - 1*v_space, ' H+:   {:>7.3f}   {:>7.2f}   {:>6.3f}'.format(round(ndensw[0], 3), round(TPER_kev[0], 2), round(A[0], 3)) , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, w_top - 2*v_space, 'He+:   {:>7.3f}   {:>7.2f}   {:>6.3f}'.format(round(ndensw[1], 3), round(TPER_kev[1], 2), round(A[1], 3)) , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, w_top - 3*v_space, ' O+:   {:>7.3f}   {:>7.2f}   {:>6.3f}'.format(round(ndensw[2], 3), round(TPER_kev[2], 2), round(A[2], 3)) , transform=ax.transAxes, fontsize=fsize, fontname=font)


    # Hot Table
    ax.text(left, h_top + 1*v_space, 'Warm Population 2 ({:.3f}%)'.format(w2_percent), transform=ax.transAxes, fontsize=fsize+2, fontname=font, fontweight='bold')
    ax.text(left + 0.05, h_top, r'    $n_i (cm^{-3})$    $T_{\perp} (keV)$    $A_i$ ', transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, h_top - 1*v_space, ' H+:   {:>7.3f}   {:>7.2f}   {:>6.3f}'.format(round(ndensw2[0], 3), round(TPER_kev2[0], 2), round(A2[0], 3)) , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, h_top - 2*v_space, 'He+:   {:>7.3f}   {:>7.2f}   {:>6.3f}'.format(round(ndensw2[1], 3), round(TPER_kev2[1], 2), round(A2[1], 3)) , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, h_top - 3*v_space, ' O+:   {:>7.3f}   {:>7.2f}   {:>6.3f}'.format(round(ndensw2[2], 3), round(TPER_kev2[2], 2), round(A2[2], 3)) , transform=ax.transAxes, fontsize=fsize, fontname=font)
    return


def get_all_DRs():
    param_dict   = data.load_and_interpolate_plasma_params(time_start, time_end, probe, pad, cold_composition=cmp)
    
    if os.path.exists(data_path) == True:
        print('Save file found: Loading...')
        data_pointer = np.load(data_path)
        all_CPDR     = data_pointer['all_CPDR']
        all_WPDR     = data_pointer['all_WPDR']
        all_k        = data_pointer['all_k']
    else:
        Nt         = param_dict['times'].shape[0]
        all_CPDR   = np.zeros((Nt, _Nk, 3), dtype=np.float64)
        all_WPDR   = np.zeros((Nt, _Nk, 3), dtype=np.complex128)
        all_k      = np.zeros((Nt, _Nk)   , dtype=np.float64)
        for ii in range(Nt):
            print('Calculating dispersion/growth relation for {}'.format(param_dict['times'][ii]))
            
            try:
                k, CPDR, warm_solns = cdr.get_dispersion_relation(
                        param_dict['field'][ii],
                        param_dict['ndensc'][:, ii],
                        param_dict['ndensw'][:, ii],
                        param_dict['temp_perp'][:, ii],
                        param_dict['A'][:, ii],
                        param_dict['ndensw2'][:, ii],
                        param_dict['temp_perp2'][:, ii],
                        param_dict['A2'][:, ii],
                        Nk=_Nk)    
                
                all_CPDR[ii, :, :] = CPDR 
                all_WPDR[ii, :, :] = warm_solns
                all_k[ii, :]       = k
            except:
                all_CPDR[ii, :, :] = np.ones((_Nk, 3), dtype=np.float64   ) * np.nan 
                all_WPDR[ii, :, :] = np.ones((_Nk, 3), dtype=np.complex128) * np.nan
                all_k[ii, :]       = np.ones(_Nk     , dtype=np.float64   ) * np.nan
                
            if ii == Nt - 1:
               print('Saving dispersion history...')
               np.savez(data_path, all_CPDR=all_CPDR, all_WPDR=all_WPDR, all_k=all_k)
    return all_CPDR, all_WPDR, all_k, param_dict


def plot_all_DRs(param_dict, all_k, all_CPDR, all_WPDR):
    Nt = param_dict['times'].shape[0]
    for ii in range(Nt):
        set_frequencies_and_variables(
                        param_dict['field'][ii] * 1e-9,
                        param_dict['ndensc'][:, ii] * 1e6,
                        param_dict['ndensw'][:, ii] * 1e6,
                        param_dict['temp_perp'][:, ii],
                        param_dict['A'][:, ii],
                        param_dict['ndensw2'][:, ii] * 1e6,
                        param_dict['temp_perp2'][:, ii],
                        param_dict['A2'][:, ii])
        
        time  = param_dict['times'][ii]
        k_vals= all_k[ii]
        CPDR  = all_CPDR[ii]
        WPDR  = all_WPDR[ii]

        ##################
        ## PLOTTING BIT ##
        ##################
        figsave_path = save_dir + 'linear_{}_{}.png'.format(save_string, ii)
        
        if os.path.exists(figsave_path) == True and overwrite == False:
            print('Plot already done, skipping...')
            continue
        
        plt.ioff()
        fig    = plt.figure(figsize=(16, 10))
        grid   = gs.GridSpec(1, 2)
        
        ax1    = fig.add_subplot(grid[0, 0])
        ax2    = fig.add_subplot(grid[0, 1])
        
        fig.text(0.34, 0.974, '{}'.format(time))
        
        plot_dispersion_multiple(ax1, ax2, k_vals, CPDR, WPDR, save=False, savepath=None, w_isnormalized=True)    

        ax1.set_title('Dispersion Relation')
        ax1.set_xlabel(r'$k (\times 10^{-6} m^{-1})$')
        ax1.set_ylabel(r'$\omega${}'.format(' (Hz)'))
        
        ax1.set_xlim(0, k_vals[-1]*1e6)
        ax1.set_ylim(0, w_cyc[0] * 1.1)
        
        ax2.set_title('Temporal Growth Rate')
        ax2.set_xlabel(r'$k (\times 10^{-6}m^{-1})$')
        ax2.set_ylabel(r'$\gamma (s^{-1})$')
        ax2.set_xlim(0, k_vals[-1]*1e6)
        
        ax2.set_ylim(None, None)
        
        y_thres_min = -0.05;  y_thres_max = 0.05
        if ax2.get_ylim()[0] < y_thres_min:
            y_min = y_thres_min
        else:
            y_min = y_thres_min
            
        if ax2.get_ylim()[0] > y_thres_max:
            y_max = None
        else:
            y_max = y_thres_max
        
        ax2.set_ylim(y_min, y_max)
        
        ax1.minorticks_on()
        ax2.minorticks_on() 
        
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        plt.setp(ax2.get_xticklabels()[0], visible=False)
        
        #%%
        if figtext == True:
            set_figure_text(ax2, ii, param_dict)
        
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0, right=0.75)
        
        if output == 'show':
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()
        elif output == 'save':
            figsave_path = save_dir + 'linear_{}_{}.png'.format(save_string, ii)
            print('Saving {}'.format(figsave_path))
            fig.savefig(figsave_path)
            plt.close('all')
    return


def plot_growth_rate_with_time(times, k_vals, growth_rate, per_tol=70, save=False):
    Nt    = times.shape[0]
    max_k = np.zeros((Nt, 3))
    max_g = np.zeros((Nt, 3))
    
    # Extract max k and max growth rate for each time, band
    for ii in range(Nt):
        for jj in range(3):
            # Filter out cases where the band shits itself (replace with np.nan)
            gr_mode = mode(growth_rate[ii, :, jj])
            num_bad = growth_rate[ii, :, jj][growth_rate[ii, :, jj] == gr_mode].shape[0]
            per_bad = num_bad / k_vals.shape[0] * 100.
            
            if per_bad > per_tol:
                max_k[ii, jj] = np.nan
                max_g[ii, jj] = np.nan
            else:
                try:
                    max_idx       = np.where(growth_rate[ii, :, jj] == growth_rate[ii, :, jj].max())[0][0]
                    max_k[ii, jj] = k_vals[ii, max_idx]
                    max_g[ii, jj] = growth_rate[ii, max_idx, jj]
                except:
                    pdb.set_trace()
    
    pearl_times, pex = get_pearl_times(time_start)
    
    plt.ioff()
    fig    = plt.figure(figsize=(16, 10))
    grid   = gs.GridSpec(1, 1)
    ax1    = fig.add_subplot(grid[0, 0])
    
    for ii in range(3):
        ax1.plot(times, max_g[:, ii], color=species_colors[ii], label=band_labels[ii])
    
    ax1.set_xlabel('Time (UT)')
    ax1.set_ylabel('Temporal Growth Rate ($s^{-1}$)')
    ax1.set_title('Growth rates (per band) with Cold Plasma Composition [{}, {}, {}]'.format(*cmp))
    ax1.legend(loc='upper right') 
    
    for ii in range(pearl_times.shape[0]):
        ax1.axvline(pearl_times[ii], c='k', linestyle='--', alpha=0.4)
    
        
    if save == True:
        figsave_path = save_dir + '_LT_timeseries_CC_{:03}_{:03}_{:03}_{}.png'.format(cmp[0], cmp[1], cmp[2], save_string)
        print('Saving {}'.format(figsave_path))
        fig.savefig(figsave_path)
        plt.close('all')
    else:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
    return max_k, max_g


if __name__ == '__main__':
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    _Nk       = 5000
    output    = 'save'
    overwrite = False
    figtext   = True
    
    time_start  = np.datetime64('2013-07-25T21:25:00')
    time_end    = np.datetime64('2013-07-25T21:47:00')
    probe       = 'a'
    pad         = 0
    
    date_string = time_start.astype(object).strftime('%Y%m%d')
    save_string = time_start.astype(object).strftime('%Y%m%d_%H%M')
    
    species_colors = ['r', 'b', 'g']
    band_labels    = [r'$H^+$', r'$He^+$', r'$O^+$']
    
    #cmp            = np.array([60, 30, 10])
    for cmp in [np.array([70, 20, 10])]:
        save_dir    = 'G://NEW_LT//event_{}_old//LINEAR_THEORY_CC_{:03}_{:03}_{:03}//'.format(date_string, cmp[0], cmp[1], cmp[2])
        data_path   = save_dir + '_chen_dispersion_{:03}_{:03}_{:03}_{}.npz'.format(cmp[0], cmp[1], cmp[2], save_string)
        
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        
        _all_CPDR, _all_WPDR, _all_k, _param_dict = get_all_DRs()
        #plot_all_DRs(_param_dict, _all_k, _all_CPDR, _all_WPDR)
        #_max_k, _max_g = plot_growth_rate_with_time(_param_dict['times'], _all_k, _all_WPDR.imag, save=True)