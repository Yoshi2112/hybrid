# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:39:35 2021

@author: Yoshi
"""
import os, warnings, pdb
import numpy as np
import numba as nb
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import spacepy as sp

kB = 1.381e-23
qp = 1.602e-19
mp = 1.673e-27
c  = 3e8

def boundary_idx64(time_arr, start, end):
    '''Returns index values corresponding to the locations of the start/end times in a numpy time array, if specified times are np.datetime64'''
    idx1 = np.where(abs(time_arr - start) == np.min(abs(time_arr - start)))[0][0] 
    idx2 = np.where(abs(time_arr - end)   == np.min(abs(time_arr - end)))[0][0]
    return (idx1, idx2) 


@nb.njit()
def calc_distro_2D(DENS, VPER, VPAR, VTH_PER, VTH_PAR):
    DIST  = np.zeros((VPER.shape[0], VPAR.shape[0]))
    
    out = DENS / np.sqrt(8 * np.pi**3 * VTH_PER**4 * VTH_PAR**2)
    for aa in nb.prange(VPER.shape[0]):
        for bb in nb.prange(VPAR.shape[0]):
            exp = np.exp(-0.5*VPER[aa]**2/VTH_PER**2
                         -0.5*VPAR[bb]**2/VTH_PAR**2)
            DIST[aa, bb] = out*exp
    return DIST


@nb.njit()
def integrate_distro_2D(DIST, VPERP, VPARA, MASS, Emin, Emax):
    dvperp = VPERP[1] - VPERP[0]
    dvpara = VPARA[1] - VPARA[0]
    
    SUM = 0.0
    for aa in nb.prange(DIST.shape[0]):
        for bb in nb.prange(DIST.shape[1]):
            VEL2 = VPERP[aa]**2 + VPARA[bb]**2
            ENRG = MASS*VEL2/(2*qp)
            
            if VPERP[aa] >= 0.0 and ENRG >= Emin and ENRG <= Emax:
                SUM += 2*np.pi*VPERP[aa]*DIST[aa, bb]*dvpara*dvperp
    return SUM


if __name__ == '__main__':
    # Load HOPE-L3-MOMENT CDF file with spacepy
    # Use moments to create a bi-maxwellian (for each time)
    # Use maths to integrate the moments above 30eV and up to 50keV
    # Use plots to illustrate what's going on
    #
    # To Do: Set temperature/anisotropy as constant. Does this cause a constant offset?
    # If so, how does the offset change with temperature? Is this a fiddle factor we could use?
    hope_dir   = 'E://DATA//RBSP//ECT//HOPE//L3//MOMENTS//'
    save_dir   = 'E://HOPE_MODEL_DENSITY//'
    if os.path.exists(save_dir) == False: os.makedirs(save_dir)
        
    time_start = np.datetime64('2015-01-16T04:05:00')
    time_end   = np.datetime64('2015-01-16T05:15:00')
    probe      = 'a'
    dstring    = time_start.astype(object).strftime('%Y%m%d')
    sp_clrs    = ['red', 'green', 'blue']
    
    # Max/Mins in v = np.sqrt(vp**2 + vx**2)
    eV     = False
    min_eV = 30.0
    max_eV = 50e3
    
    hope_file = None
    for file in os.listdir(hope_dir):
        if dstring in file:
            hope_file = hope_dir + file
            break
    
    hope_mom = sp.datamodel.fromCDF(hope_file)
    
    times  = hope_mom['Epoch_Ion']
    times  = np.asarray([np.datetime64(TIME) for TIME in times])
    st, en = boundary_idx64(times, time_start, time_end)
    
    ions     = ['p', 'he', 'o']
    mass     = np.array([1., 4., 16.])*mp
    N_ions   = len(ions) 
    dens_mod = np.zeros((N_ions, times.shape[0]), dtype=float)
    
    for ii, clr in zip(range(N_ions), sp_clrs):
        for jj in range(st, en):
            T_perp  = hope_mom['Tperp_{}_30'.format(ions[ii])][jj]*11600.
            T_para  = hope_mom['Tpar_{}_30'.format(ions[ii])][jj]*11600.
            dens_m3 = hope_mom['Dens_{}_30'.format(ions[ii])][jj]*1e6
            
            Nv        = 1001
            nvth      = 6
            
            vth_para  = np.sqrt(kB*T_para/mass[ii])
            vth_perp  = np.sqrt(kB*T_perp/mass[ii])
                    
            vperp     = np.linspace(-nvth*vth_perp, nvth*vth_perp, Nv)
            vpara     = np.linspace(-nvth*vth_para, nvth*vth_para, Nv)
            
            vperp_eV  = mass[ii]*vperp**2/(2*qp) * np.sign(vperp)
            vpara_eV  = mass[ii]*vpara**2/(2*qp) * np.sign(vpara)

            distro_2D = calc_distro_2D(dens_m3, vperp, vpara, vth_perp, vth_para)

            integral  = integrate_distro_2D(distro_2D, vperp, vpara, mass[ii], min_eV, max_eV)
            dens_mod[ii, jj] = integral*1e-6
            
            ## Mass-plot each time's Bi-Maxwellian fits with slices through the zeros.
            if False:
                distro_2D /= distro_2D.max()
                
                print('Density integration between {} eV - {} keV'.format(min_eV, max_eV*1e-3))
                print(integral, '/m3')
                print('Raw satellite density:')
                print(dens_m3, '/m3')
            
                if eV == True:
                    vperp = vperp_eV*1e-3
                    vpara = vpara_eV*1e-3
                    unit  = 'keV'
                    
                    int_min = min_eV
                    int_max = max_eV
                else:
                    vperp = vperp*1e-6
                    vpara = vpara*1e-6
                    
                    unit    = r'$\times 10^3$ km/s'
                    int_min = np.sqrt(2*min_eV*qp / mass[ii])*1e-6
                    int_max = np.sqrt(2*max_eV*qp / mass[ii])*1e-6
                
                
                plt.ioff()
                fnt = 14
                fig = plt.figure(figsize=(16, 10))
                gs  = fig.add_gridspec(4, 4)
                
                ax1 = fig.add_subplot(gs[:3, :3])
                axr = fig.add_subplot(gs[:3, 3])
                axb = fig.add_subplot(gs[3, :3])
                
                ax1.set_title('HOPE-{} {}-ION Normalized Bimaxwellian Distribution :: {}'.format(
                                                                     probe.upper(), ions[ii].upper(), times[jj]))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ax1.pcolormesh(vperp, vpara, distro_2D.T)
                
                ax1.set_xlim(vperp[0], vperp[-1])
                ax1.set_ylim(vpara[0], vpara[-1])
                ax1.axes.xaxis.set_visible(False)
                ax1.axes.yaxis.set_visible(False)
                
                # Slice through zeros
                midx = Nv//2; midy = Nv//2
                axb.plot(vperp, distro_2D[midx, :], c=clr)
                axr.plot(distro_2D[:, midy], vpara, c=clr)
                axr.yaxis.tick_right()
                axb.set_xlim(vperp[0], vperp[-1])
                axr.set_ylim(vpara[0], vpara[-1])
                
                axb.set_xlabel('$v_\perp$\n{}'.format(unit), fontsize=fnt)
                axr.set_ylabel('$v_\parallel$\n{}'.format(unit), rotation=0, fontsize=fnt, labelpad=20)
                axr.yaxis.set_label_position("right")
                
                axb.axvline( int_min, c='k', alpha=0.25)
                axb.axvline( int_max, c='k', alpha=0.25)
                axb.axvline(-int_min, c='k', alpha=0.25)
                axb.axvline(-int_max, c='k', alpha=0.25)
                
                axr.axhline( int_min, c='k', alpha=0.25)
                axr.axhline( int_max, c='k', alpha=0.25)
                axr.axhline(-int_min, c='k', alpha=0.25)
                axr.axhline(-int_max, c='k', alpha=0.25)
                
                circ  = Circle((0, 0), int_min, facecolor='None', edgecolor='k', lw=1, zorder=10)
                circ2 = Circle((0, 0), int_max, facecolor='None', edgecolor='k', lw=1, zorder=10)
                ax1.add_patch(circ)
                ax1.add_patch(circ2)
                
                fig.subplots_adjust(wspace=0, hspace=0)
                fig.text(0.73, 0.25, 'vth mult.   :: {}'.format(nvth), fontsize=12, family='monospace')
                fig.text(0.73, 0.22, 'HOPE Dens.  :: {:.2f} /cc'.format(dens_m3*1e-6), fontsize=12, family='monospace')
                fig.text(0.73, 0.19, 'Model Dens. :: {:.2f} /cc'.format(integral*1e-6), fontsize=12, family='monospace')
                
                fig.savefig(save_dir + 'rbsp{}_HOPEMODEL_{}_{}_{:04}'.format(probe, ions[ii], dstring, jj),
                                                                    bbox_inches='tight')
                plt.close('all')
                
    # Do fit for each time, plot them here (one for each species)
    if True:
        # Do Density, Temperature, Anisotropy in different plots
        fig, axes = plt.subplots(3, sharex=True)
        for ii, clr in zip(range(N_ions), sp_clrs):
            axes[ii].plot(times[st:en], hope_mom['Dens_{}_30'.format(ions[ii])][st:en], c=clr, label='Data')
            axes[ii].plot(times[st:en], dens_mod[ii, st:en], c=clr, ls='--', label='Model')
            axes[ii].legend()
      
        for ax in axes:
            axes[0].set_ylabel('$n_i$\n(/cc)', rotation=0, labelpad=30)
        
        axes[0].set_title('HOPE-{} :: REAL AND MODELLED MOMENTS :: {}'.format(probe.upper(), dstring))
        axes[-1].set_xlim(time_start, time_end)
        plt.show()
