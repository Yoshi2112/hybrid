# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:39:35 2021

@author: Yoshi
"""
import os, warnings, pdb, sys
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
    spice_dir  = 'E://DATA//RBSP//RBSPICE//3PAP//TOFxEH//'
    save_dir   = 'E://SPICE_MODEL_DENSITY//'
    if os.path.exists(save_dir) == False: os.makedirs(save_dir)
        
    time_start = np.datetime64('2013-07-25T21:00:00')
    time_end   = np.datetime64('2013-07-25T22:00:00')
    probe      = 'a'
    dstring    = time_start.astype(object).strftime('%Y%m%d')
    sp_clrs    = ['red', 'green', 'blue']
    
    # Max/Mins in v = np.sqrt(vp**2 + vx**2)
    eV     = False
    min_eV = 50e3
    max_eV = 600e3
    
    spice_file = None
    for file in os.listdir(spice_dir):
        if dstring in file:
            spice_file = spice_dir + file
            break
    
    spice_cdf = sp.datamodel.fromCDF(spice_file)
    
    times  = spice_cdf['Epoch']
    times  = np.asarray([np.datetime64(TIME) for TIME in times])
    st, en = boundary_idx64(times, time_start, time_end)

    mass     = mp
    dens_mod = np.zeros((times.shape[0]), dtype=float)
    
    for jj in range(st, en):
        dens_m3 = spice_cdf['FPDU_Density'][jj]*1e6
        
        P_perp  = spice_cdf['FPDU_PerpPressure'][jj]*1e-9
        P_para  = spice_cdf['FPDU_ParaPressure'][jj]*1e-9
        
        T_perp  = P_perp / (dens_m3 * kB)
        T_para  = P_para / (dens_m3 * kB)
        
        Nv        = 1001
        nvth      = 10
        
        vth_para  = np.sqrt(kB*T_para/mass)
        vth_perp  = np.sqrt(kB*T_perp/mass)
                
        vperp     = np.linspace(-nvth*vth_perp, nvth*vth_perp, Nv)
        vpara     = np.linspace(-nvth*vth_para, nvth*vth_para, Nv)
        
        vperp_eV  = mass*vperp**2/(2*qp) * np.sign(vperp)
        vpara_eV  = mass*vpara**2/(2*qp) * np.sign(vpara)

        distro_2D = calc_distro_2D(dens_m3, vperp, vpara, vth_perp, vth_para)

        integral     = integrate_distro_2D(distro_2D, vperp, vpara, mass, min_eV, max_eV)
        dens_mod[jj] = integral*1e-6
        
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
                int_min = np.sqrt(2*min_eV*qp / mass)*1e-6
                int_max = np.sqrt(2*max_eV*qp / mass)*1e-6
            
            
            plt.ioff()
            fnt = 14
            fig = plt.figure(figsize=(16, 10))
            gs  = fig.add_gridspec(4, 4)
            
            ax1 = fig.add_subplot(gs[:3, :3])
            axr = fig.add_subplot(gs[:3, 3])
            axb = fig.add_subplot(gs[3, :3])
            
            ax1.set_title('RBSPICE-{} PROTON Normalized Bimaxwellian Distribution :: {}'.format(
                                                                 probe.upper(), times[jj]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax1.pcolormesh(vperp, vpara, distro_2D.T)
            
            ax1.set_xlim(vperp[0], vperp[-1])
            ax1.set_ylim(vpara[0], vpara[-1])
            ax1.axes.xaxis.set_visible(False)
            ax1.axes.yaxis.set_visible(False)
            
            # Slice through zeros
            midx = Nv//2; midy = Nv//2
            axb.plot(vperp, distro_2D[midx, :], c='r')
            axr.plot(distro_2D[:, midy], vpara, c='r')
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
            
            fig.savefig(save_dir + 'rbsp{}_RBSPICEMODEL_{}_{}_{:04}'.format(probe, 'H', dstring, jj),
                                                                bbox_inches='tight')
            plt.close('all')
            
    #%%
    # Do fit for each time, plot them here (one for each species)
    if False:
        # Do Density, Temperature, Anisotropy in different plots
        fig, ax = plt.subplots(sharex=True)

        ax.plot(times[st:en], spice_cdf['FPDU_Density'][st:en], c='r', label='Data')
        ax.plot(times[st:en], dens_mod[st:en], c='r', ls='--', label='model')
  
        ax.set_ylabel('$n_H$\n(/cc)', rotation=0, labelpad=30)
        ax.legend()
        
        ax.set_title('RBSPICE-{} :: REAL AND MODELLED MOMENTS :: {}'.format(probe.upper(), dstring))
        ax.set_xlim(time_start, time_end)
        plt.show()
        
        
    # Plot density variation against trace pressure
    if True:
        tPres = 2*spice_cdf['FPDU_PerpPressure'][st:en] + spice_cdf['FPDU_ParaPressure'][st:en]
        nDiff = (spice_cdf['FPDU_Density'][st:en] - dens_mod[st:en]) / spice_cdf['FPDU_Density'][st:en] *100.
        
        # Do Density, Temperature, Anisotropy in different plots
        fig, ax = plt.subplots(sharex=True)

        ax.plot(times[st:en], tPres, c='b')
        ax.set_ylabel('Trace Pressure\n(nPa)', rotation=90, labelpad=30, color='b')
        
        ax2 = ax.twinx()
        ax2.plot(times[st:en], nDiff, c='r')
        ax2.set_ylabel('Density discrepancy\n(%%)', rotation=90, labelpad=30, color='r')
          
        #ax.set_title('RBSPICE-{} :: REAL AND MODELLED MOMENTS :: {}'.format(probe.upper(), dstring))
        ax.set_xlim(time_start, time_end)
        plt.show()
