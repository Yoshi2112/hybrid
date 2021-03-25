# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:39:35 2021

@author: Yoshi
"""
import os, warnings
import numpy as np
import numba as nb
from mpl_toolkits import mplot3d
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
def calc_distro_1D_absV(DENS, VEL, VTH):
    DIST  = np.zeros(VEL.shape[0], dtype=np.float64)
    
    for jj in nb.prange(VEL.shape[0]):
        DIST[jj] = DENS * ((2*np.pi*VTH**2) ** (-3/2)) * np.exp(-VEL[jj]**2/(2*VTH**2))
    return DIST


@nb.njit()
def calc_distro_1D(DENS, VEL, VTH):
    DIST  = np.zeros(VEL.shape[0], dtype=np.float64)
    
    for jj in nb.prange(VEL.shape[0]):
        DIST[jj] = DENS * ((2*np.pi*VTH**2) ** (-1/2)) * np.exp(-VEL[jj]**2/(2*VTH**2))
    return DIST


@nb.njit()
def calc_distro_2D(DENS, VPER, VPAR, VTH_PER, VTH_PAR):
    DIST  = np.zeros((VPER.shape[0], VPAR.shape[0]), dtype=np.float64)
    
    out = DENS / np.sqrt(8 * np.pi**3 * VTH_PER**4 * VTH_PAR**2)
    for aa in nb.prange(VPER.shape[0]):
        for bb in nb.prange(VPAR.shape[0]):
            exp = np.exp(-0.5*VPER[aa]**2/VTH_PER**2
                         -0.5*VPAR[bb]**2/VTH_PAR**2)
            DIST[aa, bb] = out*exp
    return DIST


@nb.njit()
def calc_distro_3D(DENS, VX, VY, VZ, VTHX, VTHY, VTHZ):
    DIST  = np.zeros((VX.shape[0], VY.shape[0], VZ.shape[0]), dtype=np.float64)
    
    out = DENS / np.sqrt(8 * np.pi**3 * VTHX**2 * VTHY**2 * VTHZ**2)
    for aa in nb.prange(VX.shape[0]):
        for bb in nb.prange(VY.shape[0]):
            for cc in nb.prange(VZ.shape[0]):
                exp = np.exp(-0.5*VX[aa]**2/VTHX**2 
                              -0.5*VY[bb]**2/VTHY**2
                              -0.5*VZ[cc]**2/VTHZ**2)
                DIST[aa, bb, cc] = out * exp
    return DIST


@nb.njit()
def integrate_distro_1D_absV(DIST, VEL, Emin, Emax):
    dv = np.abs(VEL[1] - VEL[0])
    
    SUM   = 0.0
    for mm in nb.prange(DIST.shape[0]):
        if VEL[mm] >= 0:
            SUM += 4*np.pi*(VEL[mm]**2)*DIST[mm]*dv
                
    print('Density integration above eV ', Emin)
    print(SUM, '/m3')
    return


@nb.njit()
def integrate_distro_1D(DIST, VEL, Emin, Emax):
    dv = np.abs(VEL[1] - VEL[0])
    
    SUM   = 0.0
    for mm in nb.prange(DIST.shape[0]):
        SUM += DIST[mm]*dv
                
    print('Density integration above eV ', Emin)
    print(SUM, '/m3')
    return


@nb.njit()
def integrate_distro_2D(DIST, VPERP, VPARA, Emin, Emax):
    dvperp = VPERP[1] - VPERP[0]
    dvpara = VPARA[1] - VPARA[0]
    
    SUM = 0.0
    for aa in nb.prange(DIST.shape[0]):
        for bb in nb.prange(DIST.shape[1]):
            VEL2 = VPERP[aa]**2 + VPARA[bb]**2
            ENRG = mp*VEL2/(2*qp)
            
            if VPERP[aa] >= 0.0 and ENRG >= Emin:
                SUM += 2*np.pi*VPERP[aa]*DIST[aa, bb]*dvpara*dvperp
                
    print('Density integration above eV ', Emin)
    print(SUM, '/m3')
    return


@nb.njit()
def integrate_distro_3D(DIST, VX, VY, VZ, Emin, Emax):
    dvx = VX[1] - VX[0]
    dvy = VY[1] - VY[0]
    dvz = VZ[1] - VZ[0]
    
    SUM = 0.0
    for aa in nb.prange(DIST.shape[0]):
        for bb in nb.prange(DIST.shape[1]):
            for cc in nb.prange(DIST.shape[1]):
                SUM += DIST[aa, bb, cc]*dvx*dvy*dvz
                
    print('Density integration above eV ', Emin)
    print(SUM, '/m3')
    return


if __name__ == '__main__':
    # Load HOPE-L3-MOMENT CDF file with spacepy
    # Use moments to create a bi-maxwellian (for each time)
    # Use maths to integrate the moments above 30eV and up to 50keV
    # Use plots to illustrate what's going on
    hope_dir   = 'E://DATA//RBSP//ECT//HOPE//L3//MOMENTS//'
    
    time_start = np.datetime64('2015-01-16T04:05:00')
    time_end   = np.datetime64('2015-01-16T05:15:00')
    probe      = 'a'
    dstring    = time_start.astype(object).strftime('%Y%m%d')
    
    # Max/Mins in v = np.sqrt(vp**2 + vx**2)
    eV     = True
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
    
    if False:
        fig, axes = plt.subplots(3, sharex=True)
        for ion, clr in zip(['p', 'he', 'o'], ['red', 'green', 'blue']):
            axes[0].plot(times[st:en], hope_mom['Dens_{}_30'.format(ion)][st:en], c=clr)
            axes[1].plot(times[st:en], hope_mom['Tperp_{}_30'.format(ion)][st:en], c=clr)
            axes[2].plot(times[st:en], hope_mom['Tperp_Tpar_{}_30'.format(ion)][st:en] - 1, c=clr)
        
        axes[0].set_ylabel('Density\n(/cc)', rotation=0, labelpad=30)
        axes[1].set_ylabel('T_perp\n(eV)', rotation=0, labelpad=30)
        axes[2].set_ylabel('A', rotation=0, labelpad=30)
        
        axes[0].set_title('HOPE-{} MOMENTS {}'.format(probe.upper(), dstring))
        axes[-1].set_xlim(time_start, time_end)
        
    
    ion = 'p'
    for ii in [st]:
        T_perp  = hope_mom['Tperp_{}_30'.format(ion)][ii]*11600.
        T_para  = hope_mom['Tpar_{}_30'.format(ion)][ii]*11600.
        dens_m3 = hope_mom['Dens_{}_30'.format(ion)][ii]*1e6
        
        Nv        = 500
        nvth      = 3
        
        vth_para  = np.sqrt(kB*T_para/mp)
        vth_perp  = np.sqrt(kB*T_perp/mp)
                
        vperp     = np.linspace(-nvth*vth_perp, nvth*vth_perp, Nv)
        vpara     = np.linspace(-nvth*vth_para, nvth*vth_para, Nv)
        
        vperp_eV  = mp*vperp**2/(2*qp) * np.sign(vperp)
        vpara_eV  = mp*vpara**2/(2*qp) * np.sign(vpara)
        
        case = 2 
        if case==0:
            distro_1Dv = calc_distro_1D_absV(dens_m3, vperp, vth_perp)
            integrate_distro_1D_absV(distro_1Dv, vperp, 0.0, 0.0)  
        elif case==1:
            distro_1D = calc_distro_1D(dens_m3, vperp, vth_perp)
            integrate_distro_1D(distro_1D, vperp, 0.0, 0.0) 
        elif case==2:
            distro_2D = calc_distro_2D(dens_m3, vperp, vpara, vth_perp, vth_para)
            integrate_distro_2D(distro_2D, vperp, vpara, min_eV, max_eV)
        else:
            distro_3D = calc_distro_3D(dens_m3, vpara, vperp, vperp,
                                       vth_para, vth_perp, vth_perp)
            integrate_distro_3D(distro_3D, vpara, vperp, vperp, 0.0, 0.0)
        
        print('Raw satellite density:')
        print(dens_m3, '/m3')
    
        if eV == True:
            vperp = vperp_eV
            vpara = vpara_eV
            unit  = 'eV'
            
            int_min = min_eV
            int_max = max_eV
        else:
            unit    = 'm/s'
            int_min = np.sqrt(2*min_eV*qp / mp)
            int_max = np.sqrt(2*max_eV*qp / mp)
        
        fnt = 16
        fig = plt.figure(figsize=(16, 10))
        gs  = fig.add_gridspec(4, 4)
        
        ax1 = fig.add_subplot(gs[:3, :3])
        axr = fig.add_subplot(gs[:3, 3])
        axb = fig.add_subplot(gs[3, :3])
        
        ax1.set_title('HOPE-{} $H^+$ Distribution :: Bimaxwellian :: {} :: Units: {}'.format(
                                                             probe.upper(), times[ii], unit))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax1.pcolormesh(vperp, vpara, distro_2D.T)
        
        ax1.set_xlim(vperp[0], vperp[-1])
        ax1.set_ylim(vpara[0], vpara[-1])
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        
        # Slice through zeros
        midx = Nv//2; midy = Nv//2
        axb.plot(vperp, distro_2D[midx, :])
        axr.plot(distro_2D[:, midy], vpara)
        axr.yaxis.tick_right()
        axb.set_xlim(vperp[0], vperp[-1])
        axr.set_ylim(vpara[0], vpara[-1])
        
        axb.set_xlabel('$v_\perp$', fontsize=fnt)
        axr.set_ylabel('$v_\parallel$', rotation=0, fontsize=fnt)
        axr.yaxis.set_label_position("right")
        
        fig.subplots_adjust(wspace=0, hspace=0)

        
        

        
        