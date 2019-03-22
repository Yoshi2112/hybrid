# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:13:37 2018

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb

def generate_fourier_analyses(arr):
    arr_kt     = np.zeros(arr.shape, dtype=complex)
    arr_wk     = np.zeros(arr.shape, dtype=complex)
    
    t = np.arange(ntimes) * dt
    k = np.arange(nx)
    
    # For each time (spatial FFT)
    for ii in range(arr.shape[0]):
        arr_kt[ii, :] = np.fft.fft(arr[ii, :] - arr[ii, :].mean())
        
    # For each gridpoint (temporal FFT)
    for jj in range(arr.shape[1]):
        arr_wk[:, jj] = np.fft.fft(arr_kt[:, jj] - arr_kt[:, jj].mean())
        
    # Conjugates
    power_k = (arr_kt[:, :arr.shape[1]/2] * np.conj(arr_kt[:, :arr.shape[1]/2])).real
    power   = (arr_wk[:arr.shape[0]/2, :arr.shape[1]/2] * np.conj(arr_wk[:arr.shape[0]/2, :arr.shape[1]/2])).real
    
    # Reals only
    # power_k = arr_kt[:, :arr.shape[1]/2].real
    # power   = arr_wk[:arr.shape[0]/2, :arr.shape[1]/2].real
    
    plt.ioff()
    fig = plt.figure(1, figsize=(12, 8))
    ax  = fig.add_subplot(111)
    
    ax.pcolormesh(np.log10(power[1:, 1:]), cmap='jet')
    ax.set_ylim(0, 100)
    ax.set_title(r'w-k Plot: $\omega/k$ (Winske code)')
    ax.set_ylabel(r'$\omega$', rotation=0)
    ax.set_xlabel('k (m-number?)')
    
    fullpath = plot_path + 'wk' + '.png'
    fig.savefig(fullpath, edgecolor='none', bbox_inches='tight')
    plt.close()
    print('w-k Plot saved')
    
    fig2 = plt.figure(2, figsize=(12, 8))
    ax2  = fig2.add_subplot(111)
    
    ax2.pcolormesh(k[:arr.shape[1]/2], t, power_k, cmap='jet')
    ax2.set_xlim(None, 32)
    ax2.set_title(r'k-t Plot: $\omega/k$ (Winske code)')
    ax2.set_ylabel(r'$\Omega_i t$', rotation=0)
    ax2.set_xlabel('k (m-number?)')
    
    fullpath = plot_path + 'kt' + '.png'
    fig2.savefig(fullpath, edgecolor='none', bbox_inches='tight')
    plt.close()
    print('k-t Plot saved')
    return


def waterfall_plot(field):
    plt.ioff()
    
    amp   = 100.                 # Amplitude multiplier of waves: 
    
    cells  = np.arange(nx)
    
    for (ii, t) in zip(np.arange(ntimes), np.arange(0, ntimes*dt, dt)):
        #if round(t, 2)%0.5 == 0:
        plt.plot(cells, amp*(arr[ii] / arr.max()) + ii, c='k', alpha=0.05)
        
    plt.xlim(0, nx)
    plt.show()
    return


def plot_energies():
    mp = 1.0
    mag_energy  = np.zeros(ntimes)
    part_energy = np.zeros((ntimes, nsp+1))
    
    for ii in range(ntimes):
        mag_energy[ii] = np.square(bx[ii, :]).sum() + np.square(by[ii, :]).sum() + np.square(bz[ii, :]).sum()
        
        vpc2 = vc[ii, 0, :] ** 2 + vc[ii, 1, :] ** 2 + vc[ii, 2, :] ** 2
        part_energy[ii, 1] = 0.5 * mp * vpc2.sum()                 # Particle total kinetic energy 
    
        vpb2 = vb[ii, 0, :] ** 2 + vb[ii, 1, :] ** 2 + vb[ii, 2, :] ** 2
        part_energy[ii, 2] = 0.5 * mp * vpb2.sum()                 # Particle total kinetic energy 

    plt.ioff()
    fig  = plt.figure()
    
    plt.plot(times, mag_energy / mag_energy.max(), label = r'$U_B$')
    
    for jj in range(1, nsp+1):
        plt.plot(times, part_energy[:, jj] / part_energy[:, jj].max(), label='$K_E$ {}'.format(jj))
    
    plt.legend()
    plt.title('Energy Distribution in Simulation')
    plt.xlabel('Time ($\Omega t$)')
    plt.xlim(0, 100)
    plt.ylabel('Normalized Energy', rotation=90)
    fullpath = plot_path + 'energy_plot'
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close('all')
    
    print('Energy plot saved')
    return


if __name__ == '__main__':
    data_path = 'E://runs//winske_anisotropy_test//vanilla_winske//save_data//'
    plot_path = 'E://runs//winske_anisotropy_test//vanilla_winske//plots//'
    by        = np.load(data_path + 'BYS' + '.npy')
    bz        = np.load(data_path + 'BZS' + '.npy')
        
    xb         = np.load(data_path + 'XB'  + '.npy')
    xc         = np.load(data_path + 'XC'  + '.npy')
    
    vb         = np.load(data_path + 'VB'  + '.npy')
    vc         = np.load(data_path + 'VC'  + '.npy')
    
    dtwci  = 0.05                       # Timestep in inverse gyrofrequency
    xmax   = 128.                       # System length in c/wpi
    wpiwci = 10000.                     # Ratio of plasma and gyro frequencies
    nsp    = 2                          # Number of ion species
    
    nspec  = np.array([5120,5120])      # Number of macroparticles per species
    wspec  = np.array([1.,1.])          # Mass of each species (in multiples of proton mass)
    dnspec = np.array([0.10,0.900])     # Density of each species (total = n0)

    vbspec = np.array([0.90,-0.10])     # Bulk velocity for each species
    btspec = np.array([10.,1.])         # Species plasma beta (parallel?)
    anspec = np.array([5.,1.])          # T_perp/T_parallel (anisotropy) for each species
    
    bete  = 1.                          # Electron beta
    theta = 0.                          # Angle between B0 and x axis

    # Define some variables from inputs
    dt   = wpiwci*dtwci                 # Time step
    thet = theta*1.74533e-2             # Theta in radians
    cth  = np.cos(thet)                 # Cos theta (for magnetic field rotation)
    sth  = np.sin(thet)                 # Sin theta (for magnetic field rotation)
    bxc  = cth/wpiwci                   # Background magnetic field: x component
    byc  = 0.                           # Background magnetic field: y component
    bzc  = sth/wpiwci                   # Background magnetic field: z component
    vye  = 0.                           # Background electric field: y component
    vze  = 0.                           # Background electric field: z component
    te0  = bete/(2.*wpiwci**2)          # Initial electron temperature
    pe0  = te0                          # Initial electron pressure
    
    ntimes = by.shape[0]
    nx     = by.shape[1]
    bx     = np.ones(by.shape) * bxc
    times  = np.arange(ntimes) * dtwci
    
    plot_energies()
    
    #generate_fourier_analyses(byc)
    #waterfall_plot(arr)