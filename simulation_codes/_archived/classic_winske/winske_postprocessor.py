# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:13:37 2018

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# def generate_fourier_analyses(arr):
#     arr_kt     = np.zeros(arr.shape, dtype=complex)
#     arr_wk     = np.zeros(arr.shape, dtype=complex)
#     
#     t = np.arange(ntimes) * dt
#     k = np.arange(nx)
#     
#     # For each time (spatial FFT)
#     for ii in range(arr.shape[0]):
#         arr_kt[ii, :] = np.fft.fft(arr[ii, :] - arr[ii, :].mean())
#         
#     # For each gridpoint (temporal FFT)
#     for jj in range(arr.shape[1]):
#         arr_wk[:, jj] = np.fft.fft(arr_kt[:, jj] - arr_kt[:, jj].mean())
#         
#     # Conjugates
#     power_k = (arr_kt[:, :arr.shape[1]/2] * np.conj(arr_kt[:, :arr.shape[1]/2])).real
#     power   = (arr_wk[:arr.shape[0]/2, :arr.shape[1]/2] * np.conj(arr_wk[:arr.shape[0]/2, :arr.shape[1]/2])).real
#     
#     # Reals only
#     # power_k = arr_kt[:, :arr.shape[1]/2].real
#     # power   = arr_wk[:arr.shape[0]/2, :arr.shape[1]/2].real
#     
#     plt.ioff()
#     fig = plt.figure(1, figsize=(12, 8))
#     ax  = fig.add_subplot(111)
#     
#     ax.pcolormesh(np.log10(power[1:, 1:]), cmap='jet')
#     ax.set_ylim(0, 100)
#     ax.set_title(r'w-k Plot: $\omega/k$ (Winske code)')
#     ax.set_ylabel(r'$\omega$', rotation=0)
#     ax.set_xlabel('k (m-number?)')
#     
#     fullpath = plot_path + 'wk' + '.png'
#     fig.savefig(fullpath, edgecolor='none', bbox_inches='tight')
#     plt.close()
#     print('w-k Plot saved')
#     
#     fig2 = plt.figure(2, figsize=(12, 8))
#     ax2  = fig2.add_subplot(111)
#     
#     ax2.pcolormesh(k[:arr.shape[1]/2], t, power_k, cmap='jet')
#     ax2.set_xlim(None, 32)
#     ax2.set_title(r'k-t Plot: $\omega/k$ (Winske code)')
#     ax2.set_ylabel(r'$\Omega_i t$', rotation=0)
#     ax2.set_xlabel('k (m-number?)')
#     
#     fullpath = plot_path + 'kt' + '.png'
#     fig2.savefig(fullpath, edgecolor='none', bbox_inches='tight')
#     plt.close()
#     print('k-t Plot saved')
#     return
# =============================================================================


def waterfall_plot(field):
    plt.ioff()
    skip  = 5
    amp   = 100.                 # Amplitude multiplier of waves: 
    
    cells  = np.arange(nx)
    plt.figure()
    for ii in np.arange(ntimes):
        if ii%skip == 0:
            plt.plot(cells, amp*(field[ii] / field.max()) + ii, c='k', alpha=0.05)
        
    plt.xlim(0, nx)
    plt.show()
    return


def get_helical_components(BYS, BZS):    
    Bt_pos = np.zeros(BYS.shape, dtype=np.complex128)
    Bt_neg = np.zeros(BYS.shape, dtype=np.complex128)
    
    for ii in range(BYS.shape[0]):
        print('Analysing time step {}'.format(ii))
        Bt_pos[ii, :], Bt_neg[ii, :] = calculate_helicity(BYS[ii], BZS[ii], nx, 1.0)

    return Bt_pos, Bt_neg


def calculate_helicity(By, Bz, NX, dx):
    '''
    For a single snapshot in time, calculate the positive and negative helicity
    components from the y, z components of a field.
    
    This code has been checked by comparing the transverse field magnitude of the inputs and outputs,
    as this should be conserved (and it is).
    '''
    x       = np.arange(0, NX*dx, dx)
    
    k_modes = np.fft.rfftfreq(x.shape[0], d=dx)
    By_fft  = (1 / k_modes.shape[0]) * np.fft.rfft(By)
    Bz_fft  = (1 / k_modes.shape[0]) * np.fft.rfft(Bz)
    
    # Four fourier coefficients from FFT (since real inputs give symmetric outputs)
    # If there are any sign issues, it'll be with the sin components, here
    # Actually, making the sines negative just flipped the helicities
    By_cos = By_fft.real
    By_sin = By_fft.imag
    Bz_cos = Bz_fft.real
    Bz_sin = Bz_fft.imag
    
    # Construct spiral mode k-coefficients
    Bk_pos = 0.5 * ( (By_cos + Bz_sin) + 1j * (Bz_cos - By_sin ) )
    Bk_neg = 0.5 * ( (By_cos - Bz_sin) + 1j * (Bz_cos + By_sin ) )
    
    # Construct spiral mode timeseries
    Bt_pos = np.zeros(x.shape[0], dtype=np.complex128)
    Bt_neg = np.zeros(x.shape[0], dtype=np.complex128)
    
    # The sign of the exponential may also be another issue, should check.
    # Actually, it'll only flip around the helicities (change direction of propagation)
    for ii in range(k_modes.shape[0]):
        Bt_pos += Bk_pos[ii] * np.exp(-2j*np.pi*k_modes[ii]*x)
        Bt_neg += Bk_neg[ii] * np.exp( 2j*np.pi*k_modes[ii]*x)
    return Bt_pos, Bt_neg


# =============================================================================
# def plot_energies():
#     mp = 1.0
#     mag_energy  = np.zeros(ntimes)
#     part_energy = np.zeros((ntimes, nsp+1))
#     
#     for ii in range(ntimes):
#         mag_energy[ii] = np.square(bx[ii, :]).sum() + np.square(by[ii, :]).sum() + np.square(bz[ii, :]).sum()
#         
#         vpc2 = vc[ii, 0, :] ** 2 + vc[ii, 1, :] ** 2 + vc[ii, 2, :] ** 2
#         part_energy[ii, 1] = 0.5 * mp * vpc2.sum()                 # Particle total kinetic energy 
#     
#         vpb2 = vb[ii, 0, :] ** 2 + vb[ii, 1, :] ** 2 + vb[ii, 2, :] ** 2
#         part_energy[ii, 2] = 0.5 * mp * vpb2.sum()                 # Particle total kinetic energy 
# 
#     plt.ioff()
#     fig  = plt.figure()
#     
#     plt.plot(times, mag_energy / mag_energy.max(), label = r'$U_B$')
#     
#     for jj in range(1, nsp+1):
#         plt.plot(times, part_energy[:, jj] / part_energy[:, jj].max(), label='$K_E$ {}'.format(jj))
#     
#     plt.legend()
#     plt.title('Energy Distribution in Simulation')
#     plt.xlabel('Time ($\Omega t$)')
#     plt.xlim(0, 100)
#     plt.ylabel('Normalized Energy', rotation=90)
#     fullpath = plot_path + 'energy_plot'
#     plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
#     plt.close('all')
#     
#     print('Energy plot saved')
#     return
# =============================================================================


if __name__ == '__main__':
    run       = 2
    data_path = 'F://runs//helicity_tests_winske_port//run_{}//fields//'.format(run)
    by        = np.load(data_path + 'BYS' + '.npy')
    bz        = np.load(data_path + 'BZS' + '.npy')
    
# =============================================================================
#     xb         = np.load(data_path + 'XB'  + '.npy')
#     xc         = np.load(data_path + 'XC'  + '.npy')
#     vb         = np.load(data_path + 'VB'  + '.npy')
#     vc         = np.load(data_path + 'VC'  + '.npy')
# =============================================================================
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
    
    #plot_energies()
    #generate_fourier_analyses(byc)
    #waterfall_plot(by)
    
    BTP, BTN = get_helical_components(by, bz)
    #%%
    By_pos = BTP.real
    By_neg = BTN.real
    Bz_pos = BTP.imag
    Bz_neg = BTN.imag
    
    sig_fig = 3
    
    skip   = 30
    amp    = 250.                 # Amplitude multiplier of waves:
    sep    = 1.
    dark   = 0.9
    cells  = np.arange(nx)
    title  = 'Winske Implicit, Beam Anisotropy Instability (Fig. 5.21)'
    
    # Raw field
    fig0 = plt.figure(figsize=(18, 10))
    axy  = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    axz  = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    
    for ii in np.arange(By_pos.shape[0]):
        if ii%skip == 0:
            axy.plot(cells, amp*(by[ii] / by.max()) + sep*ii, c='k', alpha=dark)
            axz.plot(cells, amp*(bz[ii] / bz.max()) + sep*ii, c='k', alpha=dark)

    axy.set_title('By Raw')
    axz.set_title('Bz Raw')
    
    for ax in [axy, axz]:
        ax.set_xlim(0, cells.shape[0])
        ax.set_ylim(0, None)
        ax.set_xlabel('Cell Number')
    plt.suptitle(title)
    
    # By
    fig1 = plt.figure(figsize=(18, 10))
    ax1  = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax2  = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    
    for ii in np.arange(By_pos.shape[0]):
        if ii%skip == 0:
            ax1.plot(cells, amp*(By_pos[ii] / By_pos.max()) + sep*ii, c='k', alpha=dark)
            ax2.plot(cells, amp*(By_neg[ii] / By_neg.max()) + sep*ii, c='k', alpha=dark)

    ax1.set_title('By: +ve Helicity')
    ax2.set_title('By: -ve Helicity')
    
    for ax in [ax1, ax2]:
        ax.set_xlim(0, cells.shape[0])
        ax.set_ylim(0, None)
        ax.set_xlabel('Cell Number')
    plt.suptitle(title)
    
    # Bz
    fig2 = plt.figure(figsize=(18, 10))
    ax3  = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax4  = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    
    for ii in np.arange(Bz_pos.shape[0]):
        if ii%skip == 0:
            ax3.plot(cells, amp*(Bz_pos[ii] / Bz_pos.max()) + sep*ii, c='k', alpha=dark)
            ax4.plot(cells, amp*(Bz_neg[ii] / Bz_neg.max()) + sep*ii, c='k', alpha=dark)

    ax3.set_title('Bz: +ve Helicity')
    ax4.set_title('Bz: -ve Helicity')
    
    for ax in [ax3, ax4]:
        ax.set_xlim(0, cells.shape[0])
        ax.set_ylim(0, None)
        ax.set_xlabel('Cell Number')
        
    plt.suptitle(title)
    plt.show()