# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:24:46 2019

@author: Yoshi
"""
import sys
data_scripts_dir = 'D://Google Drive//Uni//PhD 2017//Data//Scripts//'
sys.path.append(data_scripts_dir)
from analysis_scripts import analytic_signal

import numpy as np
import numba as nb
import os
import analysis_config as cf
import pdb
'''
Dump general processing scripts here that don't require global variables: i.e. 
they are completely self-contained or can be easily imported.

These will often be called by plotting scripts so that the main 'analysis'
script is shorter and less painful to work with

If a method requires more than a few functions, it will be split into its own 
module, i.e. get_growth_rates
'''
def get_B0_particle(x, v, B, sp):
    
    @nb.njit()
    def get_b1(pos, mag):
        xp_mag = np.zeros(3)
        epsil  = 1e-15

        particle_transform = cf.xmax + cf.ND*cf.dx  + epsil   # Offset to account for E/B grid and damping nodes
        
        xp          = (pos + particle_transform) / cf.dx      # Shift particle position >= 0
        Ib          = int(round(xp) - 1.0)                    # Get leftmost to nearest node
        delta_left  = Ib - xp                                 # Distance from left node in grid units
    
        W0 = 0.5  * np.square(1.5 - abs(delta_left))    # Get weighting factors
        W1 = 0.75 - np.square(delta_left + 1.)
        W2 = 1.0  - W0 - W1
        
        for kk in range(3):
            xp_mag[kk] = W0 * mag[Ib, kk] + W1 * mag[Ib + 1, kk] + W2 * mag[Ib + 2, kk]
        
        return xp_mag
    
    def eval_B0x(x):
        return cf.B_eq * (1 + cf.a * x**2)
    
    b1 = get_b1(x, B)
    
    B0_xp    = np.zeros(3)
    B0_xp[0] = eval_B0x(x)    
    b1t      = np.sqrt(b1[0] ** 2 + b1[1] ** 2 + b1[2] ** 2)
    l_cyc    = (cf.charge[sp] / cf.mass[sp]) * (B0_xp[0] + b1t)
    
    fac      = cf.a * cf.B_eq * x / l_cyc
    B0_xp[1] = v[2] * fac
    B0_xp[2] =-v[1] * fac
    return B0_xp


def get_energies(): 
    '''
    Computes and saves field and particle energies at each field/particle timestep.
    '''
    from analysis_config import NX, dx, idx_bounds, Nj, n_contr, mass,\
                                num_field_steps, num_particle_steps

    energy_file = cf.temp_dir + 'energies.npz'
    
    if os.path.exists(energy_file) == False:
        mu0 = (4e-7) * np.pi          # Magnetic Permeability of Free Space (SI units)
        kB  = 1.38065e-23             # Boltzmann's Constant (J/K)
        q   = 1.602e-19               # Elementary charge (C)
    
        mag_energy      = np.zeros( num_field_steps)
        electron_energy = np.zeros( num_field_steps)
        particle_energy = np.zeros((num_particle_steps, Nj, 2))
        
        for ii in range(num_field_steps):
            print('Loading field file {}'.format(ii))
            B, E, Ve, Te, J, q_dns = cf.load_fields(ii)
            
            mag_energy[ii]      = (0.5 / mu0) * np.square(B[1:-2]).sum() * NX * dx
            electron_energy[ii] = 1.5 * (kB * Te * q_dns / q).sum() * NX * dx
    
        for ii in range(num_particle_steps):
            print('Loading particle file {}'.format(ii))
            pos, vel = cf.load_particles(ii)
            for jj in range(Nj):
                '''
                Only works properly for theta = 0 : Fix later
                '''
                vpp2 = vel[0, idx_bounds[jj, 0]:idx_bounds[jj, 1]] ** 2
                vpx2 = vel[1, idx_bounds[jj, 0]:idx_bounds[jj, 1]] ** 2 + vel[2, idx_bounds[jj, 0]:idx_bounds[jj, 1]] ** 2
        
                particle_energy[ii, jj, 0] = 0.5 * mass[jj] * vpp2.sum() * n_contr[jj] * NX * dx
                particle_energy[ii, jj, 1] = 0.5 * mass[jj] * vpx2.sum() * n_contr[jj] * NX * dx
        
        total_energy = np.zeros(cf.num_field_steps)   # Placeholder until I interpolate this
        
        print('Saving energies to file...')
        np.savez(energy_file, mag_energy      = mag_energy,
                              electron_energy = electron_energy,
                              particle_energy = particle_energy,
                              total_energy    = total_energy)
    else:
        print('Loading energies from file...')
        energies        = np.load(energy_file) 
        mag_energy      = energies['mag_energy']
        electron_energy = energies['electron_energy']
        particle_energy = energies['particle_energy']
        total_energy    = energies['total_energy']
    return mag_energy, electron_energy, particle_energy, total_energy


def get_helical_components(overwrite):
    temp_dir = cf.temp_dir

    if os.path.exists(temp_dir + 'B_positive_helicity.npy') == False or overwrite == True:
        ftime, By = cf.get_array('By')
        ftime, Bz = cf.get_array('Bz')

        Bt_pos = np.zeros(By.shape, dtype=np.complex128)
        Bt_neg = np.zeros(By.shape, dtype=np.complex128)
        
        for ii in range(By.shape[0]):
            print('Analysing time step {}'.format(ii))
            Bt_pos[ii, :], Bt_neg[ii, :] = calculate_helicity(By[ii], Bz[ii], cf.NC + 1, cf.dx)
        
        print('Saving helicities to file...')
        np.save(temp_dir + 'B_positive_helicity.npy', Bt_pos)
        np.save(temp_dir + 'B_negative_helicity.npy', Bt_neg)
    else:
        print('Loading helicities from file...')
        ftime, By = cf.get_array('By')
        
        Bt_pos = np.load(temp_dir + 'B_positive_helicity.npy')
        Bt_neg = np.load(temp_dir + 'B_negative_helicity.npy')
    return ftime, Bt_pos, Bt_neg


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
    for ii in range(k_modes.shape[0]):
        Bt_pos += Bk_pos[ii] * np.exp(-2j*np.pi*k_modes[ii]*x)
        Bt_neg += Bk_neg[ii] * np.exp( 2j*np.pi*k_modes[ii]*x)
    return Bt_pos, Bt_neg



def basic_S(arr, k=5, h=1.0):
    N = arr.shape[0]
    S1 = np.zeros(N)
    S2 = np.zeros(N)
    S3 = np.zeros(N)
    
    for ii in range(N):
        if ii < k:
            left_vals = arr[:ii]
            right_vals = arr[ii + 1:ii + k + 1]
        elif ii >= N - k:
            left_vals  = arr[ii - k: ii]
            right_vals = arr[ii + 1:]
        else:
            left_vals  = arr[ii - k: ii]
            right_vals = arr[ii + 1:ii + k + 1]

        left_dist  = arr[ii] - left_vals
        right_dist = arr[ii] - right_vals
        
        if left_dist.shape[0] == 0:
            left_dist = np.append(left_dist, 0)
        elif right_dist.shape[0] == 0:
            right_dist = np.append(right_dist, 0)
        
        S1[ii] = 0.5 * (left_dist.max()     + right_dist.max()    )
        S2[ii] = 0.5 * (left_dist.sum() / k + right_dist.sum() / k)
        S3[ii] = 0.5 * ((arr[ii] - left_vals.sum() / k) + (arr[ii] - right_vals.sum() / k))
        
    S_ispeak = np.zeros((N, 3))
    
    for S, xx in zip([S1, S2, S3], np.arange(3)):
        for ii in range(N):
            if S[ii] > 0 and (S[ii] - S.mean()) > (h * S.std()):
                S_ispeak[ii, xx] = 1

    for xx in range(3):
        for ii in range(N):
            for jj in range(N):
                if ii != jj and S_ispeak[ii, xx] == 1 and S_ispeak[jj, xx] == 1:
                    if abs(jj - ii) <= k:
                        if arr[ii] < arr[jj]:
                            S_ispeak[ii, xx] = 0
                        else:
                            S_ispeak[jj, xx] = 0
                            
    S1_peaks = np.arange(N)[S_ispeak[:, 0] == 1]
    S2_peaks = np.arange(N)[S_ispeak[:, 1] == 1]
    S3_peaks = np.arange(N)[S_ispeak[:, 2] == 1]
    return S1_peaks, S2_peaks, S3_peaks


def get_derivative(arr):
    ''' Caculates centered derivative for values in 'arr', with forward and backward differences applied
    for boundary points'''
    
    deriv = np.zeros(arr.shape[0])
    
    deriv[0 ] = (-3*arr[ 0] + 4*arr[ 1] - arr[ 2]) / (2 * cf.dt_field)
    deriv[-1] = ( 3*arr[-1] - 4*arr[-2] + arr[-3]) / (2 * cf.dt_field)
    
    for ii in np.arange(1, arr.shape[0] - 1):
        deriv[ii] = (arr[ii + 1] - arr[ii - 1]) / (2 * cf.dt_field)
    return deriv


def get_envelope(arr):
    signal_envelope = analytic_signal(arr, dt=cf.dx)
    return signal_envelope


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    f0    = 0.4 # Hz
    t_max = 10000
    dt    = 0.1
    t     = np.arange(0, t_max, dt)
    
    signal = np.sin(2 * np.pi * f0 * t)
    sfft   = 2 / t.shape[0] * np.fft.rfft(signal)
    freqs  = np.fft.rfftfreq(t.shape[0], d=dt)
    
    plt.plot(freqs, sfft.real)
    plt.plot(freqs, sfft.imag)
    #plt.xlim(0, 0.5)
    
    
    