# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:02:41 2020

@author: Yoshi
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# To do: Maybe see if you can do this in Fourier space? What's the conversion 
#         factor? Would allow direct comparison against linear theory.
#
# Use k to get spatially varying field as well. Examine growth rates:
#       -- In fourier space (FFT and look at growth of k-modes)
def calculate_helicity(Fy, Fz):
    '''
    For a single snapshot in time, calculate the positive and negative helicity
    components from the y, z components of a field.
    '''
    print('Calculating Fourier modes...')
    n_time  = Fy.shape[0]
    n_cells = Fy.shape[1]
    
    x_idx   = np.linspace(0, n_cells*dx, n_cells)
    k_modes = np.fft.rfftfreq(n_cells, d=dx)
    
    Ft_pos = np.zeros((n_time, n_cells), dtype=np.complex128)
    Ft_neg = np.zeros((n_time, n_cells), dtype=np.complex128)
        
    Fk_pos = np.zeros((n_time, k_modes.shape[0]), dtype=np.complex128)
    Fk_neg = np.zeros((n_time, k_modes.shape[0]), dtype=np.complex128)
    
    for kk in range(n_time):
        Fy_fft  = (1 / k_modes.shape[0]) * np.fft.rfft(Fy[kk])
        Fz_fft  = (1 / k_modes.shape[0]) * np.fft.rfft(Fz[kk])
        
        # Four fourier coefficients from FFT (since real inputs give symmetric outputs)
        # If there are any sign issues, it'll be with the sin components, here
        Fy_cos = Fy_fft.real
        Fy_sin = Fy_fft.imag
        Fz_cos = Fz_fft.real
        Fz_sin = Fz_fft.imag
        
        # Construct spiral mode k-coefficients
        Fk_pos[kk] = 0.5 * ( (Fy_cos + Fz_sin) + 1j * (Fz_cos - Fy_sin ) )
        Fk_neg[kk] = 0.5 * ( (Fy_cos - Fz_sin) + 1j * (Fz_cos + Fy_sin ) )

        # The sign of the exponential may also be another issue, should check.
        for ii in range(k_modes.shape[0]):
            Ft_pos[kk] += Fk_pos[kk, ii] * np.exp(-2j*np.pi*k_modes[ii]*x_idx)
            Ft_neg[kk] += Fk_neg[kk, ii] * np.exp( 2j*np.pi*k_modes[ii]*x_idx)
    return k_modes, Fk_pos, Fk_neg


B0  = 200e-9
ne  = 200e6
mp  = 1.673e-27
me  = 9.110e-31
qp  = 1.602e-19
qe  = -qp
c   = 3e8
e0  = 8.854e-12
mu0 = 4e-7*np.pi

wpe2 = ne * qe ** 2 / (me * e0)
wpi2 = ne * qp ** 2 / (mp * e0)

wpi  = np.sqrt(wpi2)
wpe  = np.sqrt(wpe2) 
pcyc = qp * B0 / mp
ecyc = qe * B0 / me
pinv = 1. / pcyc

t_max = 200. * pinv
dt    = 0.05 * pinv
t     = np.arange(0, t_max, dt)
tinv  = t*pcyc

NX    = 512
dx    = c / wpi
x_max = NX*dx
x     = np.arange(0, x_max, dx)

sat   = 0.043  * B0

Bw1   = 4.0e-4 * B0
w1    = 0.3    * pcyc   
gr1   = 0.024  * pcyc
k1    = +w1/c  * np.sqrt(1 - wpi2/(w1**2 - w1*pcyc) - wpe2/(w1**2 - w1*ecyc))

Bw2   = 4.0e-4 * B0
w2    = 0.1    * pcyc   
gr2   = 0.03   * pcyc
k2    = -w2/c  * np.sqrt(1 - wpi2/(w2**2 - w2*pcyc) - wpe2/(w2**2 - w2*ecyc))

# Forward/Backward propagating waves :: Maybe simulate saturation by capping Bw*np.exp(gr*t) to sat
# Growth rate extraction with two counter-propagating waves important
wave_f = np.zeros((t.shape[0], x.shape[0]), dtype=np.complex128)
wave_b = np.zeros((t.shape[0], x.shape[0]), dtype=np.complex128)
for ii in range(t.shape[0]):
    for jj in range(x.shape[0]):
        wave_f[ii, jj] = Bw1 * np.exp(-1j * (w1*t[ii] - k1*x[jj])) * np.exp(gr1*t[ii])
        wave_b[ii, jj] = Bw2 * np.exp(-1j * (w2*t[ii] - k2*x[jj])) * np.exp(gr2*t[ii])

# Define wave arrays (a few different types for fun)
wave  = wave_f + wave_b
By    = wave.real
Bz    = wave.imag
Bt    = abs(wave)
B_sq  = (By ** 2 + Bz ** 2)
B_abs = (wave * np.conj(wave)).real
ln_B2 = np.log(B_sq)


# Plot wave amplitude squared vs. time for each "cell"
# Confirmed - no spatial variation in B_abs == B_sq
if False:
    fig, ax = plt.subplots()
    for ii in range(x.shape[0]):
        ax.semilogy(t, B_abs[:, ii])

# Plot wave and wave energy in time
# Validated :: Accurate!
if False:
    # Calculate total energy in 'simulation'
    U_B   = np.zeros(t.shape[0], dtype=np.float64)
    for ii in range(t.shape[0]):
        U_B[ii] = 0.5 / mu0 * B_sq[ii].sum() * dx
    
    gradient    = (np.log(U_B[-1]) - np.log(U_B[0])) / t[-1]
    growth_rate = 0.5 * gradient
    growth_rate_normalized = growth_rate / pcyc
    
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].pcolormesh(t, x/dx, By.T*1e9, cmap='jet')
    ax[0].set_xlim(0, t_max)
    ax[0].set_ylim(0, x_max/dx)
    ax[0].set_ylabel('x (dx)')
    ax[0].set_title('Growth rate :: %.4f$\Omega_H$' % growth_rate_normalized)
    
    ax[1].semilogy(t, U_B)
    ax[1].set_xlim(0, t_max)
    ax[1].set_ylabel('$U_B$')
    ax[1].set_xlabel('t (s)')
    
# Do the same thing now, but for each k
# How does the slope of a particular B(k) relate to the growth rate?
# Should be able to test by having several separate waves superposed (a situation for
# which the energy wouldn't work)
# Didn't really work, growth rate leaked into every bin
if False:
    # Plot wave    with time
    # Plot k-space with time
    # Plot dominant mode/s with time
    #
    # Able to distinguish sign of k (-ve k?, forward propagating gives P-)
    # Direction of propagation can change (via -ve w) but helicity remains the same
    # Thus only k dependent? Still needs a little work to understand.
    k, Bk_pos, Bk_neg = calculate_helicity(By*1e9, Bz*1e9)
    By_pos            = (Bk_pos.real**2).T
    By_neg            = (Bk_neg.real**2).T
    
    Bf = np.log((Bk_pos*np.conj(Bk_pos)).real).T
    Bb = np.log((Bk_neg*np.conj(Bk_neg)).real).T
    
    mode1 = Bf[104]
    mode2 = Bb[29]
    
    grad1 = (mode1[-1] - mode1[0]) / t[-1]
    gamma1   = 0.5 * grad1 / pcyc
    
    grad2 = (mode2[-1] - mode2[0]) / t[-1]
    gamma2   = 0.5 * grad2 / pcyc
    
    # Plot main modes + growth rate extraction
    fig, ax = plt.subplots()
    ax.plot(t, mode1, label='gr1 = {}'.format(gamma1))
    ax.plot(t, mode2, label='gr2 = {}'.format(gamma2))
    ax.legend()
    
    # Plot raw waves and fourier decomposition
    fig, ax = plt.subplots(nrows=4, sharex=True)
    im1 = ax[0].pcolormesh(t, x/dx, By.T*1e9, cmap='jet')
    ax[0].set_xlim(0, t_max)
    ax[0].set_ylim(0, x_max/dx)
    ax[0].set_ylabel('x (dx)', rotation=0)
    fig.colorbar(im1, ax=ax[0]).set_label('$B_y$\n(nT)', labelpad=15, rotation=0)
    
    im2 = ax[1].pcolormesh(t, x/dx, Bz.T*1e9, cmap='jet')
    ax[1].set_xlim(0, t_max)
    ax[1].set_ylim(0, x_max/dx)
    ax[1].set_ylabel('x (dx)', rotation=0)
    fig.colorbar(im2, ax=ax[1]).set_label('$B_z$\n(nT)', labelpad=15, rotation=0)
    
    # Bk.real gives By, Bk.imag gives Bz in +/-ve direction
    im3 = ax[2].pcolormesh(t, k, Bf, cmap='jet')
    ax[2].set_xlim(0, t_max)
    ax[2].set_ylabel('k (/m)', rotation=0)
    ax[2].set_xlabel('t (s)')
    fig.colorbar(im3, ax=ax[2]).set_label('$ln(P_y^+)$', labelpad=15, rotation=0)
    
    im4 = ax[3].pcolormesh(t, k, Bb, cmap='jet')
    ax[3].set_xlim(0, t_max)
    ax[3].set_ylabel('k (/m)', rotation=0)
    ax[3].set_xlabel('t (s)')
    fig.colorbar(im4, ax=ax[3]).set_label('$ln(P_y^-)$', labelpad=15, rotation=0)


## TEST :: Does FFT of B = By + iBz give identical output to above? Might be
##          same thing but quicker and easier.
# Do the same thing now, but for each k
# How does the slope of a particular B(k) relate to the growth rate?
# Should be able to test by having several separate waves superposed (a situation for
# which the energy wouldn't work)
# Didn't really work, growth rate leaked into every bin
if True:
    # Plot wave    with time
    # Plot k-space with time
    # Plot dominant mode/s with time
    #
    # Able to distinguish sign of k (-ve k?, forward propagating gives P-)
    # Direction of propagation can change (via -ve w) but helicity remains the same
    # Thus only k dependent? Still needs a little work to understand.
    
    print('Calculating helicity (maybe)')
    B_complex = (By + 1j*Bz)*1e9
    k         = np.fft.fftfreq(B_complex.shape[1], d=dx)
    B_fft     = np.fft.fft(B_complex) * 1. / k.shape[0]
    B_pwr     =(B_fft * np.conj(B_fft)).real.T
    B_log2    = np.log(B_pwr)
    
    # Power filter :: If any k has less than 10% of total power, remove entire k
    total_power = np.abs(B_pwr).sum()
    for ii in range(k.shape[0]):
        k_power = np.abs(B_pwr[ii]).sum()
        if k_power < 0.01 * total_power:
            B_pwr[ii] *= 0.0
            
    growth_rates = np.zeros(k.shape[0], dtype=np.float64)
    for ii in range(k.shape[0]):
        grad_k           = (B_log2[ii, -1] - B_log2[ii, 0]) / t[-1]
        growth_rates[ii] =  0.5 * grad_k / pcyc

    k            = np.fft.fftshift(k)
    growth_rates = np.fft.fftshift(growth_rates)
    #%%
    plt.plot(k, growth_rates)
    plt.xlabel('k (/m)')
    plt.ylabel('$\gamma/\Omega_H$')
    plt.title('Fourier-mode growth rates :: Test signal')
    plt.scatter(k1, gr1/pcyc, c='r')
    plt.scatter(k2, gr2/pcyc, c='r')

    
    if False:
        mode1     = B_log2[29]
        mode2     = B_log2[408]
        
        grad1 = (mode1[-1] - mode1[0]) / t[-1]
        gamma1= 0.5 * grad1 / pcyc
        
        grad2 = (mode2[-1] - mode2[0]) / t[-1]
        gamma2= 0.5 * grad2 / pcyc
        
        # Plot main modes + growth rate extraction
        fig, ax = plt.subplots()
        ax.plot(t, mode1, label='gr1 = {}'.format(gamma1))
        ax.plot(t, mode2, label='gr2 = {}'.format(gamma2))
        ax.legend()
        
        #%%
        # Plot raw waves and fourier decomposition
        fig, ax = plt.subplots(nrows=3, sharex=True)
        im1 = ax[0].pcolormesh(t, x/dx, By.T*1e9, cmap='jet')
        ax[0].set_xlim(0, t_max)
        ax[0].set_ylim(0, x_max/dx)
        ax[0].set_ylabel('x (dx)', rotation=0)
        fig.colorbar(im1, ax=ax[0]).set_label('$B_y$\n(nT)', labelpad=15, rotation=0)
        
        im2 = ax[1].pcolormesh(t, x/dx, Bz.T*1e9, cmap='jet')
        ax[1].set_xlim(0, t_max)
        ax[1].set_ylim(0, x_max/dx)
        ax[1].set_ylabel('x (dx)', rotation=0)
        fig.colorbar(im2, ax=ax[1]).set_label('$B_z$\n(nT)', labelpad=15, rotation=0)
        
        # Bk.real gives By, Bk.imag gives Bz in +/-ve direction
        im3 = ax[2].pcolormesh(t, k, B_log2, cmap='jet')
        ax[2].set_xlim(0, t_max)
        ax[2].set_ylabel('k (/m)', rotation=0)
        ax[2].set_xlabel('t (s)')
        fig.colorbar(im3, ax=ax[2]).set_label('$P$', labelpad=15, rotation=0)
