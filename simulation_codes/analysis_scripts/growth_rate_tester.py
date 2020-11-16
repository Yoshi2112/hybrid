# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:02:41 2020

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt

# To do: Maybe see if you can do this in Fourier space? What's the conversion 
#         factor? Would allow direct comparison against linear theory.

B0  = 200e-9
mp  = 1.673e-27
qp  = 1.602e-19
e0  = 8.854e-12
mu0 = 4e-7*np.pi

pcyc = qp * B0 / mp
pinv = 1. / pcyc

t_max = 200. * pinv
dt    = 0.05 * pinv
t     = np.arange(0, t_max, dt)
tinv  = t*pcyc

sat   = 0.043  * B0
Bw    = 4.0e-4 * B0
freq  = 0.3    * pcyc   
gr    = 0.032  * pcyc

wave  = Bw * np.exp(-1j * (freq*t)) * np.exp(gr*t)

By    = wave.real
Bz    = wave.imag
Bt    = abs(wave)
B_sq  = (By ** 2 + Bz ** 2)
ln_B2 = np.log(B_sq)

if True:
    fig, ax = plt.subplots()
    ax.plot(t, By)
    ax.plot(t, Bz)
    ax.plot(t, Bt)
    ax.set_xlim(0, t_max)
    ax.set_yscale('log')

if False:
    fig, ax = plt.subplots()
    ax.plot(tinv, ln_B2)
    #ax.set_ylim(1e-6, 1e-2)
    ax.set_xlim(0, 1200)
    ax.grid(axis='both', ls=':')
    
# Can extract growth rates either from the square of the wave field, or by 
#    just considering its norm (magnitude). Both work.
    
# =============================================================================
# growth_rate   = (np.log(Bt[-1]) - np.log(Bt[0])) / t[-1]
# gr_normalized = growth_rate / pcyc 
# =============================================================================

growth_rate   = 0.5 * (ln_B2[-1] - ln_B2[0]) / t[-1]
gr_normalized = growth_rate / pcyc 

print('GR raw: ', growth_rate)
print('GR norm:', gr_normalized)