# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:00:57 2020

@author: Yoshi
"""
import numpy as np

c   = 3e8
mp  = 1.673e-27
me  = 9.11e-31
e0  = 8.854e-12
mu0 = 4e-7 * np.pi 
kB  = 1.380649e-23
B0  = 200e-9            # This is arbitrary: Just anchors parameters to one set of solutions

# Fu input variables
wpe_wce    = 200
beta_hot   = 10
nh_ratio   = 0.4
he_ratio   = 0.00
temp_ratio = 5.0

ne      = wpe_wce ** 2 * B0 ** 2 * e0 / me      # /m3
Th_para = beta_hot * B0 ** 2 / (2*mu0*ne*kB)    # Kelvin?
hH      = nh_ratio * ne                         # Hot proton density
cHe     = he_ratio * ne                         # Cold helium density
cH      = ne - cHe - hH                         # Cold proton density
anis    = temp_ratio - 1                        # Dimensionless
Th_perp = Th_para * temp_ratio                  # Kelvin?

alpha_therm = np.sqrt(kB * Th_perp / mp)

print('Magnetic field:      {} nT'.format(B0*1e9))
print('Cold proton density: {} cc'.format(cH /1e6))
print('Cold helium density: {} cc'.format(cHe/1e6))
print('Hot  proton density: {} cc'.format(hH /1e6))
print('Hot anisotropy:      {}'.format(anis))
print('Hot T_perp:          {} eV'.format(Th_perp/11603.))
print('Perp Thermal vel.    {} c'.format(Th_perp/c))

