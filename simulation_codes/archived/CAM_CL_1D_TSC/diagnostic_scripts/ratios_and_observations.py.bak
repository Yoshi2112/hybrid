# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 09:23:55 2019

@author: Yoshi
"""

import matplotlib.pyplot as plt
import numpy as np

mu0  = 4e-7*np.pi
e0   = 8.854e-12
mi   = 1.673e-27
c    = 3e8
npts = 100
rat  = np.sqrt(e0 / mi)

B_min = 0e-9
B_max = 300e-9
B     = np.linspace(B_min, B_max, npts)

powers = range(1, 5)

for pwr in powers:
    sqrt_n = (rat * (10 ** pwr) * B) / 1e3
    n      = (rat * (10 ** pwr) * B) ** 2 / 1e6
    plt.plot(B*1e9, n, label='$cv_A^{-1} = 10^%d$' % pwr)

plt.xlim(B_min*1e9, B_max*1e9)
plt.ylim(0, 500)
plt.xlabel(r'$B_0 (nT)$')
plt.ylabel(r'$n_0 ({cm^{-3}})$')
plt.title('Plasma Regimes: \'Arbitrary\' ratio vs. Plasma Parameters')
plt.legend()

B_vals = np.array([158, 158, 134, 134])
n_vals = np.array([38, 160, 38, 160])
plt.scatter(B_vals, n_vals, label='Observations', marker='x')
plt.legend()