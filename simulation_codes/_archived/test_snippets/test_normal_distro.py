# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:17:37 2020

@author: Yoshi
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# Create array all at once, test mean and STD (and other moments?)
# Also, create an array piecemeal, of same size. Also test mean and STD.

N_vals = 1024
N_mom  = 4

big_array = np.random.normal(size=N_vals)

bit_array = np.zeros(N_vals)
for ii in range(N_vals):
    bit_array[ii] = np.random.normal()

big_moments = np.zeros(N_mom)
bit_moments = np.zeros(N_mom)

for ii in range(N_mom):
    big_moments[ii] = st.moment(big_array, moment=ii)
    bit_moments[ii] = st.moment(bit_array, moment=ii)
    
# Collect histogram, plot with moments displayed
plt.figure()
plt.hist(big_array, bins=N_vals//100)
plt.xlim(-4, 4)
#plt.ylim(0, N_vals // 20)
plt.title('Moments: {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(*big_moments))

plt.figure()
plt.hist(bit_array, bins=N_vals//100)
plt.xlim(-4, 4)
#plt.ylim(0, N_vals // 20)
plt.title('Moments: {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(*bit_moments))
