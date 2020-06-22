# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:38:59 2020

@author: Yoshi
"""

import numpy as np

# Init arrays
A = np.arange(6, dtype=float)
B = np.zeros( 6, dtype=float)

print('Original values\n', A, B)

# Backup values from A in B
B[:] = A

# Change A
A[-1] = 0
A    *= 2
A    += 10.

# Does B keep original values as backup?
print('Changed values\n', A, B)

# Replace A with original values, zero storage array
A[:] = B
B   *= 0.

print('Restored values\n', A, B)