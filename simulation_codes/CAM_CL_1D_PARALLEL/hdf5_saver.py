# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 22:10:43 2022

@author: Yoshi
"""
from timeit import default_timer as timer
import os
import numpy as np
import h5py

test_dir = 'C://compression_tests//'
if not os.path.exists(test_dir): os.makedirs(test_dir)
   
    
# How long does it take to save 100 million particles?
N = int(100e6)
pos = np.random.normal(0, 1, N)
vel = np.random.normal(0, 1, (N, 3))
idx = np.random.randint(0, 4, N)
total_size = (pos.nbytes + vel.nbytes + idx.nbytes) / 1024 / 1024 / 1024
print(f'Array size on disk: {total_size:.3f} GB')

print('Saving...')
if False:
    # Save as npz and as hdf5
    np_start = timer()
    np.savez(test_dir + 'testarr.npz', pos=pos, vel=vel, idx=idx)
    np_time = timer() - np_start
    print(f'Numpy time: {np_time:.2f} seconds')
    
if True:
    h5_start = timer()
    compression_level = 5
    h5f = h5py.File(test_dir + 'testarr_szip.h5', 'w')
    h5f.create_dataset('pos', data=pos, compression="szip")
    h5f.create_dataset('vel', data=vel, compression="szip")
    h5f.create_dataset('idx', data=idx, compression="szip")
    h5f.close()
    h5_time = timer() - h5_start
    print(f'HDF5 time: {h5_time:.2f}  seconds')