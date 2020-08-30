# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 20:16:30 2020

@author: Yoshi
"""
import numpy as np
import matplotlib.pyplot as plt

from kbit_reversed_fractions import rkbr_uniform_set

if __name__ == '__main__':
    v_thermal = 30000      # 30km/s
    xmax      = 100.0
    
    # Test 1 : Does it replicate a 2v Maxwellian ok? Yes it does!
    if False:
        n_part    = 2 ** 18    # Number of particles
        arr       = np.arange(n_part)
        
        R_vr    = rkbr_uniform_set(arr+1, base=2)
        R_theta = rkbr_uniform_set(arr  , base=3)
    
        vr  = v_thermal * np.sqrt(-2 * np.log((arr + 0.5) / n_part))
        vr2 = v_thermal * np.sqrt(-2 * np.log(R_vr))
    
        plt.figure()
        plt.hist(vr , bins=50)        
        
        plt.figure()
        plt.hist(vr2, bins=50)  
    
    # Test 2 : Loading vy, vz from spoke/ring model
    # Generate N_rings from vr, N_spokes in theta. Generate products.
    # This replicates vx, vy very well! But doesn't seem to randomize in space?
    # Is this due to the common factor between 2,4? Yes! 5 also seemed to have some structure.
    if False:
        N_rings  = 2 ** 9
        N_spokes = 2 ** 9
        
        arr_rings  = np.arange(N_rings)
        arr_spokes = np.arange(N_spokes)
        arr_pos    = np.arange(N_rings*N_spokes)
        
        R_vr    = rkbr_uniform_set(arr_rings+1, base=2)
        R_theta = rkbr_uniform_set(arr_spokes , base=3) 
        R_pos   = rkbr_uniform_set(arr_pos    , base=7)
        
        vr    = v_thermal * np.sqrt(-2 * np.log(R_vr))
        theta = R_theta * np.pi * 2
        
        vx  = np.zeros(N_rings * N_spokes)
        vy  = np.zeros(N_rings * N_spokes)
        pos = R_pos * xmax
        for ii in range(N_rings):
            for jj in range(N_spokes):
                idx = ii*N_spokes + jj
                
                vx[idx] = vr[ii] * np.sin(theta[jj])
                vy[idx] = vr[ii] * np.cos(theta[jj])
        
        plt.scatter(pos, vx, s=1)
        
    
    # Test 3 : Loading vy, vz with single spoke/ring method, i.e. just with the uniform sets - Works!
    if True:
        n_part  = 2 ** 17
        arr     = np.arange(n_part)
        
        R_vr    = rkbr_uniform_set(arr+1, base=2)
        R_theta = rkbr_uniform_set(arr  , base=3) 
        R_vrx   = rkbr_uniform_set(arr+1, base=5)
            
        pos     = np.linspace(0, 1, n_part, endpoint=True) * xmax
        vr      = 2 * v_thermal * np.sqrt(-2 * np.log(R_vr ))
        vrx     =     v_thermal * np.sqrt(-2 * np.log(R_vrx))
        theta   = R_theta * np.pi * 2
        
        vx      = vrx * np.sin(theta)
        vy      = vr  * np.sin(theta)
        vz      = vr  * np.cos(theta)
        
        fig, ax = plt.subplots(3, sharex=True)
        ax[0].hist(vx, bins=50)
        ax[1].hist(vy, bins=50)        
        ax[2].hist(vz, bins=50)  
        
        fig2, ax2 = plt.subplots(3, sharex=True)
        ax2[0].scatter(pos, vx, s=1)
        ax2[1].scatter(pos, vy, s=1)        
        ax2[2].scatter(pos, vz, s=1) 
        
        v_perp = np.sqrt(vy ** 2 + vz ** 2) * np.sign(vz)
        
        plt.figure()
        plt.scatter(vy, vz, s=1)
        
        plt.figure()
        plt.scatter(vx, v_perp, s=1)
        
        
        