# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 23:36:48 2022

@author: Yoshi
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt

def PFLCD(x, y):
    '''
    Partially-filled loss-cone distribution as per eqns 6.41-6.42 of
    Baumjohann (1997). Can take single values or 1D arrays for x, y
    
    INPUTS:
    x :: Parallel velocity variable
    y :: Perpendicular velocity variable
    
    OUTPUTS:
    fx :: Value of the distribution function at point (x,y)
    
    GLOBALS:
    vth_para :: Parallel thermal velocity (m/s)
    vth_perp :: Perpendicular thermal velocity (m/s)
    density  :: Total density of the distribution (/m3)
    beta     :: Determines the slope of the distribution in the loss cone
    delta    :: Determines loss cone fullness
    
    QUESTION: How do beta, delta relate to the size of the loss cone in degrees?
    '''
    exp1 = delta*np.exp(- y ** 2 / (2*vth_perp**2))
    exp2 = np.exp(- y ** 2 / (2*vth_perp**2)) - np.exp(- y ** 2 / (2*beta*vth_perp**2))
    gx   = exp1 + ((1 - delta) / (1 - beta)) * exp2
    
    fx  = density / (8*np.pi**3*vth_para*vth_perp**2) * np.exp(- x ** 2 / (2*vth_para**2))
    fx *= gx
    return fx

def generate_distribution(xmin, xmax, ymin, ymax, 
                          n_samples=1000):
    '''
    Randomly generates n samples of a target_function distribution by
    a Monte-Carlo rejection method.
    
    xmin, xmax -- v_perp limits
    ymin, ymax -- v_para limits
    n_samples  -- Number of samples to collect from distribution
    '''
    print('Generating Loss-Cone Distribution...')
    #print('Finding maximum value of target function')
    test_n = 500
    test_x = np.linspace(xmin, xmax, test_n)
    test_y = np.linspace(ymin, ymax, test_n)
    P_max = 0.0
    for ii in range(test_n):
        for jj in range(test_n):
            test_P = PFLCD(test_x[ii], test_y[jj])
            if test_P > P_max: P_max = test_P
    P_max *= 1.005   # Pad a little to make up for inaccuracy in test sampling
    #print('Maximum value of function is', P_max)
    
    #print('Initiating rejection test sampling')
    n_batch = 5*n_samples
    distro = np.zeros((n_samples, 2))
    acc = 0; batch_count = 0
    while acc < n_samples:
        cx = np.random.uniform(xmin, xmax, n_batch)     # Sample
        cy = np.random.uniform(ymin, ymax, n_batch)
        cP = PFLCD(cx, cy)                    # Evaluate
        z  = np.random.uniform(0, P_max, n_batch)       # Generate a uniform distribution between 0 and Py_max, z
        batch_count += 1
        # If z < P(x,y) then accept sample, otherwise reject
        for ii in range(n_batch):
            if z[ii] < cP[ii]:
                distro[acc] = cx[ii], cy[ii]
                acc += 1
                
                if acc == n_samples:
                    print('Completed after {} batches.'.format(batch_count))
                    return distro
    raise Exception('You should never get here')

if __name__ == '__main__':
    qi = 1.602e-19
    mi = 1.673e-27
    
    # Sample distribution values
    K        = 4     # Number of thermal velocities to go in x, y (multiplier constant)
    beta     = 0.8
    delta    = 0.2
    density  = 50e6  # /m3
    Tpara    = 5     # eV
    Tperp    = 5     # eV
    vth_para = np.sqrt(qi * Tpara /  mi)   # Parallel thermal velocity (m/s)
    vth_perp = np.sqrt(qi * Tperp /  mi)   # Perpendicular thermal velocity (m/s)
    
    sampled_distro = generate_distribution(-K*vth_perp, K*vth_perp, -K*vth_para, K*vth_para, n_samples=1000000)
    
    plt.ioff()
    hist, xedges, yedges, image = plt.hist2d(sampled_distro[:, 1], sampled_distro[:, 0], bins=100)
    plt.close('all')
    # Plot function
    if True:
        v_samples = 200
        
        vpara     = np.linspace(-K*vth_para, K*vth_para, v_samples, dtype=np.float32)
        vperp     = np.linspace(-K*vth_perp, K*vth_perp, v_samples, dtype=np.float32)
        
        Pv        = np.zeros((vpara.shape[0], vperp.shape[0]), dtype=np.float32)
        for ii in range(vpara.shape[0]):
            for jj in range(vperp.shape[0]):
                Pv[ii, jj] = PFLCD(vpara[ii], vperp[jj])
        Pv /= Pv.max()
                
        fig, axes = plt.subplots(2)
        axes[0].set_title('Loss Cone Distribution :: $\\beta = $%.2f :: $\delta =$ %.2f' % (beta, delta))
        
        im0 = axes[0].pcolormesh(vperp, vpara, Pv)
        fig.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].pcolormesh(yedges, xedges, hist.T)
        fig.colorbar(im1, ax=axes[1])
        
        for ax in axes:
            ax.set_xlabel('$v_\perp$', rotation=0)
            ax.set_ylabel('$v_\parallel$', rotation=0)
            ax.set_ylim(vpara.min(), vpara.max())
            ax.set_xlim(vperp.min(), vperp.max())
        
        
        
        plt.show()