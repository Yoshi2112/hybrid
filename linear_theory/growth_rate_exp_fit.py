# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:00:35 2019

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exp_func(xi, Ai, wi, gi):
    return Ai * np.exp(-1j*wi* xi).real * np.exp(gi*xi)

if __name__ == '__main__':
    qi   = 1.602e-19                                # C
    mi   = 1.673e-27                                # kg

    B0   = 9.7e-9                                   # T
    wcyc = qi * B0 / mi                             # rad/s
    
    A0   = 1.0                                      # T?
    w0   = 0.5 * wcyc                               # rad/s
    gam0 = 0.05 * w0                                # rad/s
    
    revs  = 20.                                     # Revolutions to plot for 
    gyper = 2 * np.pi / wcyc                        # Gyperiod in seconds
    max_t = revs * gyper                            # Maximum time in seconds
    dt    = 0.01 * gyper                            # Timestep in seconds
        
    xdata = np.arange(0., max_t + dt, dt)           # Time array in seconds
    ydata = exp_func(xdata, A0, w0, gam0).real
    
    yn = ydata + np.random.normal(0, 1.5, ydata.shape[0])
    popt, pcov = curve_fit(exp_func, xdata, yn, p0=[1.0, 0.5*wcyc, 0.0],
                                               bounds=(0, [10.0, wcyc, wcyc]))
    
    new_wave   = exp_func(xdata, *popt)
    
    print(*popt)
    plt.plot(xdata, yn)
    plt.plot(xdata, new_wave)
