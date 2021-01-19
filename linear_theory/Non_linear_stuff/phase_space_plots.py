# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:56:57 2021

@author: Yoshi
"""
import warnings, sys, pdb
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    S      = 0.4 
    zeta   = np.linspace(0, 2*np.pi, 250)
    theta  = np.linspace(-2, 2, 1000) 
    ze, th = np.meshgrid(zeta, theta)
    
    # Calculate curves of constancy
    const = np.zeros(th.shape)
    for ii in range(th.shape[0]):
        for jj in range(th.shape[1]):
            const[ii, jj] = th[ii, jj]**2 + 0.5 * (np.cos(ze[ii, jj]) - S * ze[ii, jj])
    
    # Work out const of separatrix
    ze0         =   np.pi + np.arcsin(np.abs(S))
    ze1         = 2*np.pi - np.arcsin(np.abs(S))    
    separatrix  = 0.5*(np.cos(ze1) - S*ze1)
    
    # Manually select contours
    
    # Plot curves
    fig, ax = plt.subplots()
    ax.contour(ze, th, const, levels=100)
    ax.contour(ze, th, const, levels=[separatrix], colors='k', linestyles='-')

    ax.axvline(ze0, c='k', ls='--', alpha=0.25)
    ax.axvline(ze1, c='k', ls='--', alpha=0.25)
