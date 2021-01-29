# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:56:57 2021

@author: Yoshi
"""
import warnings, sys, pdb
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    S_range = np.arange(-2.0, 2.01, 0.01)
    for qq in range(S_range.shape[0]):
        S      = S_range[qq]
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
            
        # Plot curves
        plt.ioff()
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.contour(ze, th, const, levels=75, colors='k', linestyles='-', alpha=0.5)
        ax.contour(ze, th, const, levels=[separatrix], colors='k', linestyles='-')
    
        ax.contourf(ze, th, const, levels=np.array([separatrix, const.max()]), colors='grey')
        ax.axvspan(ze1, 2*np.pi, color='grey')
    
        ax.axvline(ze0, c='k', ls='--', alpha=0.25)
        ax.axvline(ze1, c='k', ls='--', alpha=0.25)
        
        ax.set_title('Resonant proton trajectories :: S = {:.2f}'.format(S), fontsize=20)
        ax.set_xlabel('$\zeta$', fontsize=24)
        ax.set_ylabel('$\\frac{\\theta}{2 \omega_{tr}}$', rotation=0, labelpad=20, fontsize=24)
        
        savedir  = 'C://Users//iarey//Documents//GitHub//hybrid//linear_theory//Non_linear_stuff//phase_spaces//'
        filename = 'phase_space_{:04}.png'.format(qq)
        fig.savefig(savedir + filename)
        plt.close('all')
