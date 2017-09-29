# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:51:36 2017

@author: iarey
"""

import const
import numpy as np

seed = 21

species    = [r'$H^+$ cold', r'$H^+$ hot']     # Species name/labels : Used for plotting
temp_type  = np.asarray([0, 1])                # Particle temperature type: Hot or cold
dist_type  = np.asarray([0, 0])                # Particle distribution type: Uniform or sinusoidal/other

mass    = np.asarray([1.008, 1.008]) * 1.661e-27        # Species ion mass (amu to kg)
charge  = np.asarray([1.00 , 1.00 ]) * 1.602e-19        # Species ion charge (elementary charge units to Coulombs)

velocity    = np.asarray([-0.15*const.va, 10*const.va]) # Species bulk velocity (m/s)
proportion  = np.asarray([0.985, 0.015])                # Species proportionality of whole density
density     = np.asarray(proportion * const.ne)         # Real density as a proportion of nes
sim_repr    = np.asarray([0.5, 0.5])                    # Simulation representation: Proportion of simulation particles for each species

N_real    = const.NX * const.dx * const.ne              # Total number of real particles (charges? electrons)
N_species = np.round(const.N * sim_repr).astype(int)    # Number of sim particles for each species, total 
Nj        = len(mass)                                   # Number of species (number of columns above)    
 
n_contr   = (N_real * proportion) / N_species           # Species density contribution: Real particles per sim particle

Tpar = np.array([0.5, 0.5]) * 11603                   # Parallel ion temperature
Tper = np.array([0.5, 0.5]) * 11603                   # Perpendicular ion temperature

idx_start = np.asarray([np.sum(N_species[0:ii]    )     for ii in range(0, Nj)])    # Start index values for each species in order
idx_end   = np.asarray([np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)])    # End   index values for each species in order
idx_bounds= np.stack((idx_start, idx_end)).transpose()  # Array index boundary values: idx_bounds[species, start/end]