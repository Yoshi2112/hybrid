# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:52:42 2020

@author: Yoshi
"""
import numba as nb
import numpy as np
import matplotlib.pyplot as plt


@nb.njit()
def add_J_ext(Ji,time):
    '''
    Driven J designed as energy input into simulation. All parameters specified
    in the simulation_parameters script/file
    
    Designed as a Gaussian pulse so that things don't freak out by rising too 
    quickly. Just test with one source point at first
    
    L mode is -90 degree phase in Jz
    '''
    # Soft source wave (What t corresponds to this?)
    # Should put some sort of ramp on it?
    # Also needs to be polarised. By or Bz lagging/leading?
    phase    = -90
    gaussian = np.exp(- ((time - pulse_offset)/ pulse_width) ** 2 )

    # Set new field values in array as soft source
    Ji[center, 1] = driven_ampl*gaussian*np.sin(2 * np.pi * freq * time)
    Ji[center, 2] = driven_ampl*gaussian*np.sin(2 * np.pi * freq * time + phase * np.pi / 180.)    
    return

@nb.njit()
def add_J_ext_pol(Ji, time):
    '''
    Driven J designed as energy input into simulation. All parameters specified
    in the simulation_parameters script/file
    
    Designed as a Gaussian pulse so that things don't freak out by rising too 
    quickly. Just test with one source point at first
    
    Polarised with a LH mode only, uses five points with both w, k specified
    -- Not quite sure how to code this... do you just add a time delay (td, i.e. phase)
        to both the envelope and sin values at each point? 
        
    -- Source node as td=0, other nodes have td depending on distance from source, 
        (ii*dx) and the wave phase velocity v_ph = w/k (which are both known)
    
    P.S. A bunch of these values could be put in the simulation_parameters script.
    Optimize later (after testing shows that it actually works!)
    '''
    phase = -np.pi / 2
    
    # Points -2, -1, 0, 1, 2 dx from center
    n_pts = 10
    for off in np.arange(-n_pts, n_pts+1):
        del_t       = abs(off*dx / vph)
        
        gauss = driven_ampl * np.exp(- ((time - pulse_offset - del_t)/ pulse_width) ** 2 )
        
        Ji[center + off, 1] = gauss * np.sin(2 * np.pi * freq * (time - del_t))
        Ji[center + off, 2] = gauss * np.sin(2 * np.pi * freq * (time - del_t) + phase)    
    return

driven_ampl  = 1.0
pulse_offset = 2.5
pulse_width  = 1.0

dt   = 0.01
dx   = 16e3
vph  = 3e5
freq = 1.0
k    = 2 * np.pi * freq / vph
off  = 0.0 

Jpos   = np.arange(-(20 + 0.5)*dx, 21*dx, dx)
J      = np.zeros((Jpos.shape[0], 3))
center = Jpos.shape[0] // 2

add_J_ext(J, 0.0)


fig, ax   = plt.subplots(1, figsize=(15, 10), sharex=True)
graph_Jy, = ax.plot(Jpos, J[:, 1])
graph_Jz, = ax.plot(Jpos, J[:, 2])
ax.set_ylim(-1, 1)
for ii in range(1, 1000):
    t = ii * dt
    #add_J_ext(J, t)
    add_J_ext_pol(J, t)
    
    graph_Jy.set_ydata(J[:, 1])
    graph_Jz.set_ydata(J[:, 2])
    plt.draw()                                      # Draw plot. Updates figure with each timestep
    plt.pause(0.001)
    

