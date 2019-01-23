# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:42:13 2017

@author: iarey
"""
from simulation_parameters_1D       import dx, N, NX, va, cellpart, velocity, idx_bounds, q, mp
import simulation_parameters_1D as const
from particles_1D import boris_algorithm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

def check_cell_velocity_distribution(part, node_number, j): #
    '''Checks the velocity distribution of a particle species within a specified cell'''
    # Collect information about particles within +- 0.5dx of node_number (E-field nodes are in the cell centers)
    x_node = (node_number - 0.5) * dx   # Position of node in question
    f = np.zeros((1, 6))
    count = 0

    for ii in range(N):
        if (abs(part[0, ii] - x_node) <= 0.5*dx) and (part[2, ii] == j):
            f = np.append(f, [part[0:6, ii]], axis=0)
            count += 1

    #Plot it
    rcParams.update({'text.color'   : 'k',
            'axes.labelcolor'   : 'k',
            'axes.edgecolor'    : 'k',
            'axes.facecolor'    : 'w',
            'mathtext.default'  : 'regular',
            'xtick.color'       : 'k',
            'ytick.color'       : 'k',
            'axes.labelsize'    : 24,
            })

    fig = plt.figure(figsize=(12,10))
    fig.patch.set_facecolor('w')
    num_bins = cellpart/5

    ax_x = plt.subplot2grid((2, 3), (0,0), colspan=2, rowspan=2)
    ax_y = plt.subplot2grid((2, 3), (0,2))
    ax_z = plt.subplot2grid((2, 3), (1,2))

    xs, BinEdgesx = np.histogram((f[:, 3] - velocity[j]) / va, bins=num_bins)
    bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
    ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
    ax_x.set_xlabel(r'$v_x / v_A$')
    ax_x.set_xlim(-2, 2)

    ys, BinEdgesy = np.histogram(f[:, 4] / va, bins=num_bins)
    by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
    ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
    ax_y.set_xlabel(r'$v_y / v_A$')
    ax_y.set_xlim(-2, 2)

    zs, BinEdgesz = np.histogram(f[:, 5] / va, bins=num_bins)
    bz = 0.5 * (BinEdgesz[1:] + BinEdgesz[:-1])
    ax_z.plot(bz, zs, '-', c='c', drawstyle='steps')
    ax_z.set_xlabel(r'$v_z / v_A$')
    ax_z.set_xlim(-2, 2)

    plt.show()
    return

def check_position_distribution(part, j):
    '''Checks the spatial distribution of a particle species j within the spatial domain'''
    #Plot it
    rcParams.update({'text.color'   : 'k',
            'axes.labelcolor'   : 'k',
            'axes.edgecolor'    : 'k',
            'axes.facecolor'    : 'w',
            'mathtext.default'  : 'regular',
            'xtick.color'       : 'k',
            'ytick.color'       : 'k',
            'axes.labelsize'    : 24,
            })

    fig = plt.figure(figsize=(12,10))
    fig.patch.set_facecolor('w')
    num_bins = NX

    ax_x = plt.subplot()

    xs, BinEdgesx = np.histogram(part[0, idx_bounds[j, 0]: idx_bounds[j, 1]] / float(dx), bins=num_bins)
    bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
    ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
    ax_x.set_xlabel(r'$x_p$')
    ax_x.set_xlim(0, NX)

    plt.show()
    return

def check_velocity_distribution(part, j):
    '''Checks the velocity distribution of an entire species across the simulation domain '''
    #Plot it
    rcParams.update({'text.color'   : 'k',
            'axes.labelcolor'   : 'k',
            'axes.edgecolor'    : 'k',
            'axes.facecolor'    : 'w',
            'mathtext.default'  : 'regular',
            'xtick.color'       : 'k',
            'ytick.color'       : 'k',
            'axes.labelsize'    : 24,
            })

    fig = plt.figure(figsize=(12,10))
    fig.patch.set_facecolor('w')
    num_bins = cellpart / 5

    ax_x = plt.subplot2grid((2, 3), (0,0), colspan=2, rowspan=2)
    ax_y = plt.subplot2grid((2, 3), (0,2))
    ax_z = plt.subplot2grid((2, 3), (1,2))

    xs, BinEdgesx = np.histogram(part[3, idx_bounds[j, 0]: idx_bounds[j, 1]] / va, bins=num_bins)
    bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
    ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
    ax_x.set_xlabel(r'$v_x / v_A$')

    ys, BinEdgesy = np.histogram(part[4, idx_bounds[j, 0]: idx_bounds[j, 1]] / va, bins=num_bins)
    by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
    ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
    ax_y.set_xlabel(r'$v_y / v_A$')

    zs, BinEdgesz = np.histogram(part[5, idx_bounds[j, 0]: idx_bounds[j, 1]] / va, bins=num_bins)
    bz = 0.5 * (BinEdgesz[1:] + BinEdgesz[:-1])
    ax_z.plot(bz, zs, '-', c='c', drawstyle='steps')
    ax_z.set_xlabel(r'$v_z / v_A$')

    plt.show()
    return


def boris_alg(v0, Bp, Ep, dt):
    '''Updates the velocity of a particle using the Boris method as detailed
    in Birdsall & Langdon (1985),  pp. 59-63.

    INPUT:
        v0   -- Original particle velocity
        B    -- Magnetic field value at particle  position
        E    -- Electric field value at particle position
        dt   -- Simulation time cadence
        
    OUTPUT:
        v0   -- Updated particle velocity (overwrites input value on return)
    '''
    qi = 1.602e-19                                                      # Proton charge
    mi = 1.673e-27                                                      # Proton mass
    
    v_minus = np.zeros(3)                                               # First velocity
    v_prime = np.zeros(3)                                               # Rotation velocity
    v_plus  = np.zeros(3)                                               # Second velocity

    T = (qi * Bp / mi) * dt / 2.                        # Boris variable
    S = 2.*T / (1. + np.sqrt(T[0] ** 2 + T[1] ** 2 + T[2] ** 2))        # Boris variable

    # Actual Boris Method
    v_minus    = v0 + qi * Ep * dt / (2. * mi)

    v_prime[0] = v_minus[0] + (v_minus[1] * T[2] - v_minus[2] * T[1])   # Removed multiplicative from second term: (charge[idx] * dt / (2 * mass[idx]))
    v_prime[1] = v_minus[1] - (v_minus[0] * T[2] - v_minus[2] * T[0])
    v_prime[2] = v_minus[2] + (v_minus[0] * T[1] - v_minus[1] * T[0])

    v_plus[0]  = v_minus[0] + (v_prime[1] * S[2] - v_prime[2] * S[1])
    v_plus[1]  = v_minus[1] - (v_prime[0] * S[2] - v_prime[2] * S[0])
    v_plus[2]  = v_minus[2] + (v_prime[0] * S[1] - v_prime[1] * S[0])
 
    v_out = v_plus + qi * Ep * dt / (2. * mi)
    
    # Check
    mag1 = v_minus.dot(v_minus)
    mag2 = v_plus.dot(v_plus)
    diff = 100.*abs(mag1 - mag2) / min(mag1, mag2)
    print diff
    return v_out


def test_particle_orbit():
    v0  = 100.
    B0  = 4e-9
    Bc  = np.array([B0, 0., 0.]) 
    Ec  = np.array([0., 0., 0.]) 
    vp  = np.array([0., v0, 0.]) 
    
    resolution = 2000
    num_rev    = 1
    maxtime    = resolution*num_rev
    gyperiod   = (2 * np.pi * mp) / (q * B0) 
    dt         = gyperiod / resolution 

    v_initial  = v0 
    print '\nNumber of revolutions: {}'.format(num_rev)
    print 'Points per revolution: {}'.format(resolution)
    print 'Total number of points: {}'.format(maxtime)
    
    print '\nGyroperiod: {}s'.format(round(gyperiod, 2))
    print 'Timestep: {}s'.format(round(dt, 3))
    
    circle = plt.Circle((0, 0), v0, color='k', fill=False)
    fig = plt.figure()
    ax  = fig.gca()
    
    for ii in range(maxtime+1):
        plt.scatter(vp[1], vp[2], c='b', s=3)
        vp = boris_alg(vp, Bc, Ec, dt)
        
    plt.scatter(vp[1], vp[2], c='b', s=1)   
    #ax.add_artist(circle)
    plt.axis('equal')
    
    v_final = np.sqrt(vp[0] ** 2 + vp[1] ** 2 + vp[2] ** 2)
    
    print '\nInitial Velocity: {}m/s'.format(round(v_initial, 2))
    print 'Final Velocity:   {}m/s'.format(round(v_final, 2))    
    return

if __name__ == '__main__':
    test_particle_orbit()