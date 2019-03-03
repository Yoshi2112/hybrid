# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:42:13 2017

@author: iarey
"""
from simulation_parameters_1D       import dx, N, NX, va, cellpart, velocity, idx_bounds
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
    num_bins = cellpart/20

    ax_x = plt.subplot2grid((2, 3), (0,0), colspan=2, rowspan=2)
    ax_y = plt.subplot2grid((2, 3), (0,2))
    ax_z = plt.subplot2grid((2, 3), (1,2))

    xs, BinEdgesx = np.histogram((f[:, 3] - velocity[j]) / va, bins=num_bins)
    bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
    ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
    ax_x.set_xlabel(r'$v_x / v_A$')

    ys, BinEdgesy = np.histogram(f[:, 4] / va, bins=num_bins)
    by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
    ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
    ax_y.set_xlabel(r'$v_y / v_A$')

    zs, BinEdgesz = np.histogram(f[:, 5] / va, bins=num_bins)
    bz = 0.5 * (BinEdgesz[1:] + BinEdgesz[:-1])
    ax_z.plot(bz, zs, '-', c='c', drawstyle='steps')
    ax_z.set_xlabel(r'$v_z / v_A$')

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