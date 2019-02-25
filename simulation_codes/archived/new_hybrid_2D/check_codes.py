# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:42:13 2017

@author: iarey
"""
from const       import dx, dy, N, NX, NY, va, cellpart, xmax, ymax
from part_params import velocity, idx_bounds, Nj
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as patches

def check_cell_dist_2d(part, node, species):
    xlocs = (np.arange(0, NX + 2) - 0.5) * dx     # Spatial locations (x, y) of E-field nodes
    ylocs = (np.arange(0, NY + 2) - 0.5) * dy
    X, Y  = np.meshgrid(xlocs, ylocs)
    
    f     = np.zeros((1, 6), dtype=float)       # Somewhere to put the particle data
    count = 0                                   # Number of particles in area

    # Collect particle infomation if within (0.5dx, 0.5dy) of node
    for ii in range(N):
        if ((abs(part[0, ii] - xlocs[node[0]]) <= dx) and
            (abs(part[1, ii] - ylocs[node[1]]) <= dy) and
            (part[2, ii] == species)):

            f = np.append(f, [part[0:6, ii]], axis=0)
            count += 1
    print 'Node (%d, %d) recieving contributions from %d particles.' % (node[0], node[1], count)

    plt.rc('grid', linestyle='dashed', color='black', alpha=0.3)
    
    # Draw figure and spatial boundaries
    fig = plt.figure(1)
    ax = fig.add_subplot(111, aspect='equal')
    ax.add_patch(patches.Rectangle((0, 0), xmax, ymax, fill=False, edgecolor='green'))
   
    # Shade in cell containing E-field node. Not useful due to particle shapes?
    ax.add_patch(patches.Rectangle(((node[0] - 1.5)*dx, (node[1] - 1.5)*dy),    # Bottom left position
                                   2*dx, 2*dy,                                  # Rectangle dimensions
                                   facecolor='grey',                        # Rectangle colour
                                   edgecolor='none',                        # Rectangle edge colour (no edges)
                                   alpha=0.5))                              # Rectangle opacity
    
    # Draw cell grid
    ax.set_xticks(np.arange(0, xmax+dx, dx))
    ax.set_yticks(np.arange(0, ymax+dy, dy))
    plt.grid(True)

    # Plot data and set limits
    ax.scatter(X, Y, s=10, c='red', marker='^')                      # Draw nodes
    ax.scatter(part[0, :], part[1, :], s=1, c='blue')   # Draw particles
    ax.set_xlim(-dx, xmax+dx)
    ax.set_ylim(-dy, ymax+dy)
    ax.set_title(r'$N_{cell} = %d$' % (np.sqrt(cellpart/Nj)))
    
    fig2 = plt.figure(2, figsize=(12, 10))
    fig2.patch.set_facecolor('w')
    num_bins = 25
    vmag = np.sqrt(part[3, :] ** 2 + part[4, :] ** 2 + part[5, :] ** 2)

    xax = plt.subplot2grid((2, 2), (0, 0))  
    yax = plt.subplot2grid((2, 2), (0, 1))  
    zax = plt.subplot2grid((2, 2), (1, 0))  
    tax = plt.subplot2grid((2, 2), (1, 1))

    xs, BinEdgesx = np.histogram((f[:, 3] - velocity[species]), bins=num_bins)
    bx = 0.5*(BinEdgesx[1:] + BinEdgesx[:-1])
    xax.plot(bx, xs, '-', c='c', drawstyle='steps')
    xax.set_xlabel(r'$v_x$')

    ys, BinEdgesy = np.histogram((f[:, 4]), bins=num_bins)
    by = 0.5*(BinEdgesy[1:] + BinEdgesy[:-1])
    yax.plot(by, ys, '-', c='c', drawstyle='steps')
    yax.set_xlabel(r'$v_y$')
    
    zs, BinEdgesz = np.histogram((f[:, 5]), bins=num_bins)
    bz = 0.5*(BinEdgesz[1:] + BinEdgesz[:-1])
    zax.plot(bz, zs, '-', c='c', drawstyle='steps')
    zax.set_xlabel(r'$v_z$')

    ts, BinEdgest = np.histogram(vmag, bins=num_bins)
    bt = 0.5*(BinEdgest[1:] + BinEdgest[:-1])
    tax.plot(bt, ts, '-', c='c', drawstyle='steps')
    tax.set_xlabel(r'$|v|$')

    plt.show()
    return

def check_cell_distribution(part, node_number, j): #        
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
    ax_x.set_xlabel(r'$v_x$')
    ax_x.set_xlim(-2, 2)
    
    ys, BinEdgesy = np.histogram(f[:, 4] / va, bins=num_bins)
    by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
    ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
    ax_y.set_xlabel(r'$v_y$')
    ax_y.set_xlim(-2, 2)
    
    zs, BinEdgesz = np.histogram(f[:, 5] / va, bins=num_bins)
    bz = 0.5 * (BinEdgesz[1:] + BinEdgesz[:-1])
    ax_z.plot(bz, zs, '-', c='c', drawstyle='steps')
    ax_z.set_xlabel(r'$v_z$')
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
    ax_x.set_xlabel(r'$v_x$')
    
    ys, BinEdgesy = np.histogram(part[4, idx_bounds[j, 0]: idx_bounds[j, 1]] / va, bins=num_bins)
    by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
    ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
    ax_y.set_xlabel(r'$v_y$')
    
    zs, BinEdgesz = np.histogram(part[5, idx_bounds[j, 0]: idx_bounds[j, 1]] / va, bins=num_bins)
    bz = 0.5 * (BinEdgesz[1:] + BinEdgesz[:-1])
    ax_z.plot(bz, zs, '-', c='c', drawstyle='steps')
    ax_z.set_xlabel(r'$v_z$')

    plt.show()
    return