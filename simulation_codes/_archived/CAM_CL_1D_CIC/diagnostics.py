# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:42:13 2017

@author: iarey
"""
import simulation_parameters_1D as const
import particles_1D             as particles
import numpy as np
import matplotlib.pyplot as plt
import pdb

def check_cell_velocity_distribution(part, node_number, j): #
    '''Checks the velocity distribution of a particle species within a specified cell
    '''
    x_node = (node_number - 0.5) * const.dx   # Position of E-field node
    f = np.zeros((1, 6))
    count = 0

    for ii in np.arange(const.idx_bounds[j, 0], const.idx_bounds[j, 1]):
        if (abs(part[0, ii] - x_node) <= 0.5*const.dx):
            f = np.append(f, [part[0:6, ii]], axis=0)
            count += 1
            
    fig = plt.figure(figsize=(12,10))
    fig.suptitle('Particle velocity distribution of species {} in cell {}'.format(j, node_number))
    fig.patch.set_facecolor('w')
    #num_bins = None

    ax_x = plt.subplot2grid((2, 3), (0,0), colspan=2, rowspan=2)
    ax_y = plt.subplot2grid((2, 3), (0,2))
    ax_z = plt.subplot2grid((2, 3), (1,2))

    xs, BinEdgesx = np.histogram((f[:, 3] - const.velocity[j]) / const.va)
    bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
    ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
    ax_x.set_xlabel(r'$v_x / v_A$')
    #ax_x.set_xlim(-2, 2)

    ys, BinEdgesy = np.histogram(f[:, 4] / const.va)
    by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
    ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
    ax_y.set_xlabel(r'$v_y / v_A$')
    #ax_y.set_xlim(-2, 2)

    zs, BinEdgesz = np.histogram(f[:, 5] / const.va)
    bz = 0.5 * (BinEdgesz[1:] + BinEdgesz[:-1])
    ax_z.plot(bz, zs, '-', c='c', drawstyle='steps')
    ax_z.set_xlabel(r'$v_z / v_A$')
    #ax_z.set_xlim(-2, 2)

    plt.show()
    return

def check_position_distribution(part, j):
    '''Checks the spatial distribution of a particle species j within the spatial domain
    '''
    fig = plt.figure(figsize=(12,10))
    fig.suptitle('Particle distribution of species {} in configuration space'.format(j))
    fig.patch.set_facecolor('w')
    num_bins = const.NX

    ax_x = plt.subplot()

    xs, BinEdgesx = np.histogram(part[0, const.idx_bounds[j, 0]: const.idx_bounds[j, 1]] / float(const.dx), bins=num_bins)
    bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
    ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
    ax_x.set_xlabel(r'$x_p$')
    ax_x.set_xlim(0, const.NX)

    plt.show()
    return

def check_velocity_distribution(part, j):
    '''Checks the velocity distribution of an entire species across the simulation domain
    '''
    fig = plt.figure(figsize=(12,10))
    fig.suptitle('Velocity distribution of species {} in simulation domain'.format(j))
    fig.patch.set_facecolor('w')
    num_bins = const.cellpart / 5

    ax_x = plt.subplot2grid((2, 3), (0,0), colspan=2, rowspan=2)
    ax_y = plt.subplot2grid((2, 3), (0,2))
    ax_z = plt.subplot2grid((2, 3), (1,2))

    xs, BinEdgesx = np.histogram(part[3, const.idx_bounds[j, 0]: const.idx_bounds[j, 1]] / const.va, bins=num_bins)
    bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
    ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
    ax_x.set_xlabel(r'$v_x / v_A$')

    ys, BinEdgesy = np.histogram(part[4, const.idx_bounds[j, 0]: const.idx_bounds[j, 1]] / const.va, bins=num_bins)
    by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
    ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
    ax_y.set_xlabel(r'$v_y / v_A$')

    zs, BinEdgesz = np.histogram(part[5, const.idx_bounds[j, 0]: const.idx_bounds[j, 1]] / const.va, bins=num_bins)
    bz = 0.5 * (BinEdgesz[1:] + BinEdgesz[:-1])
    ax_z.plot(bz, zs, '-', c='c', drawstyle='steps')
    ax_z.set_xlabel(r'$v_z / v_A$')

    plt.show()
    return


def test_particle_orbit():
    
    def eval_E(pos):
        coeff = 1 / (100. * np.sqrt(pos[0] ** 2 + pos[1] ** 2) ** 3)
        
        Ex = coeff * pos[0]
        Ey = coeff * pos[1]
        Ez = 0.
        return np.array([Ex, Ey, Ez])

    def eval_B(pos, B0):
        Bx = 0.0
        By = 0.0
        Bz = B0*np.sqrt(pos[0] ** 2 + pos[1] ** 2)
        return np.array([Bx, By, Bz])

    def position_update(pos, vel, DT):
        pos += vel * DT
        return pos
    
    v0    = 1e5
    B0    = 0.01
    vp    = np.array([0., v0, 0.]) 
    rL    = (const.mass[0] * vp[1]) / (abs(const.charge[0] ) * B0)
    xp    = np.array([-rL, 0., 0.]) 

    resolution = 20
    num_rev    = 10
    maxtime    = int(resolution*num_rev)
    gyperiod   = (2 * np.pi * const.mass[0]) / (const.charge[0] * B0) 
    dt         = gyperiod / resolution 

    xy         = np.zeros((maxtime+1, 2))
    print('\nNumber of revolutions: {}'.format(num_rev))
    print('Points per revolution: {}'.format(resolution))
    print('Total number of points: {}'.format(maxtime))
    
    print('\nGyroradius: {}m'.format(rL))
    print('Gyroperiod: {}s'.format(round(gyperiod, 2)))
    print('Timestep: {}s'.format(round(dt, 3)))
    
    Bc  = eval_B(xp, B0)
    Ec  = eval_E(xp)
    vp  = particles.boris_algorithm(vp, Bc, Ec, -0.5*dt, 0)
    for ii in range(maxtime+1):
        xy[ii, 0] = xp[0]
        xy[ii, 1] = xp[1]

        Bc  = eval_B(xp, B0)
        Ec  = eval_E(xp)

        xp = position_update(xp, vp, dt)
        vp = particles.boris_algorithm(vp, Bc, Ec, dt, 0)
        
    plt.plot(xy[:, 0], xy[:, 1])
    plt.axis('equal')
    return

if __name__ == '__main__':
    test_particle_orbit()