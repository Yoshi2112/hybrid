# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:42:13 2017

@author: iarey
"""
import numpy as np
import matplotlib.pyplot as plt
import pdb

import simulation_parameters_1D as const
import particles_1D             as particles
import sources_1D               as sources
import fields_1D                as fields
from matplotlib import animation


def r_squared(data, model):                  
    '''Calculates a simple R^2 value for the fit of model against data. Accepts single dimensional
    arrays only.'''
    SS_tot = np.sum(np.square(data - data.mean()))              # Total      sum of squares
    SS_res = np.sum(np.square(data - model))                    # Residuals  sum of squares
    r_sq   = 1 - (SS_res / SS_tot)                              # R^2 calculation (most general)
    return r_sq


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
    print '\nNumber of revolutions: {}'.format(num_rev)
    print 'Points per revolution: {}'.format(resolution)
    print 'Total number of points: {}'.format(maxtime)
    
    print '\nGyroradius: {}m'.format(rL)
    print 'Gyroperiod: {}s'.format(round(gyperiod, 2))
    print 'Timestep: {}s'.format(round(dt, 3))
    
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


def test_weight_conservation():
    '''
    Plots the normalized weight for a single particle at 1e5 points along simulation
    domain. Should remain at 1 the whole time.
    '''
    nspace        = 100000
    xmax          = const.NX*const.dx
    
    positions     = np.linspace(0, xmax, nspace)
    normal_weight = np.zeros(nspace)
    
    for ii, x in zip(np.arange(nspace), positions):
        left_nodes, weights = particles.assign_weighting_TSC(np.array([x]))
        normal_weight[ii]   = weights.sum()

    plt.plot(positions/const.dx, normal_weight)
    plt.xlim(0., const.NX)
    return


def test_weight_shape():
    plt.ion()
    
    E_nodes = (np.arange(const.NX + 3) - 0.5) #* const.dx
    B_nodes = (np.arange(const.NX + 3) - 1.0) #* const.dx
    dns_test  = np.zeros(const.NX + 3) 
    
    positions = np.array([0.0]) * const.dx
    left_nodes, weights = particles.assign_weighting_TSC(positions)

    for jj in range(3):
        dns_test[left_nodes + jj] = weights[jj]

    plt.plot(E_nodes, dns_test, marker='o')

    for ii in range(const.NX + 3):
        plt.axvline(E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[ii], linestyle='--', c='b', alpha=0.2)
        
    plt.axvline(const.xmin/const.dx, linestyle='-', c='k', alpha=0.2)
    plt.axvline(const.xmax/const.dx, linestyle='-', c='k', alpha=0.2)
    
    plt.xlim(-1.5, const.NX + 2)
    return


def animate_moving_weight():
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(1,1,1)
    x   = np.arange(const.NX + 3)
    
    dt       = 0.1
    vel      = np.array([[0.3 * const.dx / dt],
                         [ 0.],
                         [ 0.]])
    
    position = np.array([0.0]) 

    E_nodes = (np.arange(const.NX + 3) - 0.5) #* const.dx
    B_nodes = (np.arange(const.NX + 3) - 1.0) #* const.dx
    
    for ii in range(150):
        dns_test  = np.zeros(const.NX + 3) 

        pos, left_nodes, weights = particles.position_update(position, vel, dt)
        
        for jj in range(3):
                dns_test[left_nodes + jj] = weights[jj]
                
        y = dns_test

        ax.clear()
        ax.plot(x, y)
        ax.set_xlim(-1.5, const.NX + 2)
        ax.set_ylim(0, 1.5)
        ax.text(1, 1.4, 'Total Weight: {}'.format(dns_test.sum()))
        
        ax.scatter(pos/const.dx, 1.0, c='r')
    
        for kk in range(const.NX + 3):
            ax.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
            ax.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
            
            ax.axvline(const.xmin/const.dx, linestyle='-', c='k', alpha=0.2)
            ax.axvline(const.xmax/const.dx, linestyle='-', c='k', alpha=0.2)
    
        plt.pause(0.05)
    plt.show()


def test_density_and_velocity_deposition():
    E_nodes = (np.arange(const.NX + 3) - 0.5) #* const.dx
    B_nodes = (np.arange(const.NX + 3) - 1.0) #* const.dx
    
    dt       = 0.1
    velocity = np.array([[0.3 * const.dx / dt, 0.0],
                         [ 0., 0.0],
                         [ 0., 0.0]])
    
    position = np.array([16.5, 16.5]) * const.dx
    idx      = np.array([0, 0]) 
    
    left_nodes, weights = particles.assign_weighting_TSC(position)
    n_i, nu_i = sources.deposit_both_moments(position, velocity, left_nodes, weights, idx)

    for jj in range(const.Nj):
        normalized_density = (const.cellpart / const.Nj)*n_i[:, jj] / const.density[jj]
        species_color = const.temp_color[jj]
        plt.plot(E_nodes, normalized_density, marker='o', c=species_color)
        
        print 'Normalized total density contribution of species {} is {}'.format(jj, normalized_density.sum())

    for ii in range(const.NX + 3):
        plt.axvline(E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[ii], linestyle='--', c='b', alpha=0.2)
        
    plt.axvline(const.xmin/const.dx, linestyle='-', c='k', alpha=0.2)
    plt.axvline(const.xmax/const.dx, linestyle='-', c='k', alpha=0.2)
    return


def test_velocity_deposition():
    E_nodes = (np.arange(const.NX + 3) - 0.5) #* const.dx
    B_nodes = (np.arange(const.NX + 3) - 1.0) #* const.dx
    
    dt       = 0.1
    velocity = np.array([[0.3 * const.dx / dt, 0.0],
                         [ 0., 0.0],
                         [ 0., 0.0]])
    
    position = np.array([16.5, 16.5]) * const.dx
    idx      = np.array([0, 0]) 
    
    left_nodes, weights = particles.assign_weighting_TSC(position)
    n_i, nu_i = sources.deposit_velocity_moments(velocity, left_nodes, weights, idx)

    for jj in range(const.Nj):
        normalized_density = (const.cellpart / const.Nj)*n_i[:, jj] / const.density[jj]
        species_color = const.temp_color[jj]
        plt.plot(E_nodes, normalized_density, marker='o', c=species_color)
        
        print 'Normalized total density contribution of species {} is {}'.format(jj, normalized_density.sum())

    for ii in range(const.NX + 3):
        plt.axvline(E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[ii], linestyle='--', c='b', alpha=0.2)
        
    plt.axvline(const.xmin/const.dx, linestyle='-', c='k', alpha=0.2)
    plt.axvline(const.xmax/const.dx, linestyle='-', c='k', alpha=0.2)
    return


def test_init_collect_moments():
   # E_nodes = (np.arange(const.NX + 3) - 0.5) #* const.dx
   # B_nodes = (np.arange(const.NX + 3) - 1.0) #* const.dx
    
    dt       = 0.1
    velocity = np.array([[0.3 * const.dx / dt, 0.0],
                         [ 0., 0.0],
                         [ 0., 0.0]])
    
    position = np.array([16.5, 16.5]) * const.dx
    idx      = np.array([0, 0]) 
    
    left_node, weights = particles.assign_weighting_TSC(position, E_nodes=True)
    
    position, left_node, weights, rho_0, rho, J_plus, J_init, G, L = sources.init_collect_moments(position, velocity, left_node, weights, idx, dt)
    return


def test_force_interpolation():
    E = np.zeros((const.NX + 3, 3))
    B = np.zeros((const.NX + 3, 3))
    
    E_nodes = (np.arange(const.NX + 3) - 0.5) #* const.dx
    B_nodes = (np.arange(const.NX + 3) - 1.0) #* const.dx
    
    B[:, 0] = np.arange(const.NX + 3) * 5e-9            # Linear
    B[:, 1] = np.sin(0.5*np.arange(const.NX + 3) + 5)   # Sinusoidal
    B[:, 2] = 4e-9                                      # Constant
    
    E[:, 0] = np.arange(const.NX + 3) * 1e-5        # Linear
    E[:, 1] = np.sin(0.5*np.arange(const.NX + 3))   # Sinusoidal
    E[:, 2] = 3e-5                                  # Constant
    
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid((3, 3), (0,0), colspan=3)
    ax2 = plt.subplot2grid((3, 3), (1,0), colspan=3)
    ax3 = plt.subplot2grid((3, 3), (2,0), colspan=3)
    #plt.tight_layout(pad=1.0, w_pad=1.8)
    fig.subplots_adjust(hspace=0)
    
    which_field = 'B'
    for ii in np.arange(0, const.NX + 2, 0.5):
        position   = np.array([ii]) * const.dx
        
        Ie, W_elec = particles.assign_weighting_TSC(position, E_nodes=True)
        Ib, W_mag  = particles.assign_weighting_TSC(position, E_nodes=False)
    
        Ep, Bp     = particles.interpolate_forces_to_particle(E, B, Ie[0], W_elec[:, 0], Ib[0], W_mag[:, 0])

        for ax, jj in zip([ax1, ax2, ax3], range(3)):
            ax.clear()
            ax.set_xlim(-1.5, const.NX + 2)
            
            if which_field == 'E':
                ax1.set_title('Electric field interpolation to Particle')
                ax.plot(E_nodes, E[:, jj])
                ax.scatter(ii, Ep[jj])
            elif which_field == 'B':
                ax1.set_title('Magnetic field interpolation to Particle')
                ax.plot(B_nodes, B[:, jj])
                ax.scatter(ii, Bp[jj])
 
            for kk in range(const.NX + 3):
                ax.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
                ax.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
                
                ax.axvline(const.xmin/const.dx, linestyle='-', c='k', alpha=0.2)
                ax.axvline(const.xmax/const.dx, linestyle='-', c='k', alpha=0.2)
    
        plt.pause(0.01)
    
    plt.show()

    return


def test_curl_B():
    NX   = 32     #const.NX

    xmin = 0.0    #const.xmin
    xmax = 2*np.pi#const.xmax
    
    dx   = xmax / NX #const.dx
    x    = np.arange(xmin, xmax, dx/100.)              # Simulation domain space 0,NX (normalized to grid)
    k    = 2.0

    # Physical location of nodes
    E_nodes = (np.arange(NX + 3) - 0.5) * dx
    B_nodes = (np.arange(NX + 3) - 1.0) * dx

    # Set analytic solutions (input/output)
    #test_field =   np.sin(k*x)
    deriv      =   k*np.cos(k*x)                           # Highly sampled output (derivative)
    
    B_input    =   np.sin(k*B_nodes)                     # Analytic input at node points
    By_anal    = - k*np.cos(k*E_nodes)                     # Analytic solution at nodes 
    Bz_anal    =   k*np.cos(k*E_nodes)                                                
    
    # Finite differences
    test_B       = np.zeros((NX + 3, 3))
    test_B[:, 0] = B_input
    test_B[:, 1] = B_input
    test_B[:, 2] = B_input
    curl_B       = fields.get_curl_B(test_B, DX=dx)
    
    plt.figure(figsize=(15, 15))
    marker_size = None
    plt.plot(B_nodes, test_B[:, 2], marker='o', c='g', alpha=0.5, label='Bz test input')
    
    plt.plot(x, -deriv, linestyle=':', c='b', label='By Analytic Solution')
    plt.scatter(E_nodes, By_anal, marker='o', c='k', s=marker_size, label='By Node Solution')
    plt.scatter(E_nodes, curl_B[:, 1], marker='x', c='b', s=marker_size, label='By Finite Difference')
    
    r2y = r_squared(curl_B[1:-2, 1], By_anal[1:-2])
    plt.gcf().text(0.15, 0.95, '$B_y R^2 = %.4f$' % r2y)
    
    r2z = r_squared(curl_B[1:-2, 2], Bz_anal[1:-2])
    plt.gcf().text(0.15, 0.93, '$B_z R^2 = %.4f$' % r2z)
    
    plt.plot(x, deriv, linestyle=':', c='r', label='Bz Analytic Solution')   
    plt.scatter(E_nodes, Bz_anal, marker='o', c='k', s=marker_size, label='Bz Node Solution')
    plt.scatter(E_nodes, curl_B[:, 2], marker='x', c='r', s=marker_size, label='Bz Finite Difference')   
    plt.title(r'Test of $\nabla \times B$')

    for kk in range(NX + 3):
        plt.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
        
        plt.axvline(xmin, linestyle='-', c='k', alpha=0.2)
        plt.axvline(xmax, linestyle='-', c='k', alpha=0.2)
    
    plt.xlim(xmin - 1.5*dx, xmax + 2*dx)
    plt.legend()
    return


def test_curl_E():
    NX   = 32   #const.NX

    xmin = 0.0  #const.xmin
    xmax = 2*np.pi#const.xmax
    
    dx   = xmax / NX
    x    = np.arange(xmin, xmax, dx/100.)              # Simulation domain space 0,NX (normalized to grid)
    k    = 2.0

    # Physical location of nodes
    E_nodes = (np.arange(NX + 3) - 0.5) * dx
    B_nodes = (np.arange(NX + 3) - 1.0) * dx

    # Set analytic solutions (input/output)
    #test_field =   np.sin(k*x)
    deriv      =   k*np.cos(k*x)                           # Highly sampled output (derivative)
    
    E_input    =   np.sin(k*E_nodes)                     # Analytic input at node points
    Ey_anal    = - k*np.cos(k*B_nodes)                     # Analytic solution at nodes 
    Ez_anal    =   k*np.cos(k*B_nodes)                                                
    
    # Finite differences
    test_E       = np.zeros((NX + 3, 3))
    test_E[:, 0] = E_input
    test_E[:, 1] = E_input
    test_E[:, 2] = E_input
    curl_E       = fields.get_curl_E(test_E, DX=dx)
    
    plt.figure(figsize=(15, 15))
    marker_size = None
    plt.plot(B_nodes, test_E[:, 2], marker='o', c='g', alpha=0.5, label='Bz test input')
    
    plt.plot(x, -deriv, linestyle=':', c='b', label='By Analytic Solution')
    plt.scatter(B_nodes, Ey_anal, marker='o', c='k', s=marker_size, label='By Node Solution')
    plt.scatter(B_nodes, curl_E[:, 1], marker='x', c='b', s=marker_size, label='By Finite Difference')
    
    r2y = r_squared(curl_E[1:-2, 1], Ey_anal[1:-2])
    plt.gcf().text(0.15, 0.95, '$B_y R^2 = %.4f$' % r2y)
    
    r2z = r_squared(curl_E[1:-2, 2], Ez_anal[1:-2])
    plt.gcf().text(0.15, 0.93, '$B_z R^2 = %.4f$' % r2z)
    
    plt.plot(x, deriv, linestyle=':', c='r', label='Bz Analytic Solution')   
    plt.scatter(B_nodes, Ez_anal, marker='o', c='k', s=marker_size, label='Bz Node Solution')
    plt.scatter(B_nodes, curl_E[:, 2], marker='x', c='r', s=marker_size, label='Bz Finite Difference')   
    plt.title(r'Test of $\nabla \times E$')

    for kk in range(NX + 3):
        plt.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
        
        plt.axvline(xmin, linestyle='-', c='k', alpha=0.2)
        plt.axvline(xmax, linestyle='-', c='k', alpha=0.2)
    
    plt.xlim(xmin - 1.5*dx, xmax + 2*dx)
    plt.legend()
    return


def test_grad_P():
    NX   = 32      #const.NX
    xmin = 0.0     #const.xmin
    xmax = 2*np.pi #const.xmax
    
    dx   = xmax / NX
    x    = np.arange(xmin, xmax, dx/100.)             # Simulation domain space 0,NX (normalized to grid)
    k    = 2.0
    
    # Physical location of nodes
    E_nodes = (np.arange(NX + 3) - 0.5) * dx
    B_nodes = (np.arange(NX + 3) - 1.0) * dx

    # Set analytic solutions (input/output)
    qn_input   = const.q*(np.sin(k*E_nodes) + 1)                    # Analytic input at node points (number density varying)
    pe_anal    = const.kB * const.Te0 * k*np.cos(k*B_nodes)         # Analytic solution at nodes 
    grad_P_anal= const.kB * const.Te0 * k*np.cos(k*x)               # Highly sampled output (derivative)

    # Finite differences
    grad_PB, grad_P     = fields.get_grad_P(qn_input, np.ones(NX + 3)*const.Te0, DX=dx)
    
    r2 = r_squared(grad_P[1:-2], pe_anal[1:-2])
    
    ## PLOT ##
    plt.figure(figsize=(15, 15))
    marker_size = None

    plt.plot(x, grad_P_anal, linestyle=':', c='b', label='Analytic Solution')
    plt.scatter(B_nodes, pe_anal, marker='o', c='k', s=marker_size, label='Node Solution')
    plt.scatter(B_nodes, grad_PB, marker='x', c='b', s=marker_size, label='Finite Difference (on B)')
    plt.scatter(E_nodes, grad_P, marker='x', c='r', s=marker_size, label='Finite Difference (on E)')
    
    plt.title(r'Test of $\nabla p_e$')

    for kk in range(NX + 3):
        plt.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
        
        plt.axvline(xmin, linestyle='-', c='k', alpha=0.2)
        plt.axvline(xmax, linestyle='-', c='k', alpha=0.2)
    
    plt.gcf().text(0.15, 0.93, '$R^2 = %.4f$' % r2)
    plt.xlim(xmin - 1.5*dx, xmax + 2*dx)
    plt.legend()
    return

if __name__ == '__main__':
    #animate_moving_weight()
    #test_curl_B()
    test_grad_P()