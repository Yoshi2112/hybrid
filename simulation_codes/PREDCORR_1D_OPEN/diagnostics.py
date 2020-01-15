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
import auxilliary_1D            as aux
from matplotlib import animation


def r_squared(data, model):                  
    '''Calculates a simple R^2 value for the fit of model against data. Accepts single dimensional
    arrays only.'''
    SS_tot = np.sum(np.square(data - data.mean()))              # Total      sum of squares
    SS_res = np.sum(np.square(data - model))                    # Residuals  sum of squares
    r_sq   = 1 - (SS_res / SS_tot)                              # R^2 calculation (most general)
    return r_sq


def check_cell_velocity_distribution(pos, vel, node_number=const.NX // 2, j=0): #
    '''Checks the velocity distribution of a particle species within a specified cell
    '''
    x_node = (node_number - 0.5) * const.dx   # Position of E-field node
    f = np.zeros((1, 3))
    count = 0

    for ii in np.arange(const.idx_bounds[j, 0], const.idx_bounds[j, 1]):
        if (abs(pos[ii] - x_node) <= 0.5*const.dx):
            f = np.append(f, [vel[0:3, ii]], axis=0)
            count += 1
            
    fig = plt.figure(figsize=(12,10))
    fig.suptitle('Particle velocity distribution of species {} in cell {}'.format(j, node_number))
    fig.patch.set_facecolor('w')
    num_bins = const.nsp_ppc // 20

    ax_x = plt.subplot2grid((2, 3), (0,0), colspan=2, rowspan=2)
    ax_y = plt.subplot2grid((2, 3), (0,2))
    ax_z = plt.subplot2grid((2, 3), (1,2))

    xs, BinEdgesx = np.histogram((f[:, 0] - const.drift_v[j]) / const.va, bins=num_bins)
    bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
    ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
    ax_x.set_xlabel(r'$v_x / v_A$')
    #ax_x.set_xlim(-2, 2)

    ys, BinEdgesy = np.histogram(f[:, 1] / const.va, bins=num_bins)
    by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
    ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
    ax_y.set_xlabel(r'$v_y / v_A$')
    #ax_y.set_xlim(-2, 2)

    zs, BinEdgesz = np.histogram(f[:, 2] / const.va, bins=num_bins)
    bz = 0.5 * (BinEdgesz[1:] + BinEdgesz[:-1])
    ax_z.plot(bz, zs, '-', c='c', drawstyle='steps')
    ax_z.set_xlabel(r'$v_z / v_A$')
    #ax_z.set_xlim(-2, 2)

    plt.show()
    return

def check_position_distribution(pos):
    '''Checks the spatial distribution of a particle species j within the spatial domain
    '''
    for j in range(const.Nj):
        fig = plt.figure(figsize=(12,10))
        fig.suptitle('Configuration space distribution of {}'.format(const.species_lbl[j]))
        fig.patch.set_facecolor('w')
        num_bins = const.NX
    
        ax_x = plt.subplot()
    
        xs, BinEdgesx = np.histogram(pos[const.idx_bounds[j, 0]: const.idx_bounds[j, 1]] / float(const.dx), bins=num_bins)
        bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
        ax_x.plot(bx, xs, '-', c=const.temp_color[const.temp_type[j]], drawstyle='steps')
        ax_x.set_xlabel(r'$x_p$')
        ax_x.set_xlim(0, const.NX)

    plt.show()
    return

def check_velocity_distribution(vel):
    '''Checks the velocity distribution of an entire species across the simulation domain
    '''
    for j in range(const.Nj):
        fig = plt.figure(figsize=(12,10))
        fig.suptitle('Velocity distribution of species {} in simulation domain'.format(j))
        fig.patch.set_facecolor('w')
        num_bins = const.nsp_ppc // 5
    
        ax_x = plt.subplot2grid((2, 3), (0,0), colspan=2, rowspan=2)
        ax_y = plt.subplot2grid((2, 3), (0,2))
        ax_z = plt.subplot2grid((2, 3), (1,2))
    
        xs, BinEdgesx = np.histogram(vel[0, const.idx_bounds[j, 0]: const.idx_bounds[j, 1]] / const.va, bins=num_bins)
        bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
        ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
        ax_x.set_xlabel(r'$v_x / v_A$')
    
        ys, BinEdgesy = np.histogram(vel[1, const.idx_bounds[j, 0]: const.idx_bounds[j, 1]] / const.va, bins=num_bins)
        by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
        ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
        ax_y.set_xlabel(r'$v_y / v_A$')
    
        zs, BinEdgesz = np.histogram(vel[2, const.idx_bounds[j, 0]: const.idx_bounds[j, 1]] / const.va, bins=num_bins)
        bz = 0.5 * (BinEdgesz[1:] + BinEdgesz[:-1])
        ax_z.plot(bz, zs, '-', c='c', drawstyle='steps')
        ax_z.set_xlabel(r'$v_z / v_A$')

    plt.show()
    return


def test_particle_orbit():
    def position_update(pos, vel, DT):
        for ii in range(3):
            pos[ii] += vel[ii, 0] * dt
        return pos
    
    NX    = 64
    B0    = 0.01
    v0    = 1e5
    mi    = 1.672622e-27                          # Mass of proton (kg)
    qi    = 1.602177e-19                          # Elementary charge (C)
    
    resolution = 10
    num_rev    = 5000
    maxtime    = int(resolution*num_rev)
    gyperiod   = (2 * np.pi * mi) / (qi * B0) 
    dt         = gyperiod / resolution 
    
    B     = np.zeros((NX + 3, 3), dtype=np.float64)
    E     = np.zeros((NX + 3, 3), dtype=np.float64)
    B[:, 0] +=  B0
    
    Ie    = np.array([NX // 2])   ; Ib    = np.array([NX // 2])
    We    = np.array([0., 1., 0.]).reshape((3, 1))
    Wb    = np.array([0., 1., 0.]).reshape((3, 1))
    idx   = np.array([0])
    
    vp    = np.array([[0.], [v0], [0.]])
    rL    = (mi * vp[1][0]) / (qi * B0)
    
    xp    = np.array([-rL, 0., 0.]) 
    
    print('\nNumber of revolutions: {}'.format(num_rev))
    print('Points per revolution: {}'.format(resolution))
    print('Total number of points: {}'.format(maxtime))
    
    print('\nGyroradius: {}m'.format(rL))
    print('Gyroperiod: {}s'.format(round(gyperiod, 2)))
    print('Timestep: {}s'.format(round(dt, 3)))
    
    particles.velocity_update(vp, Ie, We, Ib, Wb, idx, B, E, -0.5*dt)
    
    xy    = np.zeros((maxtime+1, 2))
    
    for ii in range(maxtime+1):
        xy[ii, 0] = xp[1]
        xy[ii, 1] = xp[2]

        position_update(xp, vp, dt)
        particles.velocity_update(vp, Ie, We, Ib, Wb, idx, B, E, dt)
    print(xp)
    print(vp)  
    #plt.plot(xy[:, 0], xy[:, 1])
    #plt.axis('equal')
    return


def test_weight_conservation():
    '''
    Plots the normalized weight for a single particle at 1e5 points along simulation
    domain. Should remain at 1 the whole time.
    '''
    nspace        = 10000
    xmax          = const.NX*const.dx
    
    positions     = np.linspace(0, xmax, nspace, endpoint=True)
    normal_weight = np.zeros(positions.shape[0])
    left_nodes    = np.zeros(positions.shape[0], dtype=int)
    weights       = np.zeros((3, positions.shape[0]))
    
    particles.assign_weighting_TSC(positions, left_nodes, weights, E_nodes=False)
    
    for ii in range(positions.shape[0]):
        normal_weight[ii] = weights[:, ii].sum()

    #plt.plot(positions, normal_weight)
    #plt.xlim(0., xmax)
    return


def test_weight_shape():
    plt.ion()
    
    XMIN = 0
    XMAX = const.NX * const.dx
    
    #positions     = np.array([80, 205, 340, 360])
    positions     = np.array([350])
    
    E_nodes  = (np.arange(const.NX + 2*const.ND) - const.ND  + 0.5) * const.dx
    B_nodes  = (np.arange(const.NX + 2*const.ND) - const.ND  - 0.0) * const.dx
    dns_test =  np.zeros( const.NX + 2*const.ND) 
    
    left_nodes    = np.zeros(positions.shape[0], dtype=int)
    weights       = np.zeros((3, positions.shape[0]))
    
    particles.assign_weighting_TSC(positions, left_nodes, weights, E_nodes=True)
    
    plt.figure()
    for ii in range(positions.shape[0]):
        plt.axvline(positions[ii], linestyle='-', c='k', alpha=0.2)
        for jj in range(3):
            xx = left_nodes[ii] + jj
            
            plt.axvline(positions[ii], linestyle='-', c='k', alpha=0.2)
            plt.axvline(E_nodes[  xx], linestyle='-', c='r', alpha=1.0)
            
            dns_test[xx] = weights[jj, ii]

    plt.plot(E_nodes, dns_test, marker='o')

    for ii in range(E_nodes.shape[0]):
        plt.axvline(E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[ii], linestyle='--', c='b', alpha=0.2)
    
    plt.axvline(XMIN, linestyle=':', c='k', alpha=0.5)
    plt.axvline(XMAX, linestyle=':', c='k', alpha=0.5)
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
        
        print('Normalized total density contribution of species {} is {}'.format(jj, normalized_density.sum()))

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
        
        print('Normalized total density contribution of species {} is {}'.format(jj, normalized_density.sum()))

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

        for ax, jj in zip([ax1, ax2, ax3], list(range(3))):
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
    NX   = 50     
    ND   = 10
    NC   = NX + 2*ND
    xmin = 0.0    
    xmax = 2*np.pi
    
    dx   = xmax / NX
    k    = 2.0

    # Physical location of nodes
    E_nodes  = (np.arange(NC) - ND  + 0.5) * dx
    B_nodes  = (np.arange(NC) - ND  - 0.0) * dx
    
    # Inputs and analytic solutions
    B_input       = np.zeros((NC, 3))
    B_input[:, 0] = np.cos(1.0*k*B_nodes)
    B_input[:, 1] = np.cos(1.5*k*B_nodes)
    B_input[:, 2] = np.cos(2.0*k*B_nodes)
    
    curl_B_anal       = np.zeros((NC, 3))
    curl_B_anal[:, 1] = -1.5 * k * np.sin(1.5*k*E_nodes) / dx
    curl_B_anal[:, 2] = -2.0 * k * np.sin(2.0*k*E_nodes) / dx
    
    curl_B_FD = np.zeros((NC, 3))
    fields.curl_B_term(B_input, curl_B_FD, DX=dx)
    
    ## DO THE PLOTTING ##
    plt.figure(figsize=(15, 15))
    marker_size = None
    
    #plt.plot(x, -deriv, linestyle=':', c='b', label='By Analytic Solution')
    #plt.scatter(E_nodes, curl_B_anal[:, 1], marker='o', c='k', s=marker_size, label='By Node Solution')
    plt.scatter(E_nodes, curl_B_FD[:, 1], marker='x', c='b', s=marker_size, label='By Finite Difference')
    
    #plt.plot(x, deriv, linestyle=':', c='r', label='Bz Analytic Solution')   
    #plt.scatter(E_nodes, curl_B_anal[:, 2], marker='o', c='k', s=marker_size, label='Bz Node Solution')
    plt.scatter(E_nodes, curl_B_FD[:, 2], marker='x', c='r', s=marker_size, label='Bz Finite Difference')   
    plt.title(r'Test of $\nabla \times B$')

    for kk in range(NC):
        plt.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
        
        plt.axvline(xmin, linestyle='-', c='k', alpha=0.2)
        plt.axvline(xmax, linestyle='-', c='k', alpha=0.2)
    
    plt.xlim(xmin - 1.5*dx, xmax + 2*dx)
    plt.legend()
    return


def test_curl_E():
    NX   = 50     
    ND   = 10
    NC   = NX + 2*ND
    xmin = 0.0    
    xmax = 2*np.pi
    
    dx   = xmax / NX
    k    = 2.0

    # Physical location of nodes
    E_nodes  = (np.arange(NC) - ND  + 0.5) * dx
    B_nodes  = (np.arange(NC) - ND  - 0.0) * dx
    
    E_input       = np.zeros((NC, 3))
    E_input[:, 0]         =            np.cos(1.0*k*E_nodes)
    E_input[:, 1]         =            np.cos(1.5*k*E_nodes)
    E_input[:, 2]         =            np.cos(2.0*k*E_nodes)

    curl_E_FD = np.zeros((NC, 3))
    fields.get_curl_E(E_input, curl_E_FD, DX=1)
    
    curl_E_anal       = np.zeros((NC, 3))
    curl_E_anal[:, 1] =  2.0 * k * np.sin(2.0*k*B_nodes)
    curl_E_anal[:, 2] = -1.5 * k * np.sin(1.5*k*B_nodes)
    
    
    
    ## PLOT
    plt.figure(figsize=(15, 15))
    marker_size = None

    #plt.plot(x, -deriv, linestyle=':', c='b', label='By Analytic Solution')
    #plt.scatter(B_nodes, curl_E_anal[:, 1], marker='o', c='k', s=marker_size, label='By Node Solution')
    plt.scatter(B_nodes, curl_E_FD[:, 1], marker='x', c='b', s=marker_size, label='By Finite Difference')
    
    #plt.plot(x, deriv, linestyle=':', c='r', label='Bz Analytic Solution')   
    #plt.scatter(B_nodes, curl_E_anal[:, 2], marker='o', c='k', s=marker_size, label='Bz Node Solution')
    plt.scatter(B_nodes, curl_E_FD[:, 2], marker='x', c='r', s=marker_size, label='Bz Finite Difference')   
    plt.title(r'Test of $\nabla \times E$')

    for kk in range(NC):
        plt.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
        
        plt.axvline(xmin, linestyle='-', c='k', alpha=0.2)
        plt.axvline(xmax, linestyle='-', c='k', alpha=0.2)
    
    plt.xlim(xmin - 1.5*dx, xmax + 2*dx)
    plt.legend()
    return


def test_grad_P_varying_qn():
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
    te_input   = np.ones(NX + 3)*const.Te0
    
    pe_anal    = const.kB * const.Te0 * k*np.cos(k*B_nodes)         # Analytic solution at nodes 
    grad_P_anal= const.kB * const.Te0 * k*np.cos(k*x)               # Highly sampled output (derivative)

    # Finite differences
    grad_PB, grad_P     = fields.get_grad_P(qn_input, te_input, DX=dx)
    
    r2 = r_squared(grad_P[1:-2], pe_anal[1:-2])
    
    ## PLOT ##
    plot = False
    if plot == True:
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


def test_cross_product():
    npts = 100
    x    = np.linspace(0, 1, npts)
    
    A = np.ones((npts, 3))
    B = np.ones((npts, 3))
    anal_result = np.ones((npts, 3))
    
    s1x = np.sin(2 * np.pi *   x)
    s2x = np.sin(2 * np.pi * 2*x)
    s3x = np.sin(2 * np.pi * 3*x)
    
    c1x = np.cos(2 * np.pi *   x)
    c2x = np.cos(2 * np.pi * 2*x)
    c3x = np.cos(2 * np.pi * 3*x)
    
    A[:, 0] = s1x ; B[:, 0] = c1x
    A[:, 1] = s2x ; B[:, 1] = c2x
    A[:, 2] = s3x ; B[:, 2] = c3x
    
    anal_result[:, 0] =   s2x*c3x - c2x*s3x
    anal_result[:, 1] = -(s1x*c3x - c1x*s3x)
    anal_result[:, 2] =   s2x*c3x - c2x*s3x

    test_result = aux.cross_product(A, B)
    diff        = test_result - anal_result
    print(diff)

    return



def test_E_convective():
    '''
    Tests E-field update, convective (JxB) term only by zeroing/unity-ing other terms.
    
    B-field is kept uniform in order to ensure curl(B) = 0
    '''
    xmin = 0.0     #const.xmin
    xmax = 2*np.pi #const.xmax
    k    = 1.0
    
    NX   = 32
    dx   = xmax / NX
    
    # Physical location of nodes
    E_nodes = (np.arange(NX + 3) - 0.5) * dx
    B_nodes = (np.arange(NX + 3) - 1.0) * dx
    
    qn_input   = np.ones(NX + 3)                                # Analytic input at node points (number density varying)
    B_input    = np.ones((NX + 3, 3))
    
    J_input       = np.zeros((NX + 3, 3))
    J_input[:, 0] = np.sin(1.0*k*E_nodes)
    J_input[:, 1] = np.sin(2.0*k*E_nodes)
    J_input[:, 2] = np.sin(3.0*k*E_nodes)
    
    E_FD = fields.calculate_E(B_input, J_input, qn_input, DX=dx)
    
    E_anal       = np.zeros((NX + 3, 3))
    E_anal[:, 0] = np.sin(3.0*k*E_nodes) - np.sin(2.0*k*E_nodes)
    E_anal[:, 1] = np.sin(1.0*k*E_nodes) - np.sin(3.0*k*E_nodes)
    E_anal[:, 2] = np.sin(2.0*k*E_nodes) - np.sin(1.0*k*E_nodes)
    
    plot = True
    if plot == True:
        marker_size = 20
        plt.scatter(E_nodes[1:-2], E_anal[1:-2, 0], s=marker_size, c='k')
        plt.scatter(E_nodes[1:-2], E_anal[1:-2, 1], s=marker_size, c='k')
        plt.scatter(E_nodes[1:-2], E_anal[1:-2, 2], s=marker_size, c='k')
    
        plt.scatter(E_nodes, E_FD[:, 0], s=marker_size, marker='x')
        plt.scatter(E_nodes, E_FD[:, 1], s=marker_size, marker='x')
        plt.scatter(E_nodes, E_FD[:, 2], s=marker_size, marker='x')
        
        for kk in range(NX + 3):
            plt.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
            plt.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
            
            plt.axvline(xmin, linestyle='-', c='k', alpha=0.2)
            plt.axvline(xmax, linestyle='-', c='k', alpha=0.2)
        
        #plt.gcf().text(0.15, 0.93, '$R^2 = %.4f$' % r2)
        plt.xlim(xmin - 1.5*dx, xmax + 2*dx)
        plt.legend()
    return


def test_E_convective_exelectron():
    '''
    Tests E-field update, convective (JxB) term only by zeroing/unity-ing other terms.
    
    B-field is kept uniform in order to ensure curl(B) = 0
    '''
    xmin = 0.0     #const.xmin
    xmax = 2*np.pi #const.xmax
    k    = 1.0
    
    NX   = 32
    dx   = xmax / NX
    
    # Physical location of nodes
    E_nodes = (np.arange(NX + 3) - 0.5) * dx
    B_nodes = (np.arange(NX + 3) - 1.0) * dx
    
    qn_input   = np.ones(NX + 3)                                # Analytic input at node points (number density varying)
    B_input    = np.ones((NX + 3, 3))
    
    J_input       = np.zeros((NX + 3, 3))
    J_input[:, 0] = np.sin(1.0*k*E_nodes)
    J_input[:, 1] = np.sin(2.0*k*E_nodes)
    J_input[:, 2] = np.sin(3.0*k*E_nodes)
    
    E_FD  = fields.calculate_E(       B_input, J_input, qn_input, DX=dx)
    E_FD2 = fields.calculate_E_w_exel(B_input, J_input, qn_input, DX=dx)
    
    E_anal       = np.zeros((NX + 3, 3))
    E_anal[:, 0] = np.sin(3.0*k*E_nodes) - np.sin(2.0*k*E_nodes)
    E_anal[:, 1] = np.sin(1.0*k*E_nodes) - np.sin(3.0*k*E_nodes)
    E_anal[:, 2] = np.sin(2.0*k*E_nodes) - np.sin(1.0*k*E_nodes)
    
    plot = True
    if plot == True:
        marker_size = 50
        plt.scatter(E_nodes[1:-2], E_anal[1:-2, 0], s=marker_size, c='k')
        plt.scatter(E_nodes[1:-2], E_anal[1:-2, 1], s=marker_size, c='k')
        plt.scatter(E_nodes[1:-2], E_anal[1:-2, 2], s=marker_size, c='k')
    
        plt.scatter(E_nodes, E_FD[:, 0],  s=marker_size, marker='x', c='b')
        plt.scatter(E_nodes, E_FD[:, 1],  s=marker_size, marker='x', c='b')
        plt.scatter(E_nodes, E_FD[:, 2],  s=marker_size, marker='x', c='b')

        plt.scatter(E_nodes, E_FD2[:, 0], s=marker_size, marker='+', c='r')
        plt.scatter(E_nodes, E_FD2[:, 1], s=marker_size, marker='+', c='r')
        plt.scatter(E_nodes, E_FD2[:, 2], s=marker_size, marker='+', c='r')
        
        for kk in range(NX + 3):
            plt.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
            plt.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
            
            plt.axvline(xmin, linestyle='-', c='k', alpha=0.2)
            plt.axvline(xmax, linestyle='-', c='k', alpha=0.2)
        
        #plt.gcf().text(0.15, 0.93, '$R^2 = %.4f$' % r2)
        plt.xlim(xmin - 1.5*dx, xmax + 2*dx)
        plt.legend()
    return



def test_E_hall():
    '''
    Tests E-field update, hall term (B x curl(B)) only by selection of inputs
    '''
# =============================================================================
#     grids = [32, 64, 128, 256, 512, 1024, 2048]
#     err_curl   = np.zeros(len(grids))
#     err_interp = np.zeros(len(grids))
#     err_hall   = np.zeros(len(grids))
# =============================================================================
    
    #for NX, ii in zip(grids, range(len(grids))):
    NX   = 32      #const.NX
    xmin = 0.0     #const.xmin
    xmax = 2*np.pi #const.xmax
    
    dx   = xmax / NX
    k    = 1.0
    marker_size = 70
    
    E_nodes = (np.arange(NX + 3) - 0.5) * dx                    # Physical location of nodes
    B_nodes = (np.arange(NX + 3) - 1.0) * dx
    
    ## INPUTS ##
    Bx         =           np.sin(1.0*k*B_nodes)
    By         =           np.sin(2.0*k*B_nodes)
    Bz         =           np.sin(3.0*k*B_nodes)
    
    Bxe        =           np.sin(1.0*k*E_nodes)
    Bye        =           np.sin(2.0*k*E_nodes)
    Bze        =           np.sin(3.0*k*E_nodes)
    
    dBy        = 2.0 * k * np.cos(2.0*k*E_nodes)
    dBz        = 3.0 * k * np.cos(3.0*k*E_nodes)
    
    B_input       = np.zeros((NX + 3, 3))
    B_input[:, 0] = Bx
    B_input[:, 1] = By
    B_input[:, 2] = Bz

    Be_input       = np.zeros((NX + 3, 3))
    Be_input[:, 0] = Bxe
    Be_input[:, 1] = Bye
    Be_input[:, 2] = Bze
    B_center       = aux.interpolate_to_center_cspline3D(B_input, DX=dx)
    
    
    ## TEST CURL B (AGAIN JUST TO BE SURE)
    curl_B_FD   = fields.get_curl_B(B_input, DX=dx)
    curl_B_anal = np.zeros((NX + 3, 3))
    curl_B_anal[:, 1] = -dBz
    curl_B_anal[:, 2] =  dBy


    ## ELECTRIC FIELD CALCULATION ## 
    E_FD         =   fields.calculate_E(       B_input, np.zeros((NX + 3, 3)), np.ones(NX + 3), DX=dx)
    E_FD2        =   fields.calculate_E_w_exel(B_input, np.zeros((NX + 3, 3)), np.ones(NX + 3), DX=dx)
    
    E_anal       = np.zeros((NX + 3, 3))
    E_anal[:, 0] = - (Bye * dBy + Bze * dBz)
    E_anal[:, 1] = Bxe * dBy
    E_anal[:, 2] = Bxe * dBz
    E_anal      /= const.mu0
        

# =============================================================================
#         ## Calculate errors ##
#         err_curl[ii]   = np.abs(curl_B_FD[1: -2, :] - curl_B_anal[1: -2, :]).max()
#         err_interp[ii] = np.abs(B_center[1: -2, :]  - Be_input[1: -2, :]   ).max()
#         err_hall[ii]   = np.abs(E_FD[1: -2, :]  - E_anal[1: -2, :]   ).max()
#     
#     for ii in range(len(grids) - 1):
#         order_curl   = np.log(err_curl[ii]   / err_curl[ii + 1]  ) / np.log(2)
#         order_interp = np.log(err_interp[ii] / err_interp[ii + 1]) / np.log(2)
#         order_hall   = np.log(err_hall[ii]   / err_hall[ii + 1]  ) / np.log(2)
#         
#         print ''
#         print 'Grid reduction: {} -> {}'.format(grids[ii], grids[ii + 1])
#         print 'Curl order: {}, \nC-spline Interpolation order: {}'.format(order_curl, order_interp)
#         print 'E-field Hall order: {}'.format(order_hall)
# =============================================================================
        
        
    plot = True
    if plot == True:
# =============================================================================
#         # Plot curl test
#         plt.figure()
#         plt.scatter(E_nodes, curl_B_anal[:, 1], s=marker_size, c='k')
#         plt.scatter(E_nodes, curl_B_anal[:, 2], s=marker_size, c='k')
#         plt.scatter(E_nodes, curl_B_FD[:, 1], s=marker_size, marker='x')
#         plt.scatter(E_nodes, curl_B_FD[:, 2], s=marker_size, marker='x')
#         
#         # Plot center-interpolated B test
#         plt.figure()
#         plt.scatter(B_nodes, B_input[:, 0], s=marker_size, c='b')
#         plt.scatter(B_nodes, B_input[:, 1], s=marker_size, c='b')
#         plt.scatter(B_nodes, B_input[:, 2], s=marker_size, c='b')
#         plt.scatter(E_nodes, B_center[:, 0], s=marker_size, c='r')
#         plt.scatter(E_nodes, B_center[:, 1], s=marker_size, c='r')
#         plt.scatter(E_nodes, B_center[:, 2], s=marker_size, c='r')
# =============================================================================
            
        # Plot E-field test solutions
        plt.scatter(E_nodes, E_anal[:, 0], s=marker_size, marker='o', c='k')
        plt.scatter(E_nodes, E_anal[:, 1], s=marker_size, marker='o', c='k')
        plt.scatter(E_nodes, E_anal[:, 2], s=marker_size, marker='o', c='k')
        
        plt.scatter(E_nodes, E_FD[:, 0],  s=marker_size, c='b', marker='+')
        plt.scatter(E_nodes, E_FD[:, 1],  s=marker_size, c='b', marker='+')
        plt.scatter(E_nodes, E_FD[:, 2],  s=marker_size, c='b', marker='+')
        
        plt.scatter(E_nodes, E_FD2[:, 0], s=marker_size, c='r', marker='x')
        plt.scatter(E_nodes, E_FD2[:, 1], s=marker_size, c='r', marker='x')
        plt.scatter(E_nodes, E_FD2[:, 2], s=marker_size, c='r', marker='x')
        
        for kk in range(NX + 3):
            plt.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
            plt.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
            
            plt.axvline(xmin, linestyle='-', c='k', alpha=0.2)
            plt.axvline(xmax, linestyle='-', c='k', alpha=0.2)
        
        plt.xlim(xmin - 1.5*dx, xmax + 2*dx)
    return



def test_cspline_interpolation():
    '''
    Tests E-field update, hall term (B x curl(B)) only by selection of inputs
    '''
    grids = [8, 16, 32]
    errors = np.zeros(len(grids))
    
    for NX, ii in zip(grids, list(range(len(grids)))):
        #NX   = 32      #const.NX
        xmin = 0.0     #const.xmin
        xmax = 2*np.pi #const.xmax
        
        dx   = xmax / NX
        x    = np.arange(xmin - 1.5*dx, xmax + 2*dx, dx/100.)
        k    = 1.0
        marker_size = 20
        
        # Physical location of nodes
        E_nodes = (np.arange(NX + 3) - 0.5) * dx
        B_nodes = (np.arange(NX + 3) - 1.0) * dx
    
        Bxc        = np.cos(1.0*k*x)
        Byc        = np.cos(2.0*k*x)
        Bzc        = np.cos(3.0*k*x)
        
        Bx         = np.cos(1.0*k*B_nodes)
        By         = np.cos(2.0*k*B_nodes)
        Bz         = np.cos(3.0*k*B_nodes)
        
        Bxe        = np.cos(1.0*k*E_nodes)
        Bye        = np.cos(2.0*k*E_nodes)
        Bze        = np.cos(3.0*k*E_nodes)  
        
        B_input       = np.zeros((NX + 3, 3))
        B_input[:, 0] = Bx
        B_input[:, 1] = By
        B_input[:, 2] = Bz
    
        ## TEST INTERPOLATION ##
        B_center = aux.interpolate_to_center_cspline3D(B_input, DX=dx)
    
        error_x    = abs(B_center[:, 0] - Bxe).max()
        error_y    = abs(B_center[:, 1] - Bye).max()
        error_z    = abs(B_center[:, 2] - Bze).max()
        
        errors[ii] = np.max([error_x, error_y, error_z])
        
    for ii in range(len(grids) - 1):
        order = np.log(errors[ii] / errors[ii + 1]) / np.log(2)
        print(order)

    plot = True
    if plot == True:
        plt.figure()
        plt.scatter(B_nodes, B_input[:, 0], s=marker_size, c='b', marker='x')
        plt.scatter(B_nodes, B_input[:, 1], s=marker_size, c='b', marker='x')
        plt.scatter(B_nodes, B_input[:, 2], s=marker_size, c='b', marker='x')
        
        plt.scatter(E_nodes, Bxe, s=marker_size, c='k', marker='o')
        plt.scatter(E_nodes, Bye, s=marker_size, c='k', marker='o')
        plt.scatter(E_nodes, Bze, s=marker_size, c='k', marker='o')
        
        plt.scatter(E_nodes, B_center[:, 0], s=marker_size, c='r', marker='x')
        plt.scatter(E_nodes, B_center[:, 1], s=marker_size, c='r', marker='x')
        plt.scatter(E_nodes, B_center[:, 2], s=marker_size, c='r', marker='x')
        
        plt.plot(x, Bxc, linestyle=':', c='k')
        plt.plot(x, Byc, linestyle=':', c='k')
        plt.plot(x, Bzc, linestyle=':', c='k')
        
        for kk in range(NX + 3):
            plt.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
            plt.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
            
        plt.axvline(xmin, linestyle='-', c='k', alpha=0.2)
        plt.axvline(xmax, linestyle='-', c='k', alpha=0.2)
    
        plt.xlim(xmin - 1.5*dx, xmax + 2*dx)
        plt.legend()
    return


def test_interp_cross_manual():
    '''
    Test order of cross product with interpolation, separate from hall term calculation (double check)
    '''
    grids  = [16, 32, 64, 128, 256, 512, 1024]
    errors = np.zeros(len(grids))
    
    #NX   = 32      #const.NX
    xmin = 0.0     #const.xmin
    xmax = 2*np.pi #const.xmax
    k    = 1.0
    marker_size = 20
    
    for NX, ii in zip(grids, list(range(len(grids)))):
        dx   = xmax / NX
        #x    = np.arange(xmin, xmax, dx/100.)

        # Physical location of nodes
        E_nodes = (np.arange(NX + 3) - 0.5) * dx
        B_nodes = (np.arange(NX + 3) - 1.0) * dx
        
        ## TEST INPUT FIELDS ##
        A  = np.ones((NX + 3, 3))
        Ax = np.sin(1.0*E_nodes*k)
        Ay = np.sin(2.0*E_nodes*k)
        Az = np.sin(3.0*E_nodes*k)
        
        B  = np.ones((NX + 3, 3))
        Bx = np.cos(1.0*B_nodes*k)
        By = np.cos(2.0*B_nodes*k)
        Bz = np.cos(3.0*B_nodes*k)
        
        Be  = np.ones((NX + 3, 3))
        Bxe = np.cos(1.0*E_nodes*k)
        Bye = np.cos(2.0*E_nodes*k)
        Bze = np.cos(3.0*E_nodes*k)
        
        A[:, 0] = Ax ; B[:, 0] = Bx ; Be[:, 0] = Bxe
        A[:, 1] = Ay ; B[:, 1] = By ; Be[:, 1] = Bye
        A[:, 2] = Az ; B[:, 2] = Bz ; Be[:, 2] = Bze
        
        B_inter      = aux.interpolate_to_center_cspline3D(B)
        
        ## RESULTS (AxB) ##
        anal_result       = np.ones((NX + 3, 3))
        anal_result[:, 0] = Ay*Bze - Az*Bye
        anal_result[:, 1] = Az*Bxe - Ax*Bze
        anal_result[:, 2] = Ax*Bye - Ay*Bxe
        
        test_result  = aux.cross_product(A, Be)
        inter_result = aux.cross_product(A, B_inter)

        error_x    = abs(anal_result[:, 0] - inter_result[:, 0]).max()
        error_y    = abs(anal_result[:, 1] - inter_result[:, 1]).max()
        error_z    = abs(anal_result[:, 2] - inter_result[:, 2]).max()
        
        errors[ii] = np.max([error_x, error_y, error_z])
        
    for ii in range(len(grids) - 1):
        order = np.log(errors[ii] / errors[ii + 1]) / np.log(2)
        print(order)


    # OUTPUTS (Plots the highest grid-resolution version)
    plot = False
    if plot == True:
        plt.figure()
        plt.scatter(E_nodes, anal_result[:, 0], s=marker_size, c='k', marker='o')
        plt.scatter(E_nodes, anal_result[:, 1], s=marker_size, c='k', marker='o')
        plt.scatter(E_nodes, anal_result[:, 2], s=marker_size, c='k', marker='o')
    
        plt.scatter(E_nodes, test_result[:, 0], s=marker_size, c='r', marker='x')
        plt.scatter(E_nodes, test_result[:, 1], s=marker_size, c='r', marker='x')
        plt.scatter(E_nodes, test_result[:, 2], s=marker_size, c='r', marker='x')
    
        plt.scatter(E_nodes, inter_result[:, 0], s=marker_size, c='b', marker='x')
        plt.scatter(E_nodes, inter_result[:, 1], s=marker_size, c='b', marker='x')
        plt.scatter(E_nodes, inter_result[:, 2], s=marker_size, c='b', marker='x')
    
        for kk in range(NX + 3):
            plt.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
            plt.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
            
            plt.axvline(xmin, linestyle='-', c='k', alpha=0.2)
            plt.axvline(xmax, linestyle='-', c='k', alpha=0.2)
        
        plt.xlim(xmin - 1.5*dx, xmax + 2*dx)
        plt.legend()
    return



def test_current_push():
    NX   = 32      #const.NX
    xmin = 0.0     #const.xmin
    xmax = 2*np.pi #const.xmax
    k    = 1.0
    marker_size = 20
    
    dx      = xmax / NX
    x       = np.arange(xmin, xmax, dx/100.)
    E_nodes = (np.arange(NX + 3) - 0.5) * dx
    B_nodes = (np.arange(NX + 3) - 1.0) * dx
        
    E_in    = np.zeros((NX + 3, 3))
    J_in    = np.zeros((NX + 3, 3))
    L_in    = np.ones((NX + 3))         # Charge-like  (const.)
    G_in    = np.ones((NX + 3, 3))      # Current-like (const.)
    
    Be       = np.zeros((NX + 3, 3))
    Be[:, 0] = np.sin(1.0*E_nodes*k)
    Be[:, 1] = np.sin(2.0*E_nodes*k)
    Be[:, 2] = np.sin(3.0*E_nodes*k)
    
    B       = np.zeros((NX + 3, 3))
    B[:, 0] = np.sin(1.0*B_nodes*k)
    B[:, 1] = np.sin(2.0*B_nodes*k)
    B[:, 2] = np.sin(3.0*B_nodes*k)
    
    DT        = 2.0
    
    J_out = sources.push_current(J_in, E_in, B, L_in, G_in, DT)
    
    J_an       = np.zeros((NX + 3, 3))
    J_an[:, 0] = Be[:, 2] - Be[:, 1]
    J_an[:, 1] = Be[:, 0] - Be[:, 2]
    J_an[:, 2] = Be[:, 1] - Be[:, 0]

    if True:
        plt.figure()
        plt.scatter(E_nodes, J_an[:, 0], s=marker_size, c='k', marker='o')
        plt.scatter(E_nodes, J_an[:, 1], s=marker_size, c='k', marker='o')
        plt.scatter(E_nodes, J_an[:, 2], s=marker_size, c='k', marker='o')
    
        plt.scatter(E_nodes, J_out[:, 0], s=marker_size, c='r', marker='x')
        plt.scatter(E_nodes, J_out[:, 1], s=marker_size, c='r', marker='x')
        plt.scatter(E_nodes, J_out[:, 2], s=marker_size, c='r', marker='x')
    
        for kk in range(NX + 3):
            plt.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
            plt.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
            
            plt.axvline(xmin, linestyle='-', c='k', alpha=0.2)
            plt.axvline(xmax, linestyle='-', c='k', alpha=0.2)
        
        plt.xlim(xmin - 1.5*dx, xmax + 2*dx)
        plt.legend()
    return


def test_varying_background_function():
    t  = np.arange(0, 1000)
    Bv = np.zeros((t.shape[0], const.NX + 3, 3), dtype=float)
    
    for ii in range(t.shape[0]):
        Bv[ii, :, :] = fields.uniform_time_varying_background(t[ii])

    plt.plot(t, Bv[:, 0, 0])
    return


def test_push_B_w_varying_background():
    max_inc   = 101
    DT        = 1.0
    x_offset  = 200e-9
    
    E  = np.zeros((const.NX + 3, 3))
    B  = np.zeros((const.NX + 3, 3), dtype=float)
    x  = np.arange(const.NX + 3)
    #fig, ax = plt.subplots()
    
    B[:, 0] += x_offset
    
    qq = 1
    while qq < max_inc:
        # push_B(B, E, DT, qq)
        B = fields.push_B(B, E, DT, qq, half_flag=1)
        print('t = {}s, B = {}nT'.format((qq - 0.5) * DT, B[0, :]*1e9))
# =============================================================================
#         ax.clear()
#         ax.plot(x, B[:, 0] * 1e9)
#         ax.plot(x, B[:, 1] * 1e9)
#         ax.plot(x, B[:, 2] * 1e9)
#         ax.set_title('Bx at t = {}'.format((qq - 0.5) * DT))
#         ax.set_ylim(-(const.HM_amplitude + x_offset)*1e9, (const.HM_amplitude + x_offset)*1e9)
#         ax.set_xlim(0, const.NX + 3)
#         ax.set_ylabel('Magnetic Field (nT)')
#         ax.set_xlabel('Cell number')
#         plt.pause(0.1)
# =============================================================================
        
        B = fields.push_B(B, E, DT, qq, half_flag=0)
        print('t = {}s, B = {}nT'.format(qq * DT, B[0, :]*1e9))
# =============================================================================
#         ax.clear()
#         ax.plot(x, B[:, 0] * 1e9)
#         ax.plot(x, B[:, 1] * 1e9)
#         ax.plot(x, B[:, 2] * 1e9)
#         ax.set_title('Bx at t = {}'.format(qq * DT))
#         ax.set_ylim(-(const.HM_amplitude + x_offset)*1e9, (const.HM_amplitude + x_offset)*1e9)
#         ax.set_xlim(0, const.NX + 3)
#         ax.set_ylabel('Magnetic Field (nT)')
#         ax.set_xlabel('Cell number')
#         plt.pause(0.1)
# =============================================================================
        
        qq += 1
    return


if __name__ == '__main__':
    #check_position_distribution()
    #animate_moving_weight()
    #test_particle_orbit()
    test_curl_B()
    test_curl_E()
    #test_grad_P_varying_qn()
    #test_cross_product()
    #test_cspline_interpolation()
    #test_E_convective()
    #test_E_hall()
    #test_interp_cross_manual()
    #test_CAM_CL()
    #test_current_push()
    #test_E_convective_exelectron()
    #test_varying_background_function()
    #test_push_B_w_varying_background()
    #test_weight_conservation()
    #test_weight_shape()
    