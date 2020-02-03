# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:42:13 2017

@author: iarey
"""
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import os
import pdb

import simulation_parameters_1D as const
import particles_1D             as particles
import sources_1D               as sources
import fields_1D                as fields
import auxilliary_1D            as aux
import init_1D as init
from matplotlib import animation


def test_boris():
    def velocity_update(vel_in, DT):
        qmi = 0.5 * DT * const.q / const.mp                                 # Charge-to-mass ration for ion of species idx[ii]

        Bp = np.array([B0, 0, 0])
        Ep = np.array([0 , 0, 0])

        v_minus = vel_in + qmi * Ep                                         # First E-field half-push
       
        T = qmi * Bp                                                        # Vector Boris variable
        S = 2.*T / (1. + T[0] ** 2 + T[1] ** 2 + T[2] ** 2)                 # Vector Boris variable

        v_prime    = np.zeros(3)
        v_prime[0] = v_minus[0] + v_minus[1] * T[2] - v_minus[2] * T[1]     # Magnetic field rotation
        v_prime[1] = v_minus[1] + v_minus[2] * T[0] - v_minus[0] * T[2]
        v_prime[2] = v_minus[2] + v_minus[0] * T[1] - v_minus[1] * T[0]
                
        v_plus     = np.zeros(3)
        v_plus[0]  = v_minus[0] + v_prime[1] * S[2] - v_prime[2] * S[1]
        v_plus[1]  = v_minus[1] + v_prime[2] * S[0] - v_prime[0] * S[2]
        v_plus[2]  = v_minus[2] + v_prime[0] * S[1] - v_prime[1] * S[0]

        vel_out = v_plus +  qmi * Ep                                     # Second E-field half-push
        return vel_out

    def position_update(POS, VEL, DT):
        return POS + VEL[0] * DT
    
    B0        = 2e-7
    gyfreq    = const.q * B0 / (2 * np.pi * const.mp)
    orbit_res = 0.05
    max_rev   = 1
    dt        = orbit_res / gyfreq 
    max_inc   = int(max_rev / (gyfreq*dt))
    
    pos       = 0
    vel       = np.array([0, 1000, 0])
    
    pos_history = np.zeros( max_inc)
    vel_history = np.zeros((max_inc, 3))
    
    tt = 0; t_total = 0
    
    vel = velocity_update(vel, -0.5*dt)
    while tt < max_inc:
        pos_history[tt] = pos
        vel_history[tt] = vel
        
        vel = velocity_update(vel, dt)
        pos = position_update(pos, vel, dt)  
        
        tt      += 1
        t_total += dt
    
    time = np.array([ii * dt for ii in range(max_inc)])
    plt.plot(time, vel_history, marker='o')
    return




def boundary_idx64(time_arr, start, end):
    '''Returns index values corresponding to the locations of the start/end times in a numpy time array, if specified times are np.datetime64'''
    idx1 = np.where(abs(time_arr - start) == np.min(abs(time_arr - start)))[0][0] 
    idx2 = np.where(abs(time_arr - end)   == np.min(abs(time_arr - end)))[0][0]
    return idx1, idx2


def r_squared(data, model):                  
    '''Calculates a simple R^2 value for the fit of model against data. Accepts single dimensional
    arrays only.'''
    SS_tot = np.sum(np.square(data - data.mean()))              # Total      sum of squares
    SS_res = np.sum(np.square(data - model))                    # Residuals  sum of squares
    r_sq   = 1 - (SS_res / SS_tot)                              # R^2 calculation (most general)
    return r_sq


def check_cell_velocity_distribution(pos, vel, node_number=const.NC // 2, j=0): #
    '''Checks the velocity distribution of a particle species within a specified cell
    '''
    x_node = (node_number - 0.5) * const.dx   # Position of E-field node
    f = np.zeros((1, 3))
    count = 0

    for ii in np.arange(const.idx_bounds[j, 0], const.idx_bounds[j, 1]):
        if (abs(pos[ii] - x_node) <= 0.5*const.dx):
            f = np.append(f, [vel[0:3, ii]], axis=0)
            count += 1
    
    print('{} particles counted for diagnostic'.format(count))
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
    
    Called by main_1D()
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


def test_weight_shape_and_alignment():
    plt.ion()
    
    positions  = np.array([-1.5]) * const.dx
    
    XMIN       = const.xmin
    XMAX       = const.xmax
    E_nodes    = const.E_nodes
    B_nodes    = const.B_nodes    
    
    E_grid     = False
    arr_offset = 0       if E_grid is True else 1
    X_grid     = E_nodes if E_grid is True else B_nodes
    X_color    = 'r'     if E_grid is True else 'b'
    
    Np         = positions.shape[0]
    Nc         = const.NX + 2*const.ND + arr_offset
    
    W_test     = np.zeros(Nc) 
    left_nodes = np.zeros(Np, dtype=int)
    weights    = np.zeros((3, Np))

    particles.assign_weighting_TSC(positions, left_nodes, weights, E_nodes=E_grid)
    
    plt.figure(figsize=(16,10))
    for ii in range(positions.shape[0]):
        
        plt.scatter(positions[ii], 1.0, c='k')
        
        for jj in range(3):
            xx = left_nodes[ii] + jj
            
            plt.axvline(X_grid[xx], linestyle='-', c=X_color, alpha=1.0)
            
            W_test[xx] = weights[jj, ii]
    
    plt.plot(X_grid, W_test, marker='o')
    plt.title('Total Weighting: {}'.format(W_test.sum()))

    # Draw nodes, limits
    for ii in range(E_nodes.shape[0]):
        plt.axvline(E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[ii], linestyle='--', c='b', alpha=0.2)
    
    plt.axvline(XMIN, linestyle=':', c='k', alpha=0.5)
    plt.axvline(XMAX, linestyle=':', c='k', alpha=0.5)
    plt.xlim(B_nodes[0], B_nodes[-1])
    return


def check_source_term_boundaries(qn, ji):
    '''
    Called in main_1D()
    '''
    E_nodes  = (np.arange(const.NX + 2*const.ND    ) - const.ND  + 0.5) * const.dx / 1e3
    B_nodes  = (np.arange(const.NX + 2*const.ND + 1) - const.ND  - 0.0) * const.dx / 1e3
    
    plt.figure()
    plt.plot(E_nodes, qn, marker='o')
    
    for ii in range(E_nodes.shape[0]):
        plt.axvline(E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[ii], linestyle='--', c='b', alpha=0.2)
     
    plt.axvline(B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    
    plt.axvline(const.xmin / 1e3, linestyle=':', c='k', alpha=0.5)
    plt.axvline(const.xmax / 1e3, linestyle=':', c='k', alpha=0.5)
    plt.ylabel('Charge density')
    plt.xlabel('x (km)')
    
    plt.figure()
    plt.plot(E_nodes, ji, marker='o')
    
    for ii in range(E_nodes.shape[0]):
        plt.axvline(E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[ii], linestyle='--', c='b', alpha=0.2)
     
    plt.axvline(B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    
    plt.axvline(const.xmin / 1e3, linestyle=':', c='k', alpha=0.5)
    plt.axvline(const.xmax / 1e3, linestyle=':', c='k', alpha=0.5)
    plt.ylabel('Current density')
    plt.xlabel('x (km)')
    
    
    return


def test_density_and_velocity_deposition():
    # Change dx to 1 and NX/ND/ppc to something reasonable to make this nice
    # Works fine with one species, why not two?
    E_nodes  = (np.arange(const.NX + 2*const.ND    ) - const.ND  + 0.5) * const.dx
    B_nodes  = (np.arange(const.NX + 2*const.ND + 1) - const.ND  - 0.0) * const.dx
     
    POS, VEL, IE, W_ELEC, IB, W_MAG, IDX  = init.initialize_particles()
    Q_DENS, Q_DENS_ADV, JI, NI, NU        = init.initialize_source_arrays()
    temp1D                                = np.zeros(const.NC, dtype=np.float64) 
    
    sources.collect_moments(VEL, IE, W_ELEC, IDX, Q_DENS, JI, NI, NU, temp1D, mirror=False) 

    # Plot particle position
    ypos = np.ones(POS.shape[0] // 2)
    plt.scatter(POS[:POS.shape[0] // 2 ], ypos + 0.1, c='r')
    plt.scatter(POS[ POS.shape[0] // 2:], ypos + 0.2, c='b')

    # Plot charge density
    plt.plot(E_nodes, Q_DENS / Q_DENS.max(), marker='o')
        
    for ii in range(E_nodes.shape[0]):
        plt.axvline(E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[ii], linestyle='--', c='b', alpha=0.2)
     
    plt.axvline(0         , color='k')
    plt.axvline(const.xmax, color='k')
    plt.axvline(B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    return


def check_density_deposition():
    # Change dx to 1 and NX/ND to reasonable values to make this nice
    E_nodes  = (np.arange(const.NX + 2*const.ND    ) - const.ND  + 0.5) * const.dx
    B_nodes  = (np.arange(const.NX + 2*const.ND + 1) - const.ND  - 0.0) * const.dx
     
    pos        = np.array([0.0, 1.0, 6.0])
    pos        = np.arange(0.0, 6.05, 0.05)
    
    vel        = np.zeros((3, pos.shape[0]))
    idx        = np.zeros(pos.shape[0], dtype=np.uint8)
    Ie         = np.zeros(pos.shape[0],      dtype=np.uint16)
    W_elec     = np.zeros((3, pos.shape[0]), dtype=np.float64)
    
    q_dens, q_dens_adv, Ji, ni, nu = init.initialize_source_arrays()
    temp1D                         = np.zeros(const.NC, dtype=np.float64) 
    
    particles.assign_weighting_TSC(pos, Ie, W_elec)
    sources.collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu, temp1D, mirror=False) 

    # Plot normalized charge density
    q_dens /= q_dens.max()
    
    ypos = np.ones(pos.shape[0]) * q_dens.max() * 1.1
    plt.plot(E_nodes, q_dens, marker='o')
    plt.scatter(pos, ypos, marker='x', c='k')
        
    for ii in range(E_nodes.shape[0]):
        plt.axvline(E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[ii], linestyle='--', c='b', alpha=0.2)
     
    plt.axvline(0          , linestyle=':', c='k'       , alpha=0.7)
    plt.axvline(const.xmax , linestyle=':', c='k'       , alpha=0.7)
    plt.axvline(B_nodes[ 0], linestyle='-', c='darkblue', alpha=0.7)
    plt.axvline(B_nodes[-1], linestyle='-', c='darkblue', alpha=0.7)
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
    '''
    Confirmed
    '''
    NX   = 50     
    ND   = 10
    NC   = NX + 2*ND
    xmin = 0.0    
    xmax = 2*np.pi
    
    dx   = xmax / NX
    k    = 1.0

    E_nodes  = (np.arange(NC    ) - ND  + 0.5) * dx
    B_nodes  = (np.arange(NC + 1) - ND  - 0.0) * dx

    # Inputs and analytic solutions
    B_input       = np.zeros((NC + 1, 3))
    B_input[:, 0] = np.cos(1.0*k*B_nodes)
    B_input[:, 1] = np.cos(1.5*k*B_nodes)
    B_input[:, 2] = np.cos(2.0*k*B_nodes)
    
    curl_B_anal       = np.zeros((NC, 3))
    curl_B_anal[:, 1] =  2.0 * k * np.sin(2.0*k*E_nodes) * dx
    curl_B_anal[:, 2] = -1.5 * k * np.sin(1.5*k*E_nodes) * dx
    
    curl_B_FD = np.zeros((NC, 3))
    fields.curl_B_term(B_input, curl_B_FD)
    
    ## DO THE PLOTTING ##
    plt.figure(figsize=(15, 15))
    marker_size = None
        
    plt.scatter(E_nodes, curl_B_anal[:, 1], marker='o', c='k', s=marker_size, label='By Node Solution')
    plt.scatter(E_nodes, curl_B_FD[  :, 1], marker='x', c='b', s=marker_size,   label='By Finite Difference')
      
    plt.scatter(E_nodes, curl_B_anal[:, 2], marker='o', c='k', s=marker_size, label='Bz Node Solution')
    plt.scatter(E_nodes, curl_B_FD[  :, 2], marker='x', c='r', s=marker_size,   label='Bz Finite Difference')   
    plt.title(r'Test of $\nabla \times B$')

    for ii in range(E_nodes.shape[0]):
        plt.axvline(E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[ii], linestyle='--', c='b', alpha=0.2)
     
    plt.axvline(B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    
    plt.axvline(xmin, linestyle=':', c='k', alpha=0.5)
    plt.axvline(xmax, linestyle=':', c='k', alpha=0.5)
    plt.xlabel('x (km)')
    
    plt.legend()
    return


def test_curl_E():
    '''
    Confirmed
    '''
    NX   = 50     
    ND   = 10
    NC   = NX + 2*ND
    xmin = 0.0    
    xmax = 2*np.pi
    
    dx   = xmax / NX
    k    = 1.0

    E_nodes  = (np.arange(NC    ) - ND  + 0.5) * dx
    B_nodes  = (np.arange(NC + 1) - ND  - 0.0) * dx

    # Inputs and analytic solutions
    E_input       = np.zeros((NC, 3))
    E_input[:, 0] = np.cos(1.0*k*E_nodes)
    E_input[:, 1] = np.cos(1.5*k*E_nodes)
    E_input[:, 2] = np.cos(2.0*k*E_nodes)

    curl_E_FD = np.zeros((NC + 1, 3))
    fields.get_curl_E(E_input, curl_E_FD, DX=1)
    
    curl_E_anal       = np.zeros((NC + 1, 3))
    curl_E_anal[:, 1] =  2.0 * k * np.sin(2.0*k*B_nodes) * dx
    curl_E_anal[:, 2] = -1.5 * k * np.sin(1.5*k*B_nodes) * dx
    
    
    ## PLOT
    plt.figure(figsize=(15, 15))
    marker_size = None

    plt.scatter(B_nodes, curl_E_anal[:, 1], marker='o', c='k', s=marker_size, label='By Node Solution')
    plt.scatter(B_nodes, curl_E_FD[  :, 1], marker='x', c='b', s=marker_size, label='By Finite Difference')
      
    plt.scatter(B_nodes, curl_E_anal[:, 2], marker='o', c='k', s=marker_size, label='Bz Node Solution')
    plt.scatter(B_nodes, curl_E_FD[  :, 2], marker='x', c='r', s=marker_size, label='Bz Finite Difference')   
    plt.title(r'Test of $\nabla \times E$')

    for kk in range(NC):
        plt.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
    
    plt.axvline(B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    
    plt.axvline(xmin, linestyle='-', c='k', alpha=0.2)
    plt.axvline(xmax, linestyle='-', c='k', alpha=0.2)
    
    plt.legend()
    return


def test_grad_P_varying_qn():
    k    = 2.0

    E_nodes  = (np.arange(const.NC    ) - const.ND  + 0.5) * const.dx
    B_nodes  = (np.arange(const.NC + 1) - const.ND  - 0.0) * const.dx

    # Set analytic solutions (input/output)
    qn_input   = np.cos(  2 * np.pi * k * E_nodes / const.xmax)  * const.q * const.ne
    te_input   = np.ones( const.NC)*const.Te0
    gp_diff    = np.zeros(const.NC)
    temp       = np.zeros(const.NC + 1)
    
    dne_anal   = -2 * np.pi * k * np.sin(2 * np.pi*k*E_nodes / const.xmax) * const.ne / const.xmax / const.dx
    gp_anal    = const.kB * const.Te0 * dne_anal * const.dx          # Analytic solution at nodes 

    # Finite differences
    fields.get_grad_P(qn_input, te_input, gp_diff, temp)

    ## PLOT ##
    plt.figure(figsize=(15, 15))
    marker_size = None

    plt.scatter(E_nodes, gp_anal*1e13, marker='o', c='k', s=marker_size, label='Node Solution')
    plt.scatter(E_nodes, gp_diff*1e13, marker='x', c='r', s=marker_size, label='Finite Difference')
    
    plt.title(r'Test of $\nabla p_e$')

    for kk in range(const.NC):
        plt.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
    
    plt.axvline(B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    
    plt.axvline(const.xmin, linestyle='-', c='k', alpha=0.2)
    plt.axvline(const.xmax, linestyle='-', c='k', alpha=0.2)
    
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


def test_varying_background_function():
    t  = np.arange(0, 1000)
    Bv = np.zeros((t.shape[0], const.NX + 3, 3), dtype=float)
    
    for ii in range(t.shape[0]):
        Bv[ii, :, :] = fields.uniform_time_varying_background(t[ii])

    plt.plot(t, Bv[:, 0, 0])
    return



def save_diagnostic_plots(qq, pos, vel, B, E, q_dens, Ji, sim_time, DT):
    
    plt.ioff()

    fig_size = 4, 7                                                             # Set figure grid dimensions
    fig = plt.figure(figsize=(20,10))                                           # Initialize Figure Space
    fig.patch.set_facecolor('w')                                                # Set figure face color

    npos       = pos / const.dx + const.ND                                      # Cell particle position
    nvel       = vel / const.va                                                 # Normalized velocity

    qdens_norm = q_dens / (const.density*const.charge).sum()                    # Normalized change density
     
#----- Velocity (x, y) Plots: Hot Species
    ax_vx   = plt.subplot2grid(fig_size, (0, 0), rowspan=2, colspan=3)
    ax_vy   = plt.subplot2grid(fig_size, (2, 0), rowspan=2, colspan=3)

    for jj in range(const.Nj):
        ax_vx.scatter(npos[const.idx_bounds[jj, 0]: const.idx_bounds[jj, 1]], nvel[0, const.idx_bounds[jj, 0]: const.idx_bounds[jj, 1]], s=3, c=const.temp_color[jj], lw=0, label=const.species_lbl[jj])
        ax_vy.scatter(npos[const.idx_bounds[jj, 0]: const.idx_bounds[jj, 1]], nvel[1, const.idx_bounds[jj, 0]: const.idx_bounds[jj, 1]], s=3, c=const.temp_color[jj], lw=0)

    ax_vx.legend()
    ax_vx.set_title(r'Particle velocities vs. Position (x)')
    ax_vy.set_xlabel(r'Cell', labelpad=10)

    ax_vx.set_ylabel(r'$\frac{v_x}{vA}$', rotation=90)
    ax_vy.set_ylabel(r'$\frac{v_y}{vA}$', rotation=90)

    plt.setp(ax_vx.get_xticklabels(), visible=False)
    ax_vx.set_yticks(ax_vx.get_yticks()[1:])

    for ax in [ax_vy, ax_vx]:
        ax.set_xlim(const.ND, const.NX + const.ND)
        ax.set_ylim(-20, 20)

#----- Density Plot
    ax_den = plt.subplot2grid((fig_size), (0, 3), colspan=3)                     # Initialize axes
    
    ax_den.plot(qdens_norm, color='green')                                       # Create overlayed plots for densities of each species

    for jj in range(const.Nj):
        ax_den.plot(qdens_norm, color=const.temp_color[jj])
        
    ax_den.set_title('Normalized Densities and Fields')                          # Axes title (For all, since density plot is on top
    ax_den.set_ylabel(r'$\frac{n_i}{n_0}$', fontsize=14, rotation=0, labelpad=5) # Axis (y) label for this specific axes
    ax_den.set_ylim(0, 2)
    
#----- E-field (Ex) Plot
    ax_Ex = plt.subplot2grid(fig_size, (1, 3), colspan=3, sharex=ax_den)

    ax_Ex.plot(E[:, 0], color='red', label=r'$E_x$')
    ax_Ex.plot(E[:, 1], color='cyan', label=r'$E_x$')
    ax_Ex.plot(E[:, 2], color='black', label=r'$E_x$')

    ax_Ex.set_xlim(0, const.NC)

    #ax_Jx.set_yticks(np.arange(-200e-5, 201e-5, 50e-5))
    #ax_Jx.set_yticklabels(np.arange(-150, 201, 50))
    ax_Ex.set_ylabel(r'$E$', labelpad=25, rotation=0, fontsize=14)

#----- Magnetic Field (By) and Magnitude (|B|) Plots
    ax_By = plt.subplot2grid((fig_size), (2, 3), colspan=3, sharex=ax_den)
    ax_B  = plt.subplot2grid((fig_size), (3, 3), colspan=3, sharex=ax_den)

    mag_B  = (np.sqrt(B[:, 0] ** 2 + B[:, 1] ** 2 + B[:, 2] ** 2)) / const.B0
    B_norm = B / const.B0                                                           

    ax_B.plot(mag_B, color='g')
    ax_By.plot(B_norm[:, 1], color='g') 
    ax_By.plot(B_norm[:, 2], color='b') 

    ax_B.set_xlim(0,  const.NC + 1)
    ax_By.set_xlim(0, const.NC + 1)

    ax_B.set_ylim(0, 2)
    ax_By.set_ylim(-1, 1)

    ax_B.set_ylabel( r'$|B|$', rotation=0, labelpad=20, fontsize=14)
    ax_By.set_ylabel(r'$\frac{B_{y,z}}{B_0}$', rotation=0, labelpad=10, fontsize=14)
    ax_B.set_xlabel('Cell Number')

    for ax in [ax_den, ax_Ex, ax_By]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_yticks(ax.get_yticks()[1:])

    for ax in [ax_den, ax_Ex, ax_By, ax_B]:
        qrt = const.NC / (4.)
        ax.set_xticks(np.arange(0, const.NC + qrt, qrt))
        ax.grid()

#----- Plot Adjustments
    plt.tight_layout(pad=1.0, w_pad=1.8)
    fig.subplots_adjust(hspace=0)
    
    plt.figtext(0.855, 0.94, 'Step : {:>7d}     '.format( qq     ), fontname='monospace', fontsize=14)
    plt.figtext(0.855, 0.90, 'Time : {:>7.3f} s '.format(sim_time), fontname='monospace', fontsize=14)
    plt.figtext(0.855, 0.86, 'DT   : {:>7.3f} ms'.format(DT * 1e3), fontname='monospace', fontsize=14)

    filename = 'diag%05d.png' % qq
    path     = const.drive + '//' + const.save_path + '//run_{}'.format(const.run_num) + '//diagnostic_plots//'

    if os.path.exists(path) == False:                                   # Create data directory
        os.makedirs(path)

    fullpath = path + filename
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
    print('Plot saved'.format(qq))
    plt.close('all')
    return


def visualize_inhomogenous_B():
    '''
    This isn't right, or is it? Work it out later
    '''
    L    = 4.3      ;   q = 1.602e-19
    RE   = 6371000  ;   m = 1.673e-27
    
    B_eq  = 200e-9               # Tesla at equator (Br = 0)
    a     = 4.5 / (L * RE)       # Where does the 4.5 come from? 1/s variation, but derivation?
    Wx_eq = 30e3 * q             # Initial perp energy in Joules
    vx_
    mu_eq = Wx_eq / B_eq         # First adiabatic invariant
    
    xmax = 1*RE                  # How long should this be?
    x    = np.linspace(-xmax, xmax, 1000)
    Nx   = x.shape[0]
    
    B_x = B_eq * (1 + a * x**2)
    B_r = np.zeros((Nx, 3))
    
    for ii in range(Nx):
        aa = 1.0   
        bb = B_x[ii] ** 2
        cc = a * m * 1

        
    pdb.set_trace()
    plt.figure()
    plt.plot(x / RE, B_x, c='k', label=r'$B_\parallel$')
    plt.plot(x / RE, B_r, c='r', label=r'$B_\perp$')
    plt.xlabel('x (RE)')
    plt.ylabel('nT')
    plt.legend()
    plt.show()
    
# =============================================================================
#     plt.figure()
#     plt.title('Percentage of main field composed of radial component')
#     plt.scatter(x / RE, pct)
#     plt.ylabel(r'%')
# =============================================================================
    
    return


def plot_dipole_field_line(length=True):
    '''
    Plots field lines with basic L = r*sin^2(theta) relation. Can plot
    multiple for all in Ls. Can also calculate arclengths from lat_st/lat_min
    and print as the title (can be changed if you want)
    '''
    Ls         = [4.3]
    dtheta     = 0.1
    theta      = np.arange(0, 180. + dtheta, dtheta) * np.pi / 180
        
    lat_st = 80  
    lat_en = 100
    
    plt.figure()
    plt.gcf().gca().add_artist(plt.Circle((0,0), 1.0, color='k'))
    
    for L in Ls:
        r     = L * np.sin(theta) ** 2
        x     = r * np.cos(theta)
        y     = r * np.sin(theta) 
        
        plt.scatter(y, x, c='b', s=1, marker='o')
        
    plt.axis('equal')
    plt.axhline(0, ls=':', alpha=0.2, color='k')
    plt.axvline(0, ls=':', alpha=0.2, color='k')
    
    if length == True:
        idx_start, idx_end = boundary_idx64(theta * 180 / np.pi, lat_st, lat_en)
        plt.scatter(y[idx_start:idx_end], x[idx_start:idx_end], c='k', s=4, marker='x')
        
        # Calculate arclength, r*dt
        length = 0
        for ii in range(idx_start, idx_end):
            length += r[ii] * dtheta * np.pi / 180
        plt.title('Arclength from {} deg to {} deg at L = {} : {:>5.2f} R_E'.format(lat_st, lat_en, L, length))
    return

def check_particle_position_individual():
    '''
    Verified. RC and cold population positions load fine
    '''
    pos, vel, idx = init.uniform_gaussian_distribution_quiet()

    plt.figure()
    for jj in range(const.Nj):
        st = const.idx_start[jj]
        en = const.idx_end[  jj]    
        Np = en - st
        plt.scatter(pos[st:en], np.ones(Np)*jj, color=const.temp_color[jj])
    return


def compare_parabolic_to_dipole():
    '''
    To do: Calculate difference in magnetic strength along a field line, test
    how good this parabolic approx. is. Use dipole code to compare B0_x to
    B0_mu and the radial component B0_r to mod(B0_nu, B0_phi). Plot for Colin, 
    along with method Chapter.
    
    Shoji has a simulation extent on the order of R_E (0, 800 is about 6.3R_E,
    but is that symmetric?)
    
    Coded my own a based on equivalent values at +-30 degrees off equator. Maybe 
    alter code to 
    '''
    B_surf = 3.12e-5    # Magnetic field strength at Earth surface
    L      = 5.35       # Field line L shell
    dtheta = 0.01       # Angle increment
    lat_st = 80         # Minimum latitude
    lat_en = 110        # Maximum latitude
    
    min_theta = np.arcsin(np.sqrt(1 / (L))) * 180 / np.pi
    
    # Calculate dipole field intensity (nT) and locations in (r, theta)
    theta = np.arange(min_theta, 180. + dtheta - min_theta, dtheta) * np.pi / 180
    r     = L * np.sin(theta) ** 2
    B_mu  = (B_surf / (r ** 3)) * np.sqrt(3*np.cos(theta)**2 + 1) * 1e9

    if False:
        # Convert to (x,y) for plotting
        x     = r * np.cos(theta)
        y     = r * np.sin(theta)
        
        plt.figure(1)
        plt.gcf().gca().add_artist(plt.Circle((0,0), 1.0, color='k', fill=False))
        
        plt.scatter(y, x, c=B_mu, s=1)
        plt.colorbar().set_label('|B| (nT)', rotation=0, labelpad=20, fontsize=14)
        plt.clim(None, 1000)
        plt.xlabel(r'x ($R_E$)', rotation=0)
        plt.ylabel(r'y ($R_E$)', rotation=0, labelpad=10)
        plt.title('Geomagnetic Field Intensity at L = {}'.format(L))
        plt.axis('equal')
        plt.axhline(0, ls=':', alpha=0.2, color='k')
        plt.axvline(0, ls=':', alpha=0.2, color='k')    
    
    
    # Calculate cylindrical/parabolic approximation between lat st/en
    st, en = boundary_idx64(theta * 180 / np.pi, lat_st, lat_en)
    
    length = 0
    for ii in range(st, en):
        length += r[ii] * dtheta * np.pi / 180
    
    RE   = 1.0
    z    = np.linspace(-length/2, length/2, en - st, endpoint=True) * RE
    a    = 4.5 / (L * RE)
    B0_z = B_mu.min() * (1 + a * z ** 2)
    
    print('Domain length : {:5.2f}RE'.format(length))
    print('Minimum field : {:5.2f}nT'.format(B_mu.min()))
    print('Maximum field : {:5.2f}nT'.format(B_mu[st:en].max()))
    print('Max/Min ratio : {:5.2f}'.format(B_mu[st:en].max() / B_mu.min()))
    
    plt.figure(2)
    plt.scatter(z, B0_z,           label='Cylindrical approximation')
    plt.scatter(z/RE, B_mu[st:en], label='Dipole field intensity')
    plt.xlabel(r'z ($R_E$)',     rotation=0, fontsize=14)
    plt.ylabel(r'$B_\parallel$', rotation=0, fontsize=14)
    plt.legend()
    return


def test_mirror_motion():
    '''
    Diagnostic code to call the particle pushing part of the hybrid and check
    that its solving ok. Runs with zero background E field and B field defined
    by the constant background field specified in the parameter script.
    '''
    mid    = const.NC // 2
    pos    = np.array([0])
    idx    = np.array([0])
    vel    = np.array([const.va, const.va, const.va]).reshape((3, 1))
    W_elec = np.array([0, 1, 0]).reshape((3, 1))
    W_mag  = np.array([0, 1, 0]).reshape((3, 1))
    Ie     = np.array([mid])
    Ib     = np.array([mid])
    
    B_test = np.zeros((const.NC + 1, 3), dtype=np.float64) 
    E_test = np.zeros((const.NC, 3),     dtype=np.float64) 
    
    gyfreq   = const.q * const.B_eq / (2 * np.pi * const.mp)
    ion_ts   = const.orbit_res * 0.1 / gyfreq
    vel_ts   = 0.5 * const.dx / np.max(vel[0, :])
    DT       = min(ion_ts, vel_ts)
    #DT      *= 0.1
    
    max_rev  = 10 
    max_inc  = int(max_rev / (DT*gyfreq))
    
    pos_history = np.zeros((max_inc))
    vel_history = np.zeros((max_inc, 3))
    mag_history = np.zeros((max_inc, 3))
    
    dump_every  = 1; tt = 0; halves = 0; t_total = 0
    
    Bp = particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B_test, E_test, -0.5*DT)
    while tt < max_inc:
        Bp = particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B_test, E_test, DT)
        particles.position_update(pos, vel, idx, DT, Ie, W_elec)  
        
        if tt%dump_every == 0:
            pos_history[tt // 2**halves] = pos[0]
            vel_history[tt // 2**halves] = vel[:, 0]
            mag_history[tt // 2**halves] = Bp
        tt += 1
        t_total += DT
        
        # Check timestep
        vel_ts = 0.5*const.dx/np.max(np.abs(vel[0, :]))
        if vel_ts < DT:
            print('Timestep reduced')
            particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B_test, E_test, 0.5*DT)    # Re-sync vel/pos       

            DT           /= 2
            tt           *= 2
            max_inc      *= 2
            dump_every   *= 2
            halves       += 1
            
            particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B_test, E_test, -0.5*DT)    # De-sync      
        
    time = np.arange(pos_history.shape[0])

    if False:
        ## Plots position/mag timeseries ##
        fig, ax = plt.subplots()
        ax.plot(time, pos_history*1e-3, c='k', marker='o')
        ax.set_ylabel('x (km)')
        ax.set_xlabel('t (s)')
        
        ax2 = ax.twinx()
        ax2.plot(time, mag_history[:, 0], label='B0x')
        ax2.plot(time, mag_history[:, 1], label='B0y')
        ax2.plot(time, mag_history[:, 2], label='B0z')
        ax2.legend()
        ax2.set_ylabel('nT')
        
        ax.axhline(const.xmin*1e-3, color='k', ls=':')
        ax.axhline(const.xmax*1e-3, color='k', ls=':')
        
    if False:
        ## Plots velocity timeseries ##
        fig, ax = plt.subplots()
        ax.plot(time, vel_history[:, 0]*1e-3, marker='o', label='vx')
        ax.plot(time, vel_history[:, 1]*1e-3, marker='o', label='vy')
        ax.plot(time, vel_history[:, 2]*1e-3, marker='o', label='vz')
        ax.set_ylabel('t (s)')
        ax.set_xlabel('v (km/s)')
        ax.legend()
        
    if True:
        # Plot gyromotion of particle
        plt.plot(vel_history[:, 1], vel_history[:, 2], marker='o')
        plt.ylabel('vy (km/s)')
        plt.xlabel('vz (km/s)')
    return


if __name__ == '__main__':
    #check_position_distribution()
    #animate_moving_weight()
    #test_particle_orbit()
    #test_curl_B()
    #test_curl_E()
    #test_grad_P_varying_qn()
    #test_density_and_velocity_deposition()
    #check_density_deposition()
    #visualize_inhomogenous_B()
    #plot_dipole_field_line()
    #check_particle_position_individual()
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
    #test_weight_shape_and_alignment()
    #compare_parabolic_to_dipole()
    #test_boris()
    test_mirror_motion()
