# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:42:13 2017

@author: iarey
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import pdb

import simulation_parameters_2D as const
import particles_2D             as particles
import sources_2D               as sources
import fields_2D                as fields
import auxilliary_2D            as aux
from matplotlib import animation


def r_squared(data, model):                  
    '''
    Calculates a simple R^2 value for the fit of model against data. Accepts single dimensional
    arrays only.
    '''
    SS_tot = np.sum(np.square(data - data.mean()))              # Total      sum of squares
    SS_res = np.sum(np.square(data - model))                    # Residuals  sum of squares
    r_sq   = 1 - (SS_res / SS_tot)                              # R^2 calculation (most general)
    return r_sq


def check_cell_velocity_distribution(pos, vel, nodex, nodey, j): #
    '''
    Checks the velocity distribution of a particle species within a specified cell
    
    PUSH TO 2D
    '''
    x_node    = (nodex - 0.5) * const.dx   # Position of E-field node
    y_node    = (nodey - 0.5) * const.dx   # Position of E-field node
    cell_vels = np.zeros(3)
    
    count  = 0
    for ii in np.arange(const.idx_bounds[j, 0], const.idx_bounds[j, 1]):
        if (abs(pos[0, ii] - x_node) <= 0.5*const.dx and
            abs(pos[1, ii] - y_node) <= 0.5*const.dy):
            cell_vels = np.append(cell_vels, [vel[:, ii]], axis=0)
            count += 1
            
    fig = plt.figure(figsize=(12,10))
    fig.suptitle('Particle velocity distribution of species {} in cell ({},{})'.format(j, nodex, nodey))
    fig.patch.set_facecolor('w')

    ax_x = plt.subplot2grid((2, 3), (0,0), colspan=2, rowspan=2)
    ax_y = plt.subplot2grid((2, 3), (0,2))
    ax_z = plt.subplot2grid((2, 3), (1,2))

    xs, BinEdgesx = np.histogram((cell_vels[0] - const.velocity[j]) / const.va)
    bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
    ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
    ax_x.set_xlabel(r'$v_x / v_A$')
    #ax_x.set_xlim(-2, 2)

    ys, BinEdgesy = np.histogram(cell_vels[1] / const.va)
    by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
    ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
    ax_y.set_xlabel(r'$v_y / v_A$')
    #ax_y.set_xlim(-2, 2)

    zs, BinEdgesz = np.histogram(cell_vels[2] / const.va)
    bz = 0.5 * (BinEdgesz[1:] + BinEdgesz[:-1])
    ax_z.plot(bz, zs, '-', c='c', drawstyle='steps')
    ax_z.set_xlabel(r'$v_z / v_A$')
    #ax_z.set_xlim(-2, 2)

    plt.show()
    return

def check_position_distribution(pos):
    '''Checks the spatial distribution of a particle species j within the spatial domain
    
    PUSH TO 2D
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
    
    PUSH TO 2D
    '''
    for j in range(const.Nj):
        fig = plt.figure(figsize=(12,10))
        fig.suptitle('Velocity distribution of species {} in simulation domain'.format(j))
        fig.patch.set_facecolor('w')
        num_bins = const.cellpart / 5
    
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
    Plots the normalized weight for a single particle at 1e6 points along simulation
    domain. Should remain at 1 the whole time.
    '''
    nspace_x      = 100
    nspace_y      = 100
    xmax          = const.NX*const.dx
    ymax          = const.NY*const.dy
    pos_x         = np.linspace(0, xmax, nspace_x)
    pos_y         = np.linspace(0, ymax, nspace_y)
    positions     = np.zeros((2, nspace_x * nspace_y))  # x and y
    weights       = np.zeros((2, nspace_x * nspace_y))  # elec and mag
    
    xx = 0
    for ii in range(nspace_x):
        for jj in range(nspace_y):
            positions[0, xx] = pos_x[ii]
            positions[1, xx] = pos_y[jj]
            
            pos = np.array([pos_x[ii], pos_y[jj]]).reshape((2, 1))
            
            for grid in [0, 1]:
                I = np.zeros(pos.shape)
                W = np.zeros((2, 1, 3))
                particles.assign_weighting_TSC(pos, I, W, E_nodes=(grid==0))
                
                for mm in range(3):
                    for nn in range(3):
                        weights[grid, xx] += W[0, 0, mm] * W[1, 0, nn]
                        
                print(weights[grid, xx])
            xx += 1
    return


def position_update_single(pos, vel, dt, bounds):
    pos[0] += vel[0] * dt
    pos[1] += vel[1] * dt
    
    if pos[0] < bounds[0]:
        pos[0] += bounds[1]

    if pos[0] > bounds[1]:
        pos[0] -= bounds[1]
        
    if pos[1] < bounds[2]:
        pos[1] += bounds[3]
        
    if pos[1] > bounds[3]:
        pos[1] -= bounds[3]
    return
            

def test_weight_correctness():
    '''
    Nodes being referenced correctly as of 19/07/19
    Weights being assigned properly?
    '''
    dx = const.dx
    dy = const.dy
    
    xmax    = const.NX*dx
    ymax    = const.NY*dy
    cells_x = np.arange(const.NX + 3)*dx
    cells_y = np.arange(const.NX + 3)*dy
    
    EX, EY = np.meshgrid(cells_x - 0.5*dx, cells_y - 0.5*dy)
    BX, BY = np.meshgrid(cells_x - 1.0*dx, cells_y - 1.0*dy)
        
    pos = np.array([0.5*xmax, 0.5*ymax]).reshape((2, 1))
        
    Ie = np.zeros(pos.shape)
    We = np.zeros((2, 1, 3))
    
    Ib = np.zeros(pos.shape)
    Wb = np.zeros((2, 1, 3))
        
    dt      = 0.1
    
    pos = np.array([np.random.uniform(0, xmax),
                    np.random.uniform(0, ymax)]
                  ).reshape((2, 1))
    plt.figure()
    
    
    for ii in range(1):
        particles.assign_weighting_TSC(pos, Ie, We, E_nodes=True)
        particles.assign_weighting_TSC(pos, Ib, Wb, E_nodes=False)
        
        Ie[0] -= 0.5; Ie[0] *= dx
        Ie[1] -= 0.5; Ie[1] *= dy
        
        Ib[0] -= 1.0; Ib[0] *= dx; 
        Ib[1] -= 1.0; Ib[1] *= dy; 
        
        plt.gca().clear()
        ## GRID SPACE ##
        plt.scatter(EX, EY, c='r', marker='x', s=20)     # E node locations
        plt.scatter(BX, BY, c='b', marker='x', s=20)     # B node locations
        
        plt.xlim(-1.5*dx, xmax + 1.5*dx)
        plt.ylim(-1.5*dy, ymax + 1.5*dy)
        
        for mm in range(const.NX):
            plt.vlines(mm * dx, 0, ymax, linestyles='--', alpha=0.20)
        
        for nn in range(const.NY):
            plt.hlines(nn * dy, 0, xmax, linestyles='--', alpha=0.20)
        
        plt.hlines(0   , 0, xmax, color='k')
        plt.hlines(ymax, 0, xmax, color='k')
        plt.vlines(0   , 0, ymax, color='k')
        plt.vlines(xmax, 0, ymax, color='k')
        ################
        
        ## PARTICLE STUFF ##
        particle_shape = patches.Rectangle((pos[0] - dx, pos[1] - dy), 2*dx, 2*dy, alpha=0.2, color='k')
        
        wec = 0; wbc = 0
        for ii in range(3):
            for jj in range(3):
                W_elec = We[0, 0, ii] * We[1, 0, jj]
                W_mag  = Wb[0, 0, ii] * Wb[1, 0, jj]
                
                plt.scatter(Ie[0] + ii*dx, Ie[1] + jj*dy, s=20, marker='o', c='r')
                plt.gca().annotate(W_elec, (Ie[0] + ii*dx, Ie[1] + jj*dy))
                
                plt.scatter(Ib[0] + ii*dx, Ib[1] + jj*dy, s=20, marker='o', c='b')
                plt.gca().annotate(W_mag, (Ib[0] + ii*dx, Ib[1] + jj*dy))
                
                wec += W_elec; wbc += W_mag
                
        print(wec, wbc)
        
        plt.scatter(pos[0], pos[1], c='k', marker='o', s=20)
        plt.gca().add_patch(particle_shape)
        
        plt.pause(0.2)
        
        vx  = const.dx * 0.5#np.random.uniform(-0.5, 0.5)
        vy  = const.dy * 0.2#np.random.uniform(-0.5, 0.5) 
        
        vel = np.array([[vx], [vy], [ 0.]]) / dt
    
        position_update_single(pos, vel, dt, [0, xmax, 0, ymax])
        
# =============================================================================
#         figManager = plt.get_current_fig_manager()
#         figManager.window.showMaximized()
# =============================================================================
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

def test_ghost_cell_filling():
    '''
    Seems to work fine with cos x put in x and y directions.
    
    Tested on 19/07/19
    '''
    NX   = const.NX  ; NY   = const.NY 
    xmax = const.xmax; ymax = const.ymax
    dx   = const.dx  ; dy   = const.dy
    kx   = 2 * np.pi / xmax

    cells_x = np.arange(const.NX + 3)*dx
    cells_y = np.arange(const.NX + 3)*dy
    
    EX, EY = np.meshgrid(cells_x - 0.5*dx, cells_y - 0.5*dy)
    BX, BY = np.meshgrid(cells_x - 1.0*dx, cells_y - 1.0*dy)

    E_grid_quantity          = np.zeros((NX + 3, NY + 3, 3))
    E_grid_quantity[1: NX + 1, 1: NY + 1, 0] = np.cos(1.0*kx*EY[1: NX + 1, 1:NY + 1]) 
    
    B_grid_quantity          = np.zeros((NX + 3, NY + 3, 3))
    B_grid_quantity[1: NX + 2, 1: NY + 2, 0] = np.cos(1.0*kx*BY[1: NX + 2, 1:NY + 2]) 
   
    
    if True:
        fig = plt.figure(figsize=(15, 15))
        ax  = fig.add_subplot(111, projection='3d')
        
        ## GRID SPACE ##
        ax.scatter(EX, EY, c='r', marker='x', s=20)     # E node locations
        ax.scatter(BX, BY, c='b', marker='x', s=20)     # B node locations
        
        ax.plot_wireframe(BX, BY, B_grid_quantity[:, :, 0], color='b')
        fields.B_grid_boundary_and_ghost_cells(B_grid_quantity)
        sources.manage_E_grid_ghost_cells(E_grid_quantity)
        ax.plot_wireframe(BX, BY, B_grid_quantity[:, :, 0], color='r')
        
        ax.set_xlim(-1.5*dx, xmax + 1.5*dx)
        ax.set_ylim(-1.5*dy, ymax + 1.5*dy)
        
        border = np.array([ [0   , 0   , 0],
                            [0   , ymax, 0],
                            [xmax, ymax, 0],
                            [xmax, 0   , 0],
                            [0   , 0   , 0]])
        ax.plot(border[:, 0], border[:, 1], border[:, 2], c='k')
    return



def test_density_and_velocity_deposition():
    '''
    Moment deposition seems weighed fine.
    Boundary cell thing seems reversed: x boundary stuff goes to y boundary? WTF?
    Weights and nodes themselves are calculated right
    '''
    NX   = const.NX  ; NY   = const.NY 
    xmax = const.xmax; ymax = const.ymax
    dx   = const.dx  ; dy   = const.dy

    cells_x = np.arange(NX + 3)*dx
    cells_y = np.arange(NY + 3)*dy
    EX, EY  = np.meshgrid((cells_x - 0.5*dx), (cells_y - 0.5*dy))
        
    q_dens  = np.zeros((NX + 3, NY + 3),       dtype=np.float64)  
    Ji      = np.zeros((NX + 3, NY + 3, 3),    dtype=np.float64)
    ni      = np.zeros((NX + 3, NY + 3, 1),    dtype=np.float64)
    nu      = np.zeros((NX + 3, NY + 3, 1, 3), dtype=np.float64)
    
    pos = np.array([5.0*xmax/NX, 1.0*ymax/NY]).reshape((2, 1))
    vel = np.array([1, 2, 4]).reshape((3, 1))
    
    Ie  = np.zeros(pos.shape, dtype=int)
    We  = np.zeros((2, 1, 3))
    idx = np.array([0]) 
    
    particles.assign_weighting_TSC(pos, Ie, We)
    
    #sources.deposit_moments_to_grid(vel, Ie, We, idx, ni, nu)
    sources.collect_moments(vel, Ie, We, idx, q_dens, Ji, ni, nu)

    # Normalize contribution (since we just care about positions/weights)
    q_dens /= (const.q * const.n_contr[idx[0]])
    Ji     /= (const.q * const.n_contr[idx[0]])

    if True:
        fig = plt.figure(figsize=(15, 15))
        ax  = fig.add_subplot(111)
        ################
        ## GRID SPACE ##
        ################
        ax.scatter(EX, EY, c='r', marker='x', s=20)     # E node locations
        
        ax.set_xlabel('x direction')
        ax.set_ylabel('y direction')
        ax.set_xlim(-1.5*dx, xmax + 1.5*dx)
        ax.set_ylim(-1.5*dy, ymax + 1.5*dy)
        
        border = np.array([ [0   , 0   , 0],
                            [0   , ymax, 0],
                            [xmax, ymax, 0],
                            [xmax, 0   , 0],
                            [0   , 0   , 0]])
        ax.plot(border[:, 0], border[:, 1], border[:, 2], c='k', linestyle='--')
        
        for mm in range(const.NX):
            plt.vlines(mm * dx, 0, ymax, linestyles='--', alpha=0.20)
        
        for nn in range(const.NY):
            plt.hlines(nn * dy, 0, xmax, linestyles='--', alpha=0.20)
            
        #############
        #############
        #im = ax.pcolormesh(cells_x - 1.0*dx, cells_y - 1.0*dy, q_dens[:, :].T, alpha=0.9, cmap='Greens')
        im = ax.pcolormesh(cells_x - 1.0*dx, cells_y - 1.0*dy, Ji[:, :, 2].T, alpha=0.9, cmap='Greens')
        ax.scatter(pos[0], pos[1], c='k', marker='o', s=40)
        
        plt.colorbar(im)
    return




def test_curl_B():
    NX   = const.NX  ; NY   = const.NY 
    xmax = const.xmax; ymax = const.ymax
    dx   = const.dx  ; dy   = const.dy
    kx   = 2 * np.pi / xmax
    ky   = 2 * np.pi / ymax

    cells_x = np.arange(const.NX + 3)*dx
    cells_y = np.arange(const.NY + 3)*dy
    
    EX, EY = np.meshgrid(cells_x - 0.5*dx, cells_y - 0.5*dy)
    BX, BY = np.meshgrid(cells_x - 1.0*dx, cells_y - 1.0*dy)

    B          = np.zeros((NX + 3, NY + 3, 3))
    B[:, :, 0] = (np.cos(1.0*kx*BX)*np.sin(1.0*ky*BY)).T
    B[:, :, 1] = (np.cos(1.0*kx*BX)*np.sin(1.0*ky*BY)).T
    B[:, :, 2] = (np.cos(1.0*kx*BX)*np.sin(1.0*ky*BY)).T
    
    dB          = np.zeros((NX + 3, NY + 3, 3))
    dB[:, :, 0] = (-kx * np.sin(kx*BX) * np.sin(ky*BY)).T
    dB[:, :, 1] = (ky * np.cos(kx*BX) * np.cos(ky*BY)).T
    dB[:, :, 2] =  0
    
    curl_B_compute = np.zeros(B.shape)
    fields.curl_B_term(B, curl_B_compute)
    
    curl_B_analytic = np.zeros(B.shape)
    curl_B_analytic[:, :, 0] =  dB[:, :, 1] #/ xmax
    curl_B_analytic[:, :, 1] = -dB[:, :, 0] #/ ymax
    curl_B_analytic[:, :, 2] =  dB[:, :, 0] - dB[:, :, 1]
    curl_B_analytic /= const.mu0

    
    if True:
        # plot with 2x1D arrays: cells_x - 1.0*dx, cells_y - 1.0*dy for E (-1.5 for B?)
        fig = plt.figure(figsize=(15, 15))
        ax  = fig.add_subplot(111, projection='3d')
        #ax.plot_wireframe(BX, BY, B[:, :, 0])
        ax.plot_wireframe(EX, EY, curl_B_compute[:, :, 0], color='r')
        ax.plot_wireframe(EX, EY, curl_B_analytic[:, :, 0], color='b')
        
        ## GRID SPACE ##
        #ax.scatter(EX, EY, c='r', marker='x', s=20)     # E node locations
        #ax.scatter(BX, BY, c='b', marker='x', s=20)     # B node locations
        
        ax.set_xlim(-1.5*dx, xmax + 1.5*dx)
        ax.set_ylim(-1.5*dy, ymax + 1.5*dy)
        
        border = np.array([ [0   , 0   , 0],
                            [0   , ymax, 0],
                            [xmax, ymax, 0],
                            [xmax, 0   , 0],
                            [0   , 0   , 0]])
        ax.plot(border[:, 0], border[:, 1], border[:, 2], c='k')
    return


def test_curl_E():
    NX   = 32   #const.NX

    xmin = 0.0  #const.xmin
    xmax = 2*np.pi#const.xmax
    
    dx   = xmax / NX
    x    = np.arange(xmin, xmax, dx/100.)              # Simulation domain space 0,NX (normalized to grid)
    k    = 1.0

    # Physical location of nodes
    E_nodes = (np.arange(NX + 3) - 0.5) * dx
    B_nodes = (np.arange(NX + 3) - 1.0) * dx

    Ex         =            np.cos(1.0*k*E_nodes)
    Ey         =            np.cos(1.5*k*E_nodes)
    Ez         =            np.cos(2.0*k*E_nodes)
    dEy        = -1.5 * k * np.sin(1.5*k*B_nodes)
    dEz        = -2.0 * k * np.sin(2.0*k*B_nodes)
    
    E_input       = np.zeros((NX + 3, 3))
    E_input[:, 0] = Ex
    E_input[:, 1] = Ey
    E_input[:, 2] = Ez

    curl_E_FD   = fields.get_curl_E(E_input, DX=dx)
    curl_E_anal = np.zeros((NX + 3, 3))
    curl_E_anal[:, 1] = -dEz
    curl_E_anal[:, 2] =  dEy
    
    plt.figure(figsize=(15, 15))
    marker_size = None

    #plt.plot(x, -deriv, linestyle=':', c='b', label='By Analytic Solution')
    plt.scatter(B_nodes, curl_E_anal[:, 1], marker='o', c='k', s=marker_size, label='By Node Solution')
    plt.scatter(B_nodes, curl_E_FD[:, 1], marker='x', c='b', s=marker_size, label='By Finite Difference')
    
    #plt.plot(x, deriv, linestyle=':', c='r', label='Bz Analytic Solution')   
    plt.scatter(B_nodes, curl_E_anal[:, 2], marker='o', c='k', s=marker_size, label='Bz Node Solution')
    plt.scatter(B_nodes, curl_E_FD[:, 2], marker='x', c='r', s=marker_size, label='Bz Finite Difference')   
    plt.title(r'Test of $\nabla \times E$')

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

def wireframe_test():
    
    A = np.zeros((20, 50), dtype=float) # Values at points (x, y)
    
    fig = plt.figure(figsize=(15, 15))
    ax  = fig.add_subplot(111)
    
    x_coords = np.arange(20)
    y_coords = np.arange(50)
    A[5, 25] = 1.0
    pos      = [5, 25]
    
    ax.pcolormesh(x_coords, y_coords, A.T)
    ax.scatter(pos[0], pos[1])
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    
    return


def test_B_grid_interpolation_and_boundaries():
    #B_grid_boundary_and_ghost_cells(temp3D)            # Set B-grid BC's and fill ghost cells
    #aux.linear_Bgrid_to_Egrid_scalar(temp3D, grad_P)   # Move grad_P back onto E_field grid

    NX   = const.NX  ; NY   = const.NY 
    xmax = const.xmax; ymax = const.ymax
    dx   = const.dx  ; dy   = const.dy

    kx   = 2 * np.pi / xmax
    ky   = 2 * np.pi / ymax
    
    cells_x = np.arange(NX + 3)*dx
    cells_y = np.arange(NY + 3)*dy
    
    EX, EY  = np.meshgrid((cells_x - 0.5*dx), (cells_y - 0.5*dy))
    BX, BY  = np.meshgrid((cells_x - 1.0*dx), (cells_y - 1.0*dy))

    B          = np.zeros((NX + 3, NY + 3, 3))
    B[:, :, 0] = (np.cos(1.0*kx*BX)*np.sin(1.0*ky*BY)).T
    B[:, :, 1] = (np.cos(1.0*kx*BX)*np.sin(1.0*ky*BY)).T
    B[:, :, 2] = (np.cos(1.0*kx*BX)*np.sin(1.0*ky*BY)).T

    if True:
        fig = plt.figure(figsize=(15, 15))
        ax  = fig.add_subplot(111, projection='3d')
        ################
        ## GRID SPACE ##
        ################
        ax.scatter(EX, EY, c='r', marker='x', s=20)     # E node locations
        ax.scatter(BX, BY, c='b', marker='x', s=20)     # E node locations
        
        ax.set_xlabel('x direction')
        ax.set_ylabel('y direction')
        ax.set_xlim(-1.5*dx, xmax + 1.5*dx)
        ax.set_ylim(-1.5*dy, ymax + 1.5*dy)
        
        border = np.array([ [0   , 0   , 0],
                            [0   , ymax, 0],
                            [xmax, ymax, 0],
                            [xmax, 0   , 0],
                            [0   , 0   , 0]])
        ax.plot(border[:, 0], border[:, 1], border[:, 2], c='k', linestyle='--')
        
        for mm in range(const.NX):
            plt.vlines(mm * dx, 0, ymax, linestyles='--', alpha=0.20)
        
        for nn in range(const.NY):
            plt.hlines(nn * dy, 0, xmax, linestyles='--', alpha=0.20)
            
        #############
        #############
        #pdb.set_trace()
        ax.plot_wireframe(cells_x, cells_y, B[:, :, 0].T, color='r')
        
        #im = ax.pcolormesh(cells_x - 1.0*dx, cells_y - 1.0*dy, Ji[:, :, 2].T, alpha=0.9, cmap='Greens')

    return

if __name__ == '__main__':
    #test_weight_conservation()
    #test_weight_correctness()
    test_curl_B()
    #test_ghost_cell_filling()
    #test_density_and_velocity_deposition()
    #test_B_grid_interpolation_and_boundaries()