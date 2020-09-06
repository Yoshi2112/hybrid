# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:42:13 2017

@author: iarey
"""
import math
import numpy as np
import numba as nb
import matplotlib        as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pdb

import simulation_parameters_1D as const
import particles_1D             as particles
import sources_1D               as sources
import fields_1D                as fields
import auxilliary_1D            as aux
import init_1D as init

from timeit import default_timer as timer

@nb.njit()
def roundup(x, nearest=10.):
    '''
    Rounds up x to the nearest multiple specified by 'nearest'
    Returns float
    '''
    return int(math.ceil(x / nearest)) * nearest


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


def plot_velocity_distribution_2D_histogram(pos, vel, suff=''):
    '''
    For each point in time
     - Collect particle information for particles near cell, plus time component
     - Store in array
     - Plot using either hexbin or hist2D
     
    Issue : Bins along v changing depending on time (how to set max/min bins? Specify arrays manually)
    '''
    lt = ['x', 'y', 'z']
    
    for jj in range(const.Nj):
        # Manually specify bin edges for histogram: Set vbins for 3*vth?
        vlim = 10 if const.temp_type[jj] == 1 else 5
        
        vbins = np.linspace(-vlim, vlim, 101, endpoint=True)
        xbins = np.linspace(const.xmin/const.dx, const.xmax/const.dx, const.NX + 1, endpoint=True)
            
        save_dir = const.drive + const.save_path + '//INIT//'
        filename = 'fv_vs_x_species_{}'.format(jj) + suff
        
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
    
        # Do the plotting
        plt.ioff()
        
        fig, axes = plt.subplots(3, figsize=(15, 10), sharex=True)
        axes[0].set_title('f(v) vs. x :: INIT :: {}'.format(const.species_lbl[jj]))
        
        st = const.idx_start[jj]
        en = const.idx_end[jj]
        
        for kk in range(3):
            counts, xedges, yedges, im1 = axes[kk].hist2d(pos[0, st:en]/const.dx, vel[kk, st:en]/const.va, 
                                                    bins=[xbins, vbins], vmin=0)
    
            cb = fig.colorbar(im1, ax=axes[kk], pad=0.015)
            cb.set_label('Counts')
            
            axes[kk].set_ylim(-vlim, vlim)
            axes[kk].set_ylabel('v{}\n($v_A$)'.format(lt[kk]), rotation=0)
            
        axes[kk].set_xlim(const.xmin/const.dx, const.xmax/const.dx)
        axes[kk].set_xlabel('Position (cell)')
    
        fig.subplots_adjust(hspace=0.065)
        
        plt.savefig(save_dir + filename, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close('all')
    return


def plot_initial_sources(q_dens, Ji, E, B):
    fig, ax = plt.subplots(4, 2, sharex=True)
    ax[0, 0].plot(const.E_nodes/const.dx, q_dens)
    ax[0, 0].set_ylabel('$\\rho_c$', rotation=0, fontsize=14, labelpad=20)
    
    ax[1, 0].plot(const.E_nodes/const.dx, Ji[:, 0])
    ax[1, 0].set_ylabel('$J_{ix}$', rotation=0, fontsize=14, labelpad=20)
    
    ax[2, 0].plot(const.E_nodes/const.dx, Ji[:, 1])
    ax[2, 0].set_ylabel('$J_{iy}$', rotation=0, fontsize=14, labelpad=20)
    
    ax[3, 0].plot(const.E_nodes/const.dx, Ji[:, 2])
    ax[3, 0].set_ylabel('$J_{iz}$', rotation=0, fontsize=14, labelpad=20)
    
    ax[0, 1].plot(const.E_nodes/const.dx, E[:, 0])
    ax[0, 1].set_ylabel('$E_x$', rotation=0, fontsize=14, labelpad=20)
    ax[1, 1].plot(const.E_nodes/const.dx, E[:, 1])
    ax[1, 1].set_ylabel('$E_y$', rotation=0, fontsize=14, labelpad=20)
    ax[2, 1].plot(const.E_nodes/const.dx, E[:, 1])
    ax[2, 1].set_ylabel('$E_z$', rotation=0, fontsize=14, labelpad=20)
    
    for mm in range(4):
        for nn in range(2):
            for ii in range(const.E_nodes.shape[0]):
                ax[mm, nn].axvline(const.E_nodes[ii]/const.dx, linestyle='--', c='r', alpha=0.2)
                ax[mm, nn].axvline(const.B_nodes[ii]/const.dx, linestyle='--', c='b', alpha=0.2)
             
            ax[mm, nn].axvline(const.xmin/const.dx, color='k')
            ax[mm, nn].axvline(const.xmax/const.dx, color='k')
            ax[mm, nn].axvline(const.B_nodes[ 0]/const.dx, linestyle='-', c='darkblue', alpha=1.0)
            ax[mm, nn].axvline(const.B_nodes[-1]/const.dx, linestyle='-', c='darkblue', alpha=1.0)
        
    fig.subplots_adjust(hspace=0)
    return


def plot_initial_positions_scatter(pos, highlight=None):
    '''
    Shows physical position of each particle on a scatterplot of (x, 1), as well
    as the E, B grids and boundaries.
    '''
    fig, ax = plt.subplots()

    for jj in range(const.Nj):
        st = const.idx_start[jj]
        en = const.idx_end[jj]
        ax.scatter(pos[0, st: en], np.ones(const.N_species[jj])*jj)
        
    if highlight is not None:
        for this in highlight:
            ax.scatter(pos[0,this], 1.0, c='k', marker='x')
        
    for ii in range(const.E_nodes.shape[0]):
        ax.axvline(const.E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        ax.axvline(const.B_nodes[ii], linestyle='--', c='b', alpha=0.2)
     
    ax.axvline(const.xmin, color='k')
    ax.axvline(const.xmax, color='k')
    ax.axvline(const.B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    ax.axvline(const.B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    return


def check_cell_velocity_distribution(pos, vel, node_number=const.NX // 2, j=0): #
    '''
    Checks the velocity distribution of a particle species within a specified cell
    '''
    # Account for damping nodes. Node_number should be "real" node count.
    node_number += const.ND
    x_node = const.E_nodes[node_number]
    f      = np.zeros((1, 3))
    
    count = 0
    for ii in np.arange(const.idx_start[j], const.idx_end[j]):
        if (abs(pos[0, ii] - x_node) <= 0.5*const.dx):
            f = np.append(f, [vel[0:3, ii]], axis=0)
            count += 1

    print('{} particles counted for diagnostic'.format(count))
    fig = plt.figure(figsize=(12,10))
    fig.suptitle('Particle velocity distribution of species {} in cell {}'.format(j, node_number))
    fig.patch.set_facecolor('w')
    num_bins = const.nsp_ppc[j] // 20

    ax_x = plt.subplot2grid((2, 3), (0,0), colspan=2, rowspan=2)
    ax_y = plt.subplot2grid((2, 3), (0,2))
    ax_z = plt.subplot2grid((2, 3), (1,2))
    
    for ax in [ax_x, ax_y, ax_z]:
        ax.axvline(0, c='k', ls=':', alpha=0.25)

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


def check_cell_velocity_distribution_2D(pos, vel, node_number=const.NX // 2, jj=0, show_cone=False, save=False, qq=None): #
    '''
    Checks the velocity distribution of a particle species within a specified cell
    
    Starts at "cell 0" - i.e. only counts real cells. 
    Damping cell offset is handled by function
    '''
    if node_number is None:
        node_number = np.arange(const.NX)
        
    # Account for damping nodes. Node_number should be "real" node count.
    for node in node_number:
        print('Collecting particle info in cell {}'.format(node))
        node += const.ND
        x_node = const.E_nodes[node]
        f      = np.zeros((1, 3))
        
        count = 0
        for ii in np.arange(const.idx_start[jj], const.idx_end[jj]):
            if (abs(pos[0, ii] - x_node) <= 0.5*const.dx):
                f = np.append(f, [vel[0:3, ii]], axis=0)
                count += 1
    
        plt.ioff()
        #print('{} particles counted for diagnostic'.format(count))
        fig, ax1 = plt.subplots(figsize=(15,10))
        fig.suptitle('Particle velocity distribution of species {} in cell {}'.format(jj, node))
        fig.patch.set_facecolor('w')
    
        V_PERP = np.sign(f[:, 2]) * np.sqrt(f[:, 1] ** 2 + f[:, 2] ** 2) / const.va
        V_PARA = f[:, 0] / const.va
    
        ax1.scatter(V_PERP, V_PARA, s=1, c=const.temp_color[jj])
    
        ax1.set_ylabel('$v_\parallel (/v_A)$')
        ax1.set_xlabel('$v_\perp (/v_A)$')
        ax1.axis('equal')
        
        # Plot loss cone lines 
        B_at_nodes     = fields.eval_B0x(np.array([x_node + const.dx, x_node - const.dx]))
        end_loss_cones = np.arcsin(np.sqrt(B_at_nodes/const.B_A)).max()                    # Calculate max loss cone
        
        yst, yend = ax1.get_ylim()                  # Get V_PARA lims
        yarr      = np.linspace(yst, yend, 100)     # V_PARA values on axis
        xarr      = yarr * np.tan(end_loss_cones)   # Calculate V_PERP values
        ax1.plot(xarr,  yarr, c='k', label='Loss Cone: {:.1f} deg'.format(end_loss_cones * 180. / np.pi))
        ax1.plot(xarr, -yarr, c='k')
        ax1.legend()
        ax1.set_xlim(yst, yend)
        ax1.set_ylim(yst, yend)
        
        if save == True:
            save_path = const.drive + '//' + const.save_path + '//' + 'run_{}//LCD_plot_sp_{}//'.format(const.run, jj)
            
            if os.path.exists(save_path) == False:
                os.makedirs(save_path)
                
            if ii is None:
                fig.savefig(save_path + 'LCD_cell_{:04}_sp{}.png'.format(node, jj))
            else:
                fig.savefig(save_path + 'LCD_cell_{:04}_sp{}_{:08}.png'.format(node, jj, qq))
            plt.close('all')
        else:
            plt.show()
    return


def check_position_distribution(pos, num_bins=None):
    '''Checks the spatial distribution of a particle species j within the spatial domain
    
    Called by main_1D()
    '''
    for j in range(const.Nj):
        fig = plt.figure(figsize=(12,10))
        fig.suptitle('Configuration space distribution of {}'.format(const.species_lbl[j]))
        fig.patch.set_facecolor('w')
        
        if num_bins is None:
            num_bins = const.NX
    
        ax_x = plt.subplot()
    
        xs, BinEdgesx = np.histogram(pos[0, const.idx_start[j]: const.idx_end[j]], bins=num_bins)
        bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
        ax_x.plot(bx, xs, '-', c=const.temp_color[const.temp_type[j]], drawstyle='steps')
        ax_x.set_xlabel(r'$x_p (m)$')
        ax_x.set_xlim(const.xmin, const.xmax)

    plt.show()
    return

def check_velocity_distribution(vel):
    '''Checks the velocity distribution of an entire species across the simulation domain
    '''
    for j in range(const.Nj):
        fig = plt.figure(figsize=(12,10))
        fig.suptitle('Velocity distribution of species {} in simulation domain'.format(j))
        fig.patch.set_facecolor('w')
        num_bins = const.nsp_ppc[j] // 5
    
        ax_x = plt.subplot2grid((2, 3), (0,0), colspan=2, rowspan=2)
        ax_y = plt.subplot2grid((2, 3), (0,2))
        ax_z = plt.subplot2grid((2, 3), (1,2))
    
        xs, BinEdgesx = np.histogram(vel[0, const.idx_start[j]: const.idx_end[j]] / const.va, bins=num_bins)
        bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
        ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
        ax_x.set_xlabel(r'$v_x / v_A$')
    
        ys, BinEdgesy = np.histogram(vel[1, const.idx_start[j]: const.idx_end[j]] / const.va, bins=num_bins)
        by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
        ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
        ax_y.set_xlabel(r'$v_y / v_A$')
    
        zs, BinEdgesz = np.histogram(vel[2, const.idx_start[j]: const.idx_end[j]] / const.va, bins=num_bins)
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
    # Use the sim_params with only one species
    POS, VEL, IE, W_ELEC, IB, W_MAG, IDX  = init.initialize_particles()
    Q_DENS, Q_DENS_ADV, JI, NI, NU        = init.initialize_source_arrays()
    temp1D                                = np.zeros(const.NC, dtype=np.float64) 
    
    sources.collect_moments(VEL, IE, W_ELEC, IDX, Q_DENS, JI, NI, NU, temp1D) 
    
    if False:
        # Two species
        ypos = np.ones(POS.shape[1] // 2)
        
        plt.scatter(POS[0, :POS.shape[1] // 2 ], ypos + 0.1, c='r')
        plt.scatter(POS[0,  POS.shape[1] // 2:], ypos + 0.2, c='b')
    else:
        # One species
        ypos = np.ones(POS.shape[1])
        plt.scatter(POS[0], ypos + 0.1, c='b')
        
    # Plot charge density
    plt.plot(const.E_nodes, Q_DENS / Q_DENS.max(), marker='o')
        
    for ii in range(const.E_nodes.shape[0]):
        plt.axvline(const.E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(const.B_nodes[ii], linestyle='--', c='b', alpha=0.2)
     
    plt.axvline(const.xmin, color='k')
    plt.axvline(const.xmax, color='k')
    plt.axvline(const.B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(const.B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    return


def check_density_deposition():
    # Change dx to 1 and NX/ND to reasonable values to make this nice
    # Don't forget to comment out the density floor AND re-enable it when done.
    #positions   = np.array([2.00])
    dy = 0.5
    positions = np.arange(-const.NX/2, const.NX/2 + dy, dy)

    Np  = positions.shape[0]
    pos = np.zeros((3, Np))
    
    for ii in range(Np):
        pos[0, ii] = positions[ii]
    
    vel        = np.zeros((3, Np))
    idx        = np.zeros(Np, dtype=np.uint8)
    Ie         = np.zeros(Np, dtype=np.uint16)
    W_elec     = np.zeros((3, Np), dtype=np.float64)
    
    q_dens, q_dens_adv, Ji, ni, nu = init.initialize_source_arrays()
    temp1D                         = np.zeros(const.NC, dtype=np.float64) 
    
    particles.assign_weighting_TSC(pos, Ie, W_elec)
    sources.collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu, temp1D) 
    
    # Plot normalized charge density
    q_dens /= (const.q * const.n_contr[0])
    #pdb.set_trace()
    #print(q_dens.sum())
    ypos = np.ones(Np) * q_dens.max() * 1.1
    plt.plot(const.E_nodes, q_dens, marker='o')
    plt.scatter(pos[0], ypos, marker='x', c='k')
        
    for ii in range(const.E_nodes.shape[0]):
        plt.axvline(const.E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(const.B_nodes[ii], linestyle='--', c='b', alpha=0.2)
     
    plt.axvline(const.xmin, linestyle=':', c='k'       , alpha=0.7)
    plt.axvline(const.xmax, linestyle=':', c='k'       , alpha=0.7)
    plt.axvline(const.B_nodes[ 0], linestyle='-', c='darkblue', alpha=0.7)
    plt.axvline(const.B_nodes[-1], linestyle='-', c='darkblue', alpha=0.7)
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
    NC   = const.NC   
    k    = 2 * np.pi / (2 * const.xmax)

    # Inputs and analytic solutions
    B_input       = np.zeros((NC + 1, 3))
    B_input[:, 0] = np.cos(1.0*k*const.B_nodes)
    B_input[:, 1] = np.cos(1.5*k*const.B_nodes)
    B_input[:, 2] = np.cos(2.0*k*const.B_nodes)
    
    curl_B_anal       = np.zeros((NC, 3))
    curl_B_anal[:, 1] =  2.0 * k * np.sin(2.0*k*const.E_nodes)
    curl_B_anal[:, 2] = -1.5 * k * np.sin(1.5*k*const.E_nodes)
    
    curl_B_FD = np.zeros((NC, 3))
    fields.curl_B_term(B_input, curl_B_FD)
    
    curl_B_FD *= const.mu0
    
    ## DO THE PLOTTING ##
    plt.figure(figsize=(15, 15))
    marker_size = None
        
    plt.scatter(const.E_nodes, curl_B_anal[:, 1], marker='o', c='k', s=marker_size, label='By Node Solution')
    plt.scatter(const.E_nodes, curl_B_FD[  :, 1], marker='x', c='b', s=marker_size, label='By Finite Difference')
      
    plt.scatter(const.E_nodes, curl_B_anal[:, 2], marker='o', c='k', s=marker_size, label='Bz Node Solution')
    plt.scatter(const.E_nodes, curl_B_FD[  :, 2], marker='x', c='r', s=marker_size, label='Bz Finite Difference')   
    plt.title(r'Test of $\nabla \times B$')

    for ii in range(const.E_nodes.shape[0]):
        plt.axvline(const.E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(const.B_nodes[ii], linestyle='--', c='b', alpha=0.2)
     
    plt.axvline(const.B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(const.B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    
    plt.axvline(const.xmin, linestyle=':', c='k', alpha=0.5)
    plt.axvline(const.xmax, linestyle=':', c='k', alpha=0.5)
    plt.xlabel('x (km)')
    
    plt.legend()
    return


def test_curl_E():
    '''
    Confirmed with parabolic B0 code
    '''
    NC   = const.NC   
    k    = 2 * np.pi / (2 * const.xmax)

    # Inputs and analytic solutions
    E_input       = np.zeros((NC, 3))
    E_input[:, 0] = np.cos(1.0*k*const.E_nodes)
    E_input[:, 1] = np.cos(1.5*k*const.E_nodes)
    E_input[:, 2] = np.cos(2.0*k*const.E_nodes)

    curl_E_FD = np.zeros((NC + 1, 3))
    fields.get_curl_E(E_input, curl_E_FD)
    
    curl_E_anal       = np.zeros((NC + 1, 3))
    curl_E_anal[:, 1] =  2.0 * k * np.sin(2.0*k*const.B_nodes)
    curl_E_anal[:, 2] = -1.5 * k * np.sin(1.5*k*const.B_nodes)
    
    
    ## PLOT
    plt.figure(figsize=(15, 15))
    marker_size = None

    if False:
        plt.scatter(const.E_nodes, E_input[:, 0], marker='o', c='k', s=marker_size, label='Ex Node Solution')
        plt.scatter(const.E_nodes, E_input[:, 1], marker='o', c='k', s=marker_size, label='Ey Node Solution')
        plt.scatter(const.E_nodes, E_input[:, 2], marker='o', c='k', s=marker_size, label='Ez Node Solution')

    if True:
        plt.scatter(const.B_nodes, curl_E_anal[:, 1], marker='o', c='k', s=marker_size, label='By Node Solution')
        plt.scatter(const.B_nodes, curl_E_FD[  :, 1], marker='x', c='b', s=marker_size, label='By Finite Difference')
          
        plt.scatter(const.B_nodes, curl_E_anal[:, 2], marker='o', c='k', s=marker_size, label='Bz Node Solution')
        plt.scatter(const.B_nodes, curl_E_FD[  :, 2], marker='x', c='r', s=marker_size, label='Bz Finite Difference')   
    
    plt.title(r'Test of $\nabla \times E$')

    ## Add node markers and boundaries
    for kk in range(NC):
        plt.axvline(const.E_nodes[kk], linestyle='--', c='r', alpha=0.2)
        plt.axvline(const.B_nodes[kk], linestyle='--', c='b', alpha=0.2)
    
    plt.axvline(const.B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(const.B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    
    plt.axvline(const.xmin, linestyle='-', c='k', alpha=0.2)
    plt.axvline(const.xmax, linestyle='-', c='k', alpha=0.2)
    
    plt.legend()
    return


def test_grad_P():
    '''
    Verified for parabolic B0 :: Analytic solutions are a pain, but these come
    out looking sinusoidal as expected
    '''
    k    = 2 * np.pi / (2 * const.xmax)
    
    # Set analytic solutions (input/output)
    if False:
        q_dens    = np.cos(1.0 * k * const.E_nodes)  * const.q * const.ne
        te_input  = np.ones(const.NC)*const.Te0_scalar
    elif False:
        q_dens    = np.ones(const.NC) * const.q * const.ne
        te_input  = np.cos(1.0 * k * const.E_nodes)*const.Te0_scalar
    else:
        q_dens    = np.cos(1.0 * k * const.E_nodes)* const.q * const.ne
        te_input  = np.cos(1.0 * k * const.E_nodes)*const.Te0_scalar
    
    gp_diff   = np.zeros(const.NC)
    temp      = np.zeros(const.NC + 1)
    
    # Finite differences
    fields.get_grad_P(q_dens, te_input, gp_diff, temp)

    ## PLOT ##
    plt.figure(figsize=(15, 15))
    marker_size = None

    plt.scatter(const.E_nodes, gp_diff, marker='x', c='r', s=marker_size, label='Finite Difference')
    
    plt.title(r'Test of $\nabla p_e$')

    for kk in range(const.NC):
        plt.axvline(const.E_nodes[kk], linestyle='--', c='r', alpha=0.2)
        plt.axvline(const.B_nodes[kk], linestyle='--', c='b', alpha=0.2)
    
    plt.axvline(const.B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(const.B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    
    plt.axvline(const.xmin, linestyle='-', c='k', alpha=0.2)
    plt.axvline(const.xmax, linestyle='-', c='k', alpha=0.2)
    
    plt.legend()
    return


def test_grad_P_with_init_loading():
    '''
    Init density is homogenous :: Confirmed
    
    '''
    pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, temp_N = init.initialize_particles()
    q_dens, q_dens2, Ji, ni, nu                          = init.initialize_source_arrays()
    
    sources.collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu)
    
    te_input  = np.ones(const.NC)*const.Te0_scalar
    gp_diff   = np.zeros(const.NC)
    temp      = np.zeros(const.NC + 1)
    
    fields.get_grad_P(q_dens, te_input, gp_diff, temp)
    
    fig, ax = plt.subplots(4, figsize=(15, 10), sharex=True)
    ax[0].set_title('Particle Moments')
    ax[0].plot(const.E_nodes, q_dens)
    ax[1].plot(const.E_nodes, Ji[:, 0])
    ax[2].plot(const.E_nodes, Ji[:, 1])
    ax[3].plot(const.E_nodes, Ji[:, 2])
    
    ax[0].set_ylabel(r'$\rho_c$')
    ax[1].set_ylabel(r'$J_x$')
    ax[2].set_ylabel(r'$J_y$')
    ax[3].set_ylabel(r'$J_z$')
    
    fig2, ax2 = plt.subplots(figsize=(15, 10))
    
    ax2.set_title('E-field contribution of initial density loading')
    ax2.plot(const.E_nodes, gp_diff / q_dens)
    
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

    test_result = np.zeros(A.shape)
    aux.cross_product(A, B, test_result)
    
    plt.plot(anal_result[:, 0], marker='o')
    plt.plot(test_result[:, 0], marker='x')
    plt.show()
# =============================================================================
#     diff        = test_result - anal_result
#     print(diff)
# =============================================================================

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
    Tests cubic spline interpolation
    '''
    k           = 1.0
    marker_size = 20

    # Interpolation
    B_input       = np.zeros((const.NC + 1, 3))
    B_input[:, 0] = np.cos(1.0*k*const.B_nodes)
    B_input[:, 1] = np.cos(2.0*k*const.B_nodes)
    B_input[:, 2] = np.cos(3.0*k*const.B_nodes)
    
    
    # Analytic solution
    Bxe        = np.cos(1.0*k*const.E_nodes)
    Bye        = np.cos(2.0*k*const.E_nodes)
    Bze        = np.cos(3.0*k*const.E_nodes)  
    
    B_center      = np.zeros((const.NC, 3))

    ## TEST INTERPOLATION ##
    B_center = aux.interpolate_edges_to_center(B_input, B_center)

    plot = True
    if plot == True:
        plt.figure()
        plt.scatter(const.B_nodes, B_input[:, 0], s=marker_size, c='b', marker='x')
        plt.scatter(const.B_nodes, B_input[:, 1], s=marker_size, c='b', marker='x')
        plt.scatter(const.B_nodes, B_input[:, 2], s=marker_size, c='b', marker='x')
        
        plt.scatter(const.E_nodes, Bxe, s=marker_size, c='k', marker='o')
        plt.scatter(const.E_nodes, Bye, s=marker_size, c='k', marker='o')
        plt.scatter(const.E_nodes, Bze, s=marker_size, c='k', marker='o')
        
        plt.scatter(const.E_nodes, B_center[:, 0], s=marker_size, c='r', marker='x')
        plt.scatter(const.E_nodes, B_center[:, 1], s=marker_size, c='r', marker='x')
        plt.scatter(const.E_nodes, B_center[:, 2], s=marker_size, c='r', marker='x')
        
        for kk in range(const.NC):
            plt.axvline(const.E_nodes[kk], linestyle='--', c='r', alpha=0.2)
            plt.axvline(const.B_nodes[kk], linestyle='--', c='b', alpha=0.2)
        plt.axvline(const.B_nodes[kk+1], linestyle='--', c='b', alpha=0.2)
            
        plt.axvline(const.xmin, linestyle='-', c='k', alpha=0.2)
        plt.axvline(const.xmax, linestyle='-', c='k', alpha=0.2)
    
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



def plot_dipole_field_line(length=True, get_from_sim=True):
    '''
    Plots field lines with basic L = r*sin^2(theta) relation. Can plot
    multiple for all in Ls. Can also calculate arclengths from lat_st/lat_min
    and print as the title (can be changed if you want)
    '''
    
    dtheta     = 0.1
    theta      = np.arange(0, 180. + dtheta, dtheta) * np.pi / 180
        
    if get_from_sim == False:
        Ls     = [4.3]
        lat_st = 80  
        lat_en = 100
    else:
        Ls     = [const.L]
        lat_st = 90 - const.theta_xmax * 180./np.pi
        lat_en = 90 + const.theta_xmax * 180./np.pi
    
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
        plt.title('Arclength MLAT {} to {} deg at L = {} : {:>5.2f} R_E'.format(90 - lat_st, 90 + lat_en, L, length))
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
    B_surf    = 3.12e-5    # Magnetic field strength at Earth surface
    L         = 5.35       # Field line L shell
    dtheta    = 0.01       # Angle increment
    
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
    
    else:
    
        # Calculate cylindrical/parabolic approximation between lat st/en
        lat_width = 30         # Latitudinal width (from equator)

        st, en = boundary_idx64(theta * 180 / np.pi, 90 - lat_width, 90 + lat_width)
        
        length = 0
        for ii in range(st, en):
            length += r[ii] * dtheta * np.pi / 180
        
        RE   = 1.0
        sfac = 1.1
        z    = np.linspace(-length/2, length/2, en - st, endpoint=True) * RE
        a    = sfac / (L * RE)
        B0_z = B_mu.min() * (1 + a * z ** 2)
        
        print('Domain length : {:5.2f}RE'.format(length))
        print('Minimum field : {:5.2f}nT'.format(B_mu.min()))
        print('Maximum field : {:5.2f}nT'.format(B_mu[st:en].max()))
        print('Max/Min ratio : {:5.2f}'.format(B_mu[st:en].max() / B_mu.min()))
        
        plt.figure(2)
        plt.scatter(z/RE, B0_z,        label='Cylindrical approximation', s=4)
        plt.scatter(z/RE, B_mu[st:en], label='Dipole field intensity', s=4)
        plt.title(r'Approximation for $a = \frac{%.1f}{LR_E}$, lat. width %s deg' % (sfac, lat_width), fontsize=18)
        plt.xlabel(r'z ($R_E$)',     rotation=0, fontsize=14)
        plt.ylabel(r'$B_\parallel$', rotation=0, fontsize=14)
        plt.legend()
    return


def test_boris():
    '''
    Contains stripped down version of the particle-push (velocity/position updates) loop
    in order to test accuracy of boris pusher.
    '''
    B0        = const.B_xmax
    v0_perp   = const.va
    
    gyfreq    = const.gyfreq / (2 * np.pi)
    orbit_res = 0.05
    max_rev   = 1000
    dt        = orbit_res / gyfreq 
    max_t     = max_rev   / gyfreq
    max_inc   = int(max_t / dt)
    
    pos       = np.array([0])
    vel       = np.array([const.va, const.va, const.va]).reshape((3, 1))
    idx       = np.array([0])
    
    # Dummy arrays so the functions work
    E       = np.zeros((const.NC    , 3))
    B       = np.zeros((const.NC + 1, 3))
    
    W_mag   = np.array([0, 1, 0]).reshape((3, 1))
    W_elec  = np.array([0, 1, 0]).reshape((3, 1))
    Ie      = np.zeros(pos.shape[0], dtype=int)
    Ib      = np.zeros(pos.shape[0], dtype=int)
    
    pos_history = np.zeros( max_inc)
    vel_history = np.zeros((max_inc, 3))
    
    B[:, 0]+= B0; tt = 0; t_total = 0
    
    particles.assign_weighting_TSC(pos, Ie, W_elec)
    particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, -0.5*dt)
    while tt < max_inc:
        pos_history[tt] = pos
        vel_history[tt] = vel[:, 0]
        
        particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, dt)
        particles.position_update(pos, vel, dt, Ie, W_elec)  
        
        if pos[0] < const.xmin:
            pos[0] += const.xmax
        elif pos[0] > const.xmax:
            pos[0] -= const.xmax
                
        tt      += 1
        t_total += dt
    
    if True:
        time = np.array([ii * dt for ii in range(max_inc)])
        plt.plot(time, vel_history[:, 0], marker='o', label=r'$v_x$')
        plt.plot(time, vel_history[:, 1], marker='o', label=r'$v_y$')
        plt.plot(time, vel_history[:, 2], marker='o', label=r'$v_z$')
        plt.title('Particle gyromotion /w Boris Pusher: B0 = {:.1f}nT, v0y = {:.0f}km/s, $T_c$ = {:5.3f}s'.format(B0*1e9, v0_perp*1e-3, 1/gyfreq))
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
    elif False:
        plt.plot(vel_history[max_inc //2:, 1], vel_history[max_inc //2:, 2])
        plt.title('Particle gyromotion /w Boris Pusher: B0 = {:.1f}nT, v0y = {:.0f}km/s, $T_c$ = {:5.3f}s'.format(B0*1e9, v0_perp*1e-3, 1/gyfreq))
        plt.xlabel('y-Velocity (m/s)')
        plt.ylabel('z-Velocity (m/s)')
        plt.axis('equal')
    elif False:
        eval_gyperiods = 5
        eval_idxs      = int(eval_gyperiods / orbit_res) + 1
        plt.plot(vel_history[:eval_idxs, 1], vel_history[:eval_idxs, 2], label='First {}'.format(eval_gyperiods))
        plt.plot(vel_history[eval_idxs:, 1], vel_history[eval_idxs:, 2], label=' Last {}'.format(eval_gyperiods))
        plt.title('Particle gyromotion /w Boris Pusher: First and Last {} gyroperiods'.format(eval_gyperiods))
        plt.xlabel('y-Velocity (m/s)')
        plt.ylabel('z-Velocity (m/s)')
        plt.axis('equal')
    return


def get_atan(y, x):
    
    if x > 0:
        v=np.arctan(y/x)

    if y >= 0 and x < 0:
        v = np.pi + np.arctan(y/x)

    if y < 0 and x < 0:
        v = -np.pi + np.arctan(y/x)

    if y > 0 and x == 0:
        v = np.pi/2

    if y < 0 and x == 0:
        v = -np.pi/2

    if v < 0:
        v = v + 2*np.pi

    return v


def do_particle_run(max_rev=50, v_mag=1.0, pitch=45.0, dt_mult=1.0):
    '''
    Contains full particle pusher including timestep checker to simulate the motion
    of a single particle in the analytic magnetic field field. No wave magnetic/electric
    fields present.
    '''    
    # Initialize arrays (W/I dummy - not used)
    
    # Particle index 12804 :: lval 40
    # Init position: [-1020214.38977955,  -100874.673573  ,        0.        ]
    # Init velocity: [ -170840.94864185, -8695629.67092295,  3474619.54765129]
    
    Np     = 500000
    print('Simulating {} particles for {} gyroperiods'.format(Np, max_rev))
    idx    = np.zeros(Np, dtype=int)
    
    W_elec = np.zeros((3, Np))
    W_mag  = np.zeros((3, Np))
    Ep     = np.zeros((3, Np))
    Bp     = np.zeros((3, Np))
    v_prime= np.zeros((3, Np))
    S      = np.zeros((3, Np))
    T      = np.zeros((3, Np))
    Ie     = np.zeros(Np, dtype=int)
    Ib     = np.zeros(Np, dtype=int)
    qmi    = np.zeros(Np)
    B_test = np.zeros((const.NC + 1, 3), dtype=np.float64) 
    E_test = np.zeros((const.NC, 3),     dtype=np.float64) 
    
    vel       = np.zeros((3, Np), dtype=np.float64)
    pos       = np.zeros((3, Np), dtype=np.float64)
    
    if False:
        # Load particle config from file
        print('Loading parameters from simulation run...')
        fpath     = 'F:/runs/validation_runs/run_0/extracted/lost_particle_info.npz'
        data      = np.load(fpath)
        
        #lval      = data['lval']
        lost_pos  = data['lost_pos']
        lost_vel  = data['lost_vel']
        #lost_idx  = data['lost_idx']
        
        lost_ii   = 0
        
        vel[:, 0] = lost_vel[0, :, lost_ii]
        pos[:, 0] = lost_pos[0, :, lost_ii]

        pdb.set_trace()
    else:
        # Set initial velocity based on pitch angle and particle energy
        v_par  = v_mag * const.va * np.cos(pitch * np.pi / 180.)
        v_perp = v_mag * const.va * np.sin(pitch * np.pi / 180.)
        
        #initial_gyrophase  = 270            # degrees
        #initial_gyrophase *= np.pi / 180.   # convert to radians
        
        vel[0, :] =  v_par
        vel[1, :] = 0.0  #- v_perp * np.sin(initial_gyrophase)
        vel[2, :] = -v_perp  #v_perp * np.cos(initial_gyrophase)
                
        rL = const.mp * v_perp / (const.q * const.B_eq)
        
        pos[0, :] = 0.0  
        pos[1, :] = rL   #rL * np.cos(initial_gyrophase)
        pos[2, :] = 0.0  #rL * np.sin(initial_gyrophase)

    # Initial quantities
    init_pos = pos.copy() 
    init_vel = vel.copy()
    gyfreq   = const.gyfreq / (2 * np.pi)
    ion_ts   = const.orbit_res / gyfreq
    vel_ts   = 0.5 * const.dx / np.max(np.abs(vel[0, :]))

    DT       = min(ion_ts, vel_ts) * dt_mult
    
    # Target: 25000 cyclotron periods (~1hrs)
    max_t    = max_rev / gyfreq
    max_inc  = int(max_t / DT) + 1

    time        = None#np.zeros((max_inc))
    pos_history = None#np.zeros((max_inc, Np, 3))
    vel_history = None#np.zeros((max_inc, Np, 3))
    mag_history = None#np.zeros((max_inc, 3))
    pos_gphase  = np.zeros((max_inc - 1))
    vel_gphase  = np.zeros((max_inc - 1))

    # Retard velocity for stability
    particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B_test, E_test, v_prime, S, T, qmi, -0.5*DT)
    
    # Record initial values
# =============================================================================
#     time[       0      ] = 0.                      # t = 0
#     pos_history[0, :, :] = pos[:, :].T             # t = 0
#     vel_history[0, :, :] = vel[:, :].T             # t = -0.5
# =============================================================================

    tt = 0; t_total = 0
    start_time = timer()
    while tt < max_inc - 1:
# =============================================================================
#         pos_gphase[tt] = get_atan(pos[2,0], pos[1,0]) * 180. / np.pi
#         vel_gphase[tt] = (get_atan(vel[2,0], vel[1,0]) * 180. / np.pi + 90.)%360.
#         print('P/V Gyrophase :: {:.2f}, {:.2f}'.format(pos_gphase[tt], vel_gphase[tt]))
# =============================================================================
        
        # Increment so first loop is at t = 1*DT
        tt      += 1
        t_total += DT
        
        particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B_test, E_test, v_prime, S, T, qmi, DT)
        particles.position_update(pos, vel, idx, DT, Ie, W_elec)
        
# =============================================================================
#         time[         tt]       = t_total
#         pos_history[  tt, :, :] = pos[:, :].T
#         vel_history[  tt, :, :] = vel[:, :].T
# =============================================================================
    runtime = timer() - start_time
    print('Particle push time : {}s'.format(runtime))
    return init_pos, init_vel, time, pos_history, vel_history, mag_history, DT, max_t, pos_gphase, vel_gphase


def straighten_out_soln(approx, exact):
    '''
    Use the sign of the approximation to work out what the sign of the exact value should be.
    Used on timeseries
    '''
    for jj in range(3):
        for ii in range(approx.shape[0]):
            if approx[ii, jj] < 0:
                exact[ii, jj] *= -1.0
    return


def velocity_update(pos, vel, dt):  
    '''
    Based on Appendix A of Ch5 : Hybrid Codes by Winske & Omidi.
    
    Cut down specifically for analytic B-field (no interpolation of wave fields,
    and no E-field)
    '''
    for ii in range(pos.shape[0]):
        qmi = const.q / const.mp
        vn  = vel[:, ii]
        
        B_p = particles.eval_B0_particle(pos[ii], vel[:, ii], qmi, np.array([0., 0., 0.]))
        
        # Intermediate calculations
        h  = qmi * dt
        f  = 1 - (h**2) / 2 * (B_p[0]**2 + B_p[1]**2 + B_p[2]**2)
        g  = h / 2 * (B_p[0]*vn[0] + B_p[1]*vn[1] + B_p[2]*vn[2])
    
        # Velocity push
        vel_x = f * vn[0] + h * (g * B_p[0] + (vn[1]*B_p[2] - vn[2]*B_p[1]) )
        vel_y = f * vn[1] + h * (g * B_p[1] - (vn[0]*B_p[2] - vn[2]*B_p[0]) )
        vel_z = f * vn[2] + h * (g * B_p[2] + (vn[0]*B_p[1] - vn[1]*B_p[0]) )
        
        vel[0, ii] = vel_x
        vel[1, ii] = vel_y
        vel[2, ii] = vel_z
    return B_p, B_p


def test_mirror_motion():
    '''
    Diagnostic code to call the particle pushing part of the hybrid and check
    that its solving ok. Runs with zero background E field and B field defined
    by the constant background field specified in the parameter script.
    '''
    max_rev = 1
    v_mags  = np.array([1.0])#, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], dtype=float)
    pitches = np.array([45.0])#, 50, 55, 60, 65, 70, 75, 80, 85, 90], dtype=float)
    
    mu_percentages = np.zeros((v_mags.shape[0], pitches.shape[0]), dtype=float)
    
    for ii in range(v_mags.shape[0]):
        for jj in range(pitches.shape[0]):
            init_pos, init_vel, time, pos_history, vel_history, mag_history,\
            mag_history_exact, DT, max_t = do_particle_run(max_rev=max_rev, v_mag=v_mags[ii], pitch=pitches[jj])

            #mag_history_x = np.zeros(pos_history.shape[0])
        
            # Calculate parameter timeseries using recorded values
            init_vperp  = np.sqrt(init_vel[1] ** 2 + init_vel[2] ** 2)
            #init_vpara = init_vel[0]
            #init_KE    = 0.5 * const.mp * init_vel ** 2
            #init_pitch = np.arctan(init_vperp / init_vpara) * 180. / np.pi
            init_mu    = 0.5 * const.mp * init_vperp ** 2 / const.B_eq
            
            #vel_perp      = np.sqrt(vel_history[:, 1] ** 2 + vel_history[:, 2] ** 2)
            #vel_para      = vel_history[:, 0]
            #vel_perp      = np.sqrt(vel_history[:, 1] ** 2 + vel_history[:, 2] ** 2)
            #vel_magnitude = np.sqrt(vel_history[:, 0] ** 2 + vel_history[:, 1] ** 2 + vel_history[:, 2] ** 2)
            
            #B_para      = mag_history[:, 0]
            #B_perp      = np.sqrt(mag_history[:, 1] ** 2 + mag_history[:, 2] ** 2)
            #B_magnitude = np.sqrt(mag_history[:, 0] ** 2 + mag_history[:, 1] ** 2 + mag_history[:, 2] ** 2)
            
            KE_perp = 0.5 * const.mp * (vel_history[:, 1] ** 2 + vel_history[:, 2] ** 2)
            #KE_para = 0.5 * const.mp *  vel_history[:, 0] ** 2
            #KE_tot  = KE_para + KE_perp
            
            #mu_x = KE_perp / np.sqrt(mag_history_x ** 2 + mag_history[:, 1] ** 2 + mag_history[:, 2] ** 2)
            
            #mu          = KE_perp / B_magnitude
            #mu_percent  = (mu.max() - mu.min()) / init_mu * 100.
            
            #print('v_mag = {:5.2f} :: pitch = {:4.1f} :: delta_mu = {}'.format(v_mags[ii], pitches[jj], mu_percent))
            
            #mu_percentages[ii, jj] = mu_percent
    
    
    if True:
        # Check velocity timeseries to work out why its not smooth
        ## Plots velocity/mag timeseries ##
        fig, axes = plt.subplots(2, sharex=True)
        
        axes[0].plot(time, pos_history[:, 0]* 1e-3, label='x')
        axes[1].plot(time, vel_history[:, 0]* 1e-3, label='vx')

        
        axes[0].set_ylabel('v (km)')
        axes[0].set_xlabel('t (s)')
        #axes[0].set_title(r'Velocity/Magnetic Field at Particle, v0 = [%4.1f, %4.1f, %4.1f]km/s, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg' % (init_vel[0, 0], init_vel[1, 0], init_vel[2, 0], const.loss_cone, init_pitch))
        #axes[0].legend()

        #axes[1].legend()

        axes[1].set_xlim(0, None)
    
    if False:
        # Plot approximate solution vs. quadratic solutions for magnetic field
        # mag_history_y_exact = np.zeros((max_inc, 2), dtype=np.complex128)
        
        errors = np.abs(mag_history - mag_history_exact)
        
        fig, axes = plt.subplots(3, sharex=True)
        
        axes[0].set_title('Comparison of approximate vs. exact solutions for B0r components: Exact Driver')
        
        axes[0].plot(time, mag_history[      :, 1] * 1e9, label='By Approx'    , c='b', marker='o')
        axes[0].plot(time, mag_history_exact[:, 1] * 1e9, label='By Exact +ve' , c='r', marker='x')
        axes[0].set_ylabel('B0y (nT)', rotation=0, labelpad=20)
        axes[0].legend()
        
        axes[1].plot(time, mag_history[      :, 2] * 1e9, label='Bz Approx'    , c='b', marker='o')
        axes[1].plot(time, mag_history_exact[:, 2] * 1e9, label='Bz Exact +ve' , c='r', marker='x')
        axes[1].set_ylabel('B0z (nT)', rotation=0, labelpad=20)
        axes[1].legend()
        
        axes[2].plot(time, errors[:, 1] * 1e9, label='By' , c='b')
        axes[2].plot(time, errors[:, 2] * 1e9, label='Bz' , c='r')
        axes[2].set_ylabel('Abs. Error\n(nT)', rotation=0, labelpad=20)
        axes[2].legend()
        
        axes[2].set_xlabel('Time (s)')
        axes[2].set_xlim(0, time[-1])
        
        
    if False:
        # Plot approximate solution vs. quadratic solutions for magnetic field
        # mag_history_y_exact = np.zeros((max_inc, 2), dtype=np.complex128)
        
        errors = np.abs(mag_history - mag_history_exact)
        
        mu_exact = KE_perp / np.sqrt(mag_history_exact[:, 0] ** 2
                                   + mag_history_exact[:, 1] ** 2
                                   + mag_history_exact[:, 2] ** 2)
        
        mu_error = np.abs(mu - mu_exact)
        
        fig, axes = plt.subplots(2, sharex=True)
        
        axes[0].set_title('Comparison of approximate vs. exact solutions for B0r components: Approx. Driver')
        
        axes[0].plot(time, mu       * 1e10, label=r'$\mu$ Approx sol.' , c='b', marker='o')
        axes[0].plot(time, mu_exact * 1e10, label=r'$\mu$ Exact  sol.' , c='r', marker='x')
        axes[0].set_ylabel(r'$\mu (\times 10^{10})$', rotation=0, labelpad=40)
        axes[0].legend()
        
        axes[1].plot(time, mu_error * 1e10, label=r'$\mu$ error' , c='b')
        axes[1].set_ylabel(r'Abs. Error $(\times 10^{10})$', rotation=0, labelpad=40)
        axes[1].legend()
        
        axes[1].set_xlabel('Time (s)')
        axes[1].set_xlim(0, time[-1])
    
    
    if False:
        # Basic mu plot with v_perp, |B| also plotted
        fig, axes = plt.subplots(4, sharex=True)
        
        axes[0].plot(time, mu*1e10, label='$\mu(t_v)$', lw=0.5, c='k')
        
        axes[0].set_title(r'First Invariant $\mu$ for single trapped particle :: DT = %7.5fs :: Max $\delta \mu = $%6.4f%% :: $|v|$ = %.1f$v_A$ :: $\alpha_0$ = %.1f$^\circ$' % (DT, mu_percent, v_mag, pitch))
        axes[0].set_ylabel(r'$(\times 10^{-10})$', rotation=0, labelpad=30)
        axes[0].get_yaxis().get_major_formatter().set_useOffset(False)
        axes[0].axhline(init_mu*1e10, c='k', ls=':')
        
        axes[1].plot(time, vel_perp*1e-3, lw=0.5, c='k')
        axes[1].set_ylabel('$v_\perp$\n(km/s)', rotation=0, labelpad=20)
        
        axes[2].plot(time, pos_history*1e-3, lw=0.5, c='k')
        axes[2].set_ylabel('$x$\n(km)', rotation=0, labelpad=20)

        axes[3].plot(time, B_magnitude*1e9, lw=0.5, c='k')
        axes[3].set_ylabel('$|B|(t_v)$ (nT)', rotation=0, labelpad=20)

        axes[3].set_xlabel('Time (s)')
        axes[3].set_xlim(0, time[-1])
    
    if False:
        # Interpolate mag_x to half positions (where v is calculated)
        # The first value of mag_interp lines up with v[1]
        
        print('Minimum mu value: {}'.format(mu.min()))
        print('Maximum mu value: {}'.format(mu.max()))
        print('Percent change  : {}%%'.format(mu_percent))
        
        # Invariant (with parameters) timeseries
        fig, axes = plt.subplots(3, sharex=True)
        axes[0].plot(time,            mu*1e10, label='$\mu(t_v)$', lw=0.5, c='k')
        #axes[0].set_title(r'First Invariant $\mu$ for single trapped particle, v0 = [%3.1f, %3.1f, %3.1f]$v_{A,eq}^{-1}$, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg' % (init_vel[0, 0]/const.va, init_vel[1, 0]/const.va, init_vel[2, 0]/const.va, const.loss_cone, init_pitch))
        axes[0].set_title(r'First Invariant $\mu$ for single trapped particle :: TEST DT = %7.5fs :: Max $\delta \mu = $%6.4f%%' % (DT, mu_percent))
        axes[0].set_ylabel(r'$(\times 10^{-10})$', rotation=0, labelpad=30)
        axes[0].get_yaxis().get_major_formatter().set_useOffset(False)
        axes[0].axhline(init_mu*1e10, c='k', ls=':')
        axes[0].set_ylim(3.84, 3.99)
        # Check mu calculation against different sources of B: Source of oscillation? Nope
        mag_interp           = (mag_history_x[1:] + mag_history_x[:-1]) / 2
        mag_interp_magnitude = np.sqrt(mag_interp ** 2 + mag_history[1:, 1] ** 2 + mag_history[1:, 2] ** 2)
        
        mu_interp  = 0.5*const.mp*(vel_history[1:, 1] ** 2 + vel_history[1:, 2] ** 2) / mag_interp_magnitude

        axes[0].plot(time,          mu_x*1e10, label='$\mu(t_x)$')
        axes[0].plot(time[1:], mu_interp*1e10, label='$\mu(t_v)$ interpolated')
        axes[0].legend()
        
        axes[1].plot(time, vel_perp*1e-3, lw=0.5, c='k')
        axes[1].set_ylabel('$v_\perp$\n(km/s)', rotation=0, labelpad=20)

        axes[2].plot(time, B_magnitude*1e9, lw=0.5, c='k')
        axes[2].set_ylabel('$|B|(t_v)$ (nT)', rotation=0, labelpad=20)

        axes[2].set_xlabel('Time (s)')
        axes[2].set_xlim(0, time[-1])


    if False:
        ## Plots velocity/mag timeseries ##
        fig, axes = plt.subplots(2, sharex=True)
        
        axes[0].plot(time, vel_history[:, 1]* 1e-3, label='vy')
        axes[0].plot(time, vel_history[:, 2]* 1e-3, label='vz')
        axes[0].plot(time, vel_perp         * 1e-3, label='v_perp')
        axes[0].plot(time, vel_para*1e-3, label='v_para')
        
        axes[0].set_ylabel('v (km)')
        axes[0].set_xlabel('t (s)')
        axes[0].set_title(r'Velocity/Magnetic Field at Particle, v0 = [%4.1f, %4.1f, %4.1f]km/s, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg' % (init_vel[0, 0], init_vel[1, 0], init_vel[2, 0], const.loss_cone, init_pitch))
        #axes[0].set_xlim(0, None)
        axes[0].legend()
        
        axes[1].plot(time, B_magnitude,       label='|B0|')
        #axes[1].plot(time, mag_history[:, 0], label='B0x')
        #axes[1].plot(time, mag_history[:, 1], label='B0y')
        #axes[1].plot(time, mag_history[:, 2], label='B0z')
        axes[1].legend()
        axes[1].set_ylabel('t (s)')
        axes[1].set_ylabel('B (nT)')
        axes[1].set_xlim(0, None)
        
    if False:
        ## Plots 3-velocity timeseries ##
        fig, axes = plt.subplots(3, sharex=True)
        axes[0].set_title(r'Position/Velocity of Particle : v0 = [%3.1f, %3.1f, %3.1f]$v_A$, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg' % (init_vel[0, 0]/const.va, init_vel[1, 0]/const.va, init_vel[2, 0]/const.va, const.loss_cone, init_pitch))

        axes[0].plot(time, pos_history*1e-3)
        axes[0].set_xlabel('t (s)')
        axes[0].set_ylabel(r'x (km)')
        axes[0].axhline(const.xmin*1e-3, color='k', ls=':')
        axes[0].axhline(const.xmax*1e-3, color='k', ls=':')
        axes[0].legend()
        
        axes[1].plot(time, vel_history[:, 0]/const.va, label='$v_\parallel$')
        axes[1].plot(time,            vel_perp/const.va, label='$v_\perp$')
        axes[1].set_xlabel('t (s)')
        axes[1].set_ylabel(r'$v_\parallel$ ($v_{A,eq}^{-1}$)')
        axes[1].legend()
        
        axes[2].plot(time, vel_history[:, 1]/const.va, label='vy')
        axes[2].plot(time, vel_history[:, 2]/const.va, label='vz')
        axes[2].set_xlabel('t (s)')
        axes[2].set_ylabel(r'$v_\perp$ ($v_{A,eq}^{-1}$)')
        axes[2].legend()
        
        for ax in axes:
            ax.set_xlim(0, 100)
            
    if False:
        # Plot gyromotion of particle vx vs. vy
        plt.title('Particle gyromotion: {} gyroperiods ({:.1f}s)'.format(max_rev, max_t))
        plt.scatter(vel_history[:, 1], vel_history[:, 2], c=time)
        plt.colorbar().set_label('Time (s)')
        plt.ylabel('vy (km/s)')
        plt.xlabel('vz (km/s)')
        plt.axis('equal')
        
    if False:
        ## Plot parallel and perpendicular kinetic energies/velocities
        plt.figure()
        plt.title('Kinetic energy of single particle: Full Bottle')
        plt.plot(time, KE_para/const.q, c='b', label=r'$KE_\parallel$')
        plt.plot(time, KE_perp/const.q, c='r', label=r'$KE_\perp$')
        plt.plot(time, KE_tot /const.q, c='k', label=r'$KE_{total}$')
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
        plt.ylabel('Energy (eV)')
        plt.xlabel('Time (s)')
        plt.legend()
    
    if False:           
        percent = abs(KE_tot - init_KE.sum()) / init_KE.sum() * 100. 

        plt.figure()
        plt.title('Total kinetic energy change')

        plt.plot(time, percent*1e12)
        #plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        #plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
        #plt.ylim(-0.1e-3, 1e-3)
        plt.xlim(0, time[-1])
        plt.ylabel(r'Percent change ($\times 10^{-12}$)')
        plt.xlabel('Time (s)')
        
    if False:
        # Plots vx, v_perp vs. x - should be constant at any given x        
        fig, ax = plt.subplots(1)
        ax.set_title(r'Velocity vs. Space: v0 = [%4.1f, %4.1f, %4.1f]$v_{A,eq}^{-1}$ : %d gyroperiods (%5.2fs)' % (init_vel[0, 0], init_vel[1, 0], init_vel[2, 0], max_rev, max_t))
        ax.plot(pos_history*1e-3, vel_history[:, 0]*1e-3, c='b', label=r'$v_\parallel$')
        ax.plot(pos_history*1e-3, vel_perp,               c='r', label=r'$v_\perp$')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('v (km/s)')
        ax.set_xlim(const.xmin*1e-3, const.xmax*1e-3)
        ax.legend()

    if False:
        # Invariant and parameters vs. space
        fig, axes = plt.subplots(3, sharex=True)
        axes[0].plot(pos_history*1e-3, mu*1e10)
        axes[0].set_title(r'First Invariant $\mu$ for single trapped particle, v0 = [%3.1f, %3.1f, %3.1f]$v_{A,eq}^{-1}$, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg, $t_{max} = %5.0fs$' % (init_vel[0, 0]/const.va, init_vel[1, 0]/const.va, init_vel[2, 0]/const.va, const.loss_cone, init_pitch, max_t))
        axes[0].set_ylabel(r'$\mu (\times 10^{-10})$', rotation=0, labelpad=20)
        axes[0].get_yaxis().get_major_formatter().set_useOffset(False)
        axes[0].axhline(init_mu*1e10, c='k', ls=':')
        
        axes[1].plot(pos_history*1e-3, KE_perp/const.q)
        axes[1].set_ylabel(r'$KE_\perp (eV)$', rotation=0, labelpad=20)

        axes[2].plot(pos_history*1e-3, B_magnitude*1e9)
        axes[2].set_ylabel(r'$|B|$ (nT)', rotation=0, labelpad=20)
        
        axes[2].set_xlabel('Position (km)')
        axes[2].set_xlim(const.xmin*1e-3, const.xmax*1e-3)
        
    if False:
        fig, axes = plt.subplots(2, sharex=True)
        axes[0].plot(pos_history*1e-3, mag_history[:, 0] * 1e9, lw=0.25, c='k')
        axes[0].set_title(r'Magnetic fields taken at v, x times :: v0 = [%3.1f, %3.1f, %3.1f]$v_{A,eq}^{-1}$, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg, $t_{max} = %5.0fs$' % (init_vel[0, 0]/const.va, init_vel[1, 0]/const.va, init_vel[2, 0]/const.va, const.loss_cone, init_pitch, max_t))
        axes[0].set_ylabel(r'$B0_x(t_v) (nT)$', rotation=0, labelpad=20)
        
        axes[1].plot(pos_history*1e-3, mag_history_x * 1e9, lw=0.25, c='k')
        axes[1].set_ylabel(r'$B0_x(t_x) (nT)$', rotation=0, labelpad=20)
        
        axes[1].set_xlabel('Position (km)')
        #axes[1].set_xlim(-100, 100)
        
        #for ax in axes:
            #ax.set_ylim(200, 201)
    
    if False:
        ## Average out cyclotron motion for KE, B - see what effect that has on mu
        ## DT = 0.0155s
        ## Bounce period    ~ 83.0s
        ## Cyclotron period ~  0.1704s
        ## Smooth with a moving centered moving average of 0.2s
        import pandas as pd
        DT       = time[1] - time[0]
        win_size = int(0.2 / DT) + 1
        
        KE_pd = pd.Series(KE_perp)
        Bm_pd = pd.Series(B_magnitude)
        
        KE_av = KE_pd.rolling(center=True, window=win_size).mean()
        Bm_av = Bm_pd.rolling(center=True, window=win_size).mean()
        
        mu_average = KE_av / Bm_av
        
        fig, ax = plt.subplots(3, sharex=True)
        
        ax[0].set_title('Invariant Calculation :: Rolling average test :: window = {:5.3f}s, Tc = {:6.4f}s'.format(DT * win_size, 0.1704))
        
        ax[0].plot(time, mu        *1e10, label='$\mu$',     c='k', marker='o')
        ax[0].plot(time, mu_average*1e10, label='$\mu$ av.', c='r', marker='x', ms=1.0)
        ax[0].set_ylabel('$\mu$\n(J/T)', rotation=0, labelpad=20)
        ax[0].axhline(init_mu*1e10, c='k', ls=':')
        
        ax[1].plot(time, KE_perp/const.q, label='$KE_\perp$',     c='k', marker='o')
        ax[1].plot(time, KE_av/const.q,   label='$KE_\perp$ av.', c='r', marker='x', ms=1.0)
        ax[1].set_ylabel('$KE_\perp$\n(eV)', rotation=0, labelpad=20)
        
        ax[2].plot(time, B_magnitude*1e9, label='$|B|$',     c='k', marker='o')
        ax[2].plot(time, Bm_av*1e9,       label='$|B|$ av.', c='r', marker='x', ms=1.0)
        ax[2].set_ylabel('|B|\n(nT)', rotation=0, labelpad=20)
        ax[2].set_xlabel('Time (s)')
        
        for axes in ax:
            axes.legend()
            axes.set_xlim(0, time[-1])
    
    if False:
        # Check mag_history[0] (B0x) against eval_B0x(pos_history)
        # Cut off first value in mag_history because it records it twice
        # This might be the lag issue between the two elements.
        fig, ax = plt.subplots(2, sharex=True)
        ax[0].set_title('Recorded vs. Evaluated (using position history) B0x')
        ax[0].plot(time, mag_history[:, 0] * 1e9, label='B0x from v')
        ax[0].plot(time, mag_history_x     * 1e9, label='B0x from x')
        ax[0].legend()
        ax[0].set_ylabel('nT')
        
        diff = (mag_history[1:, 0] - mag_history_x[:-1]) * 1e9
        ax[1].set_title('Difference')
        ax[1].plot(time[:-1], diff)
        ax[1].set_ylabel('nT')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_xlim(0, time[-1])
   
    if False:
        # Check dot product of v_perp with mag_history. Should always be perpendicular
        # vx set to zero since we only care about v_perp
        v_dot_B = vel_history[:, 0] * mag_history[:, 0] * 0 \
                + vel_history[:, 1] * mag_history[:, 1] \
                + vel_history[:, 2] * mag_history[:, 2]
        
        fig, ax = plt.subplots(2, sharex=True)
        ax[0].set_title('$v_\perp \cdot B$ test :: Should always be perpendicular')
        ax[0].plot(time, v_dot_B)
        ax[0].set_ylabel('$v_\perp \cdot B$', rotation=0, labelpad=20)

        ax[1].plot(time, mu)
        ax[1].set_ylabel('$\mu$', rotation=0, labelpad=20)
        ax[1].set_xlabel('Time (s)')
        ax[1].set_xlim(0, time[-1])
    return
    

@nb.njit()
def analytic_B0_equation(R, X, A, BEQ, B0):
    for ii in nb.prange(X.shape[0]):
        for jj in nb.prange(R.shape[0]):
            B0[0, ii, jj] = BEQ * (1 + A * X[ii]**2)
            B0[1, ii, jj] = - A * R[jj] * BEQ * np.abs(X[ii])
    return


def test_B0_analytic():
    '''
    Analytically evaluate what magnetic field mapping this function is giving us.
    
    Combine B0y,z outputs to BOr. Compare expectation to theory? Also maybe specify rho
    instead? Find some way to get velocity out of that to input.
    
    # Test: The idea is that for a given v_perp, BO_r should come out the same. Is this true?
    #       If so, v_perp can be input -> turns our 2D parameter space into a 1D (and output as well)
    #       If not, there's a problem with either the derivation or the code... although both are
    #       super simple. I guess we'll find out...
    '''
    b1  = np.zeros(3)
    qmi = const.qm_ratios[0]
    
    if False:
        # Test: For a given (vr, x) is the resulting magnetic field output (B0x, B0r) constant in theta?
        # Yes, B0r is constant in theta, as expected
        x  = const.xmax
        vx = const.va
        vr = const.va
        
        Ntheta = 1000
        theta  = np.linspace(0, 2*np.pi, Ntheta)
        B0x    = np.zeros(Ntheta)
        B0r    = np.zeros(Ntheta)
        vya    = np.zeros(Ntheta)
        vza    = np.zeros(Ntheta)
        
        for ii in range(Ntheta):
            # Go around whole orbit
            vy  = vr * np.cos(theta[ii])
            vz  = vr * np.sin(theta[ii])
            vel = np.array([vx, vy, vz])
            vya[ii] = vy
            vza[ii] = vz
            # Check if 
            B0_particle = particles.eval_B0_particle(x, vel, qmi, b1)
            
            br = np.sqrt(B0_particle[1] ** 2 + B0_particle[2] ** 2)
            
            B0x[ii] = B0_particle[0]
            B0r[ii] = br
        
        fig, ax = plt.subplots()
        plt.plot(theta, B0x*1e9, label='B0_x', c='k')
        ax.plot(theta,  B0r*1e9, label='B0_r', c='r')
        ax.set_xlabel('Particle orbit phase (rad)')
        ax.set_ylabel('B0 component (nT)')
        ax.set_title( 'Analytic B0 around single particle orbit: Constant position, constant v_radial')
        ax.legend()
        
        ax2 = ax.twinx()
        ax2.plot(theta, vya, label='vy', c='orange')
        ax2.plot(theta, vza, label='vz', c='b')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.legend()
    
    if True:
        # Plot magnetic mirror from raw equation (independent of energy yet, r != rL yet)
        Nx  = 1000; Nr = 40000
        
        x_space   = np.linspace(const.xmin , const.xmax, Nx)    # Simulation domain
        r_space   = np.linspace(0, 400000, Nr)                  # Derived for a maximum speed of around 400km/s
        B0_output = np.zeros((2, Nx, Nr))                       # Each spatial point, radial distance
        
        print('Plotting magnetic map from equations...')
        analytic_B0_equation(r_space, x_space, const.a, const.B_eq, B0_output)
        
        B_magnitude = np.sqrt(B0_output[0] ** 2 + B0_output[1] ** 2)

        if True:
            # Not great: Flips the vertical axis without telling you (and plots the wrong y axis)
            plt.figure()
            plt.imshow(np.flip(B0_output[1].T*1e9, axis=0), cmap=mpl.cm.get_cmap('Blues_r'),
                       extent=[const.xmin, const.xmax, 0, 400000], aspect='auto')
            plt.title(r'Contour Plot of $B0_r$ with abs(x) in equation')
            plt.xlabel('x (m)')
            plt.ylabel('r (m)')
            plt.colorbar().set_label('Magnetic Field (nT)')
            
            plt.figure()
            plt.imshow(np.flip(B0_output[0].T*1e9, axis=0), cmap='Blues',
                       extent=[const.xmin, const.xmax, 0, 400000], aspect='auto')
            plt.title(r'Contour Plot of $B0_x$')
            plt.xlabel('x (m)')
            plt.ylabel('r (m)')
            plt.colorbar().set_label('Magnetic Field (nT)')   
            
            plt.figure()
            plt.imshow(np.flip(B_magnitude.T*1e9, axis=0), cmap='Blues',
                       extent=[const.xmin, const.xmax, 0, 400000], aspect='auto')
            plt.title(r'Contour Plot of $|B0|$')
            plt.xlabel('x (m)')
            plt.ylabel('r (m)')
            plt.colorbar().set_label('Magnetic Field (nT)')   
        else:
            # Backup plot that does it right, but takes ages
            plt.figure()
            plt.contourf(x_space*1e-3, r_space*1e-3, B0_output[1].T*1e9, levels=100, cmap='bwr')
            plt.title(r'Contour Plot of $B0_r$')
            plt.xlabel('x (km)')
            plt.ylabel('r (km)')
            plt.colorbar().set_label('Magnetic Field (nT)')
            
            plt.figure()
            plt.contourf(x_space*1e-3, r_space*1e-3, B0_output[0].T*1e9, levels=100, cmap='bwr')
            plt.title(r'Contour Plot of $B0_x$')
            plt.xlabel('x (km)')
            plt.ylabel('r (km)')
            plt.colorbar().set_label('Magnetic Field (nT)')       
    return


@nb.njit()
def interrogate_B0_function(theta=0):
    '''
    Arguments: eval_B0_particle(x, v, qmi, b1)
    
    For each point in space.
    For v_para = 0 (shouldn't matter, v[0] isn't used)
    For v_perp between +/- a few v_A
    
    Have v0y, v0z related by gyrophase, but that shouldn't matter
    For each v_perp, calculate rL based on B0x. Maybe can also calculate
    the error for B0r this way?
    
    rL will change in x due to varying B0x -> cyclotron frequency
    rL will change in v due to varying v_perp
    
    DEPRECATED. RETURNS TOO MUCH.
    '''
    qmi = const.q / const.mp
    b1  = np.zeros(3)

    Nx = 2500       # Number of points in space
    Nv = 2500       # Number of points in velocity
    VM = 10         # vA multiplier: Velocity space is between +/- va * VM

    x_axis      = np.linspace(  const.xmin,  const.xmax, Nx)    
    v_perp_axis = np.linspace(-VM*const.va, VM*const.va, Nv)
    vx          = 0.

    B0xr_surface = np.zeros((3, Nx, Nv))
    x_pos        = np.zeros((Nx, Nv))
    y_pos        = np.zeros((Nx, Nv))
    z_pos        = np.zeros((Nx, Nv))
    r_pos        = np.zeros((Nx, Nv))
    
    for ii in nb.prange(Nx):
        for jj in range(Nv):
            vy = v_perp_axis[jj] * np.cos(theta * np.pi / 180.)
            vz = v_perp_axis[jj] * np.sin(theta * np.pi / 180.)
            
            B0 = particles.eval_B0_particle(x_axis[ii], np.array([vx, vy, vz]), qmi, b1)
            
            B0xr_surface[0, ii, jj] = B0[0]
            B0xr_surface[1, ii, jj] = B0[1]
            B0xr_surface[2, ii, jj] = B0[2]
            
            r_pos[ii, jj] = const.mp * v_perp_axis[jj] / (const.q * B0[0])
            
            x_pos[ii, jj] = x_axis[ii]
            y_pos[ii, jj] = r_pos[ii, jj] * np.cos(theta * np.pi / 180.)
            z_pos[ii, jj] = r_pos[ii, jj] * np.sin(theta * np.pi / 180.)
            
    return x_axis, v_perp_axis, B0xr_surface, x_pos, y_pos, z_pos, r_pos


def save_B0_map_3D():
    '''
    Deprecated. Too much to work with.
    '''
    save_dir = 'F://runs//magnetic_bottle_savefiles//'
    
    r = 0
    for theta in np.arange(0, 360, 0.5):
        print('Analysing bottle for theta = {} degrees'.format(theta))
        x_axis, v_perp_axis, B0xr_surface, x_pos, y_pos, z_pos, r_pos = interrogate_B0_function(theta=theta)
    
        d_fullpath = save_dir + 'magbottle_%05d' % r
    
        np.savez(d_fullpath,
                 theta = np.array([theta]),
                 x_axis = x_axis,
                 v_perp_axis = v_perp_axis,
                 B0xr_surface = B0xr_surface,
                 x_pos = x_pos,
                 y_pos = y_pos,
                 z_pos = z_pos,
                 r_pos = r_pos)
        r += 1
    return


def plot_B0_function():
    x_axis, v_perp_axis, B0xr_surface, x_pos, y_pos, z_pos, r_pos = interrogate_B0_function()

    B0_r = np.sqrt(B0xr_surface[1] **2 + B0xr_surface[2] ** 2)

    if False: # Plot in x-v space
        plt.figure()
        plt.contourf(x_axis*1e-3, v_perp_axis*1e-3, B0xr_surface[0].T*1e9, levels=100)
        plt.title('B0x in x-v space')
        plt.colorbar().set_label('$B_{0x}$\n(nT)', rotation=0, labelpad=20)
        plt.xlabel('x (km)')
        plt.ylabel('$v_\perp$\n(km/s)', rotation=0)
        
        plt.figure()
        plt.contourf(x_axis*1e-3, v_perp_axis*1e-3, B0_r.T*1e9, levels=100)
        plt.title('B0r in x-v space')
        plt.colorbar().set_label('$B_{0r}$\n(nT)', rotation=0, labelpad=20)
        plt.xlabel('x (km)')
        plt.ylabel('$v_\perp$\n(km/s)', rotation=0)
        
    # For each value of (x, v), an rL is associated with it.
    if True:  
        plt.figure()
        
        # For each value of x, scatterplot all the values of rL
        # color is magnetic field
        for ii in range(x_axis.shape[0]):
            xi = np.ones(r_pos[ii].shape[0]) * x_axis[ii]
            
            if x_axis[ii] == 0:
                pdb.set_trace()
            
            plt.scatter(xi*1e-3, r_pos[ii]*1e-3, c=B0_r[ii]*1e9)
            
        plt.colorbar().set_label('$B_{0r}$\n(nT)', rotation=0, labelpad=20)
        plt.title('B0r in x-r space')
        plt.xlabel('x (km)')
        plt.ylabel('r (km)', rotation=0)
    return


@nb.njit()
def return_2D_slice_rtheta(x_val=0., Nv=500, Nt=360, VM=10.):
    '''
    For a specified x, creates a 2D slice using velocity/theta space
    
    Has to be regridded onto something more regular.
    '''
    qmi    = const.q / const.mp
    b1     = np.zeros(3)
    vx     = 0.

    v_perp_axis = np.linspace(-VM*const.va, VM*const.va, Nv)
    theta_axis  = np.linspace(0, 360., Nt)

    # In order to keep the coordinates flat
    B0xr_surface = np.zeros((3, Nv * Nt))
    y_pos        = np.zeros((Nv * Nt))
    z_pos        = np.zeros((Nv * Nt))
    
    for ii in nb.prange(Nv):
        for jj in range(Nt):
            xx = ii * Nt + jj       # Flattened array index
            
            vy = v_perp_axis[ii] * np.cos(theta_axis[jj] * np.pi / 180.)
            vz = v_perp_axis[ii] * np.sin(theta_axis[jj] * np.pi / 180.)
            B0 = particles.eval_B0_particle(x_val, np.array([vx, vy, vz]), qmi, b1)
            
            B0xr_surface[0, xx] = B0[0]
            B0xr_surface[1, xx] = B0[1]
            B0xr_surface[2, xx] = B0[2]

            r_pos     = const.mp * v_perp_axis[ii] / (const.q * B0[0])
            y_pos[xx] = r_pos * np.cos(theta_axis[jj] * np.pi / 180.)
            z_pos[xx] = r_pos * np.sin(theta_axis[jj] * np.pi / 180.)
            
    return y_pos, z_pos, B0xr_surface


def plot_single_2D_slice():
    '''
    For a single slice, interpolate onto 165km x 165km grid 
    
    Could do this for N positions along the x-axis, and then slice them however I want.
    '''
    from scipy.interpolate import griddata
    
    # Get irregularly spaced (y,z) points with 3D field at each point
    x_val = const.xmax
    y_pos, z_pos, B0xr_slice = return_2D_slice_rtheta(x_val=x_val)

    length = 165        # Box diameter for interpolation
    Nl     = 100        # Number of gridpoints per side
    
    #plt.scatter(y_pos, z_pos, B0xr_surface[1])
    
    # Create grid
    y_axis = np.linspace(-length, length, Nl)
    z_axis = np.linspace(-length, length, Nl)
    yi, zi = np.meshgrid(y_axis, z_axis)
    
    print('Interpolating B0x')
    interpolated_B0x = griddata(np.array([y_pos, z_pos]).T, B0xr_slice[0]*1e9, (yi, zi), method='cubic')
    
    print('Interpolating B0y')
    interpolated_B0y = griddata(np.array([y_pos, z_pos]).T, B0xr_slice[1]*1e9, (yi, zi), method='cubic')
    
    print('Interpolating B0z')
    interpolated_B0z = griddata(np.array([y_pos, z_pos]).T, B0xr_slice[2]*1e9, (yi, zi), method='cubic')
    
    if True:
        plt.figure()
        plt.quiver(y_axis, z_axis, interpolated_B0y, interpolated_B0z)
    
    if False:
        plt.figure()
        plt.imshow(interpolated_B0x.T*1e9, extent=(-length, length, -length, length), origin='lower')
        plt.title('B0x Magnetic Bottle y-z plane slice at x = {}'.format(x_val))
        plt.xlabel('y (km)')
        plt.ylabel('z (km)')
        plt.colorbar().set_label('B0x\n(nT)', rotation=0, labelpad=20)
        
        plt.figure()
        plt.imshow(interpolated_B0y.T*1e9, extent=(-length, length, -length, length), origin='lower')
        plt.title('B0y Magnetic Bottle y-z plane slice at x = {}'.format(x_val))
        plt.xlabel('y (km)')
        plt.ylabel('z (km)')
        plt.colorbar().set_label('B0y\n(nT)', rotation=0, labelpad=20)
        
        plt.figure()
        plt.imshow(interpolated_B0z.T*1e9, extent=(-length, length, -length, length), origin='lower')
        plt.title('B0z Magnetic Bottle y-z plane slice at x = {}'.format(x_val))
        plt.xlabel('y (km)')
        plt.ylabel('z (km)')
        plt.colorbar().set_label('B0z\n(nT)', rotation=0, labelpad=20)
    return


def calculate_all_2D_slices():
    '''
    For a single slice, interpolate onto 165km x 165km grid 
    
    Could do this for N positions along the x-axis, and then slice them however I want.
    '''
    from scipy.interpolate import griddata
    save_dir = 'F://runs//magnetic_bottle_savefiles//'
    
    # Get extrema
    y_pos, z_pos, B0xr_slice = return_2D_slice_rtheta(x_val=const.xmax)
    
    length = roundup(y_pos.max()*1e-3, nearest=10.)     # Box diameter for interpolation (in km)
    Nl     = 100                                        # Number of gridpoints per side
    Nx     = 100                                        # Number of gridpoints along simulation domain

    x_axis = np.linspace(const.xmin, const.xmax, Nx)
    
    # Create grid for each 2D slice (in m)
    y_axis = np.linspace(-length*1e3, length*1e3, Nl)
    z_axis = np.linspace(-length*1e3, length*1e3, Nl)
    yi, zi = np.meshgrid(y_axis, z_axis)
        
    field_shape = np.zeros((Nx, Nl, Nl, 3))
    
    ii = 0
    for x_val in x_axis:
        d_slicepath = save_dir + 'bottleslice_%05d' % ii
        
        print('Slice x = {:5.2f}km'.format(x_val * 1e-3))
        y_pos, z_pos, B0xr_slice = return_2D_slice_rtheta(x_val=x_val)

        print('Interpolating...')
        field_shape[ii, :, :, 0] = griddata(np.array([y_pos, z_pos]).T, B0xr_slice[0]*1e9, np.array([y_axis, z_axis]).T, method='cubic')
        field_shape[ii, :, :, 1] = griddata(np.array([y_pos, z_pos]).T, B0xr_slice[1]*1e9, np.array([y_axis, z_axis]).T, method='cubic')
        field_shape[ii, :, :, 2] = griddata(np.array([y_pos, z_pos]).T, B0xr_slice[2]*1e9, np.array([y_axis, z_axis]).T, method='cubic')
        pdb.set_trace()
        print('Saving...')
        np.savez(d_slicepath,
                 x_val = np.array([x_val]),
                 B0xi  = field_shape[ii, :, :, 0],
                 B0yi  = field_shape[ii, :, :, 1],
                 B0zi  = field_shape[ii, :, :, 2])

        ii += 1
        
    # Save result
    print('Saving total result')
    d_slicepath_all = save_dir + '_bottleslice_all'
    
    np.savez(d_slicepath_all,
             x_axis       = x_axis,
             y_axis       = y_axis,
             z_axis       = z_axis,
             field_shape  = field_shape)
    return


@nb.njit()
def smart_interrogate_B0(Nx=100, Nl=50, v_max_va=10):
    '''
    Uses positions (y,z) on a grid to calculate required v_perp
    (from rL) and theta from arctan2. This aligns theta=0 with the +y axis.
    This changes the signs for velocity, as in a RHCS with this geometry, a
    particle would gyrate CCW as viewed from behind (going down +x)
    '''
    # Get extrema, where rL is at maximum (boundary)
    y_pos, z_pos, B0xr_slice = return_2D_slice_rtheta(x_val=const.xmax, VM=v_max_va)
    
    length = roundup(y_pos.max()*1e-3, nearest=10.)     # Box diameter for interpolation (in km)

    x_axis = np.linspace(const.xmin, const.xmax, Nx)
    y_axis = np.linspace(-length*1e3, length*1e3, Nl)
    z_axis = np.linspace(-length*1e3, length*1e3, Nl)
    
    qmi    = const.q / const.mp
    b1     = np.zeros(3)
    vx     = 0.

    B0_grid = np.zeros((Nx, Nl, Nl, 3))
    
    for ii in nb.prange(Nx):
        
        B0x = fields.eval_B0x(x_axis[ii])  
        
        for jj in range(Nl):
            for kk in range(Nl):
                rL      = np.sqrt(y_axis[jj] ** 2 + z_axis[kk] ** 2)
                v_perp  = rL * qmi * B0x
                theta   = np.arctan2(z_axis[kk], y_axis[jj])
                
                vy =    v_perp * np.sin(theta)
                vz =  - v_perp * np.cos(theta)
                
                B0 = particles.eval_B0_particle(x_axis[ii], np.array([vx, vy, vz]), qmi, b1)
                
                B0_grid[ii, jj, kk, 0] = B0[0]
                B0_grid[ii, jj, kk, 1] = B0[1]
                B0_grid[ii, jj, kk, 2] = B0[2]
    
    return x_axis, y_axis, z_axis, B0_grid


def smart_plot_2D_planes():
    '''
    Do slices in each of 3 planes
    
    x,y,z axes are 1D
    B0_grid is B0 at each (x,y,z) with 3 components (shape [3, x, y, z])
    
    CHECK AXES ORDER :: MIGHT BE LABELLED WRONG/BACKWARDS (NOTICABLE FOR Y,Z .vs X)
    '''
    savedir = 'F://runs//bottle_plane_plots//'
    v_max   = 15
    x_axis, y_axis, z_axis, B0_grid = smart_interrogate_B0(v_max_va=v_max)

    B0r_max = 8

    # yz slice at some x
    for ii in range(x_axis.shape[0]):
        x_mid = ii
        B0r   = np.sqrt(B0_grid[x_mid, :, :, 1] ** 2 + B0_grid[x_mid, :, :, 2] ** 2)
        
        fig, ax = plt.subplots(figsize=(16,10))
        im1 = ax.quiver(y_axis*1e-3, z_axis*1e-3, B0_grid[x_mid, :, :, 1].T, B0_grid[x_mid, :, :, 2].T, B0r*1e9, clim=(0, B0r_max))
        
        ax.set_xlabel('y (km)')
        ax.set_ylabel('z (km)')
        ax.set_title('YZ slice at X = {:5.2f} km :: B0r vectors :: Vmax = {}vA'.format(x_axis[x_mid]*1e-3, v_max))
        fig.colorbar(im1).set_label('B0r (nT)')
        
        filename = 'yz_plane_%05d.png' % ii
        savepath = savedir + 'yz_plane//' + filename
        plt.savefig(savepath)
        print('yz plot {} saved'.format(ii))
        plt.close('all')
    
    # xy slice at some z
    for jj in range(z_axis.shape[0]):
        z_mid = jj
        B0xy  = np.sqrt(B0_grid[:, :, z_mid, 0] ** 2 + B0_grid[:, :, z_mid, 1] ** 2)
          
        fig, ax = plt.subplots(figsize=(16,10))
        im2 = ax.quiver(x_axis*1e-3, y_axis*1e-3, B0_grid[:, :, z_mid, 0].T, B0_grid[:, :, z_mid, 1].T, B0xy*1e9, clim=(200, 425))
        
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_title('XY slice at Z = {:5.2f} km :: B0xy vectors :: Vmax = {}vA'.format(z_axis[z_mid]*1e-3, v_max))
        fig.colorbar(im2).set_label('B0xy (nT)')   
        
        filename = 'xy_plane_%05d.png' % jj
        savepath = savedir + 'xy_plane//' + filename
        plt.savefig(savepath)
        print('xy plot {} saved'.format(jj))
        plt.close('all')
    
    # xz slice at some y
    for kk in range(y_axis.shape[0]):
        y_mid = kk
        B0xz  = np.sqrt(B0_grid[:, y_mid, :, 0] ** 2 + B0_grid[:, y_mid, :, 2] ** 2)
          
        fig, ax = plt.subplots(figsize=(16,10))
        im3 = ax.quiver(x_axis*1e-3, z_axis*1e-3, B0_grid[:, y_mid, :, 0].T, B0_grid[:, y_mid, :, 2].T, B0xz*1e9, clim=(200, 425))
        
        ax.set_xlabel('x (km)')
        ax.set_ylabel('z (km)')
        ax.set_title('XZ slice at Y = {:5.2f} km :: B0xz vectors :: Vmax = {}vA'.format(y_axis[y_mid]*1e-3, v_max))
        fig.colorbar(im3).set_label('B0xz (nT)')  
        
        filename = 'xz_plane_%05d.png' % kk
        savepath = savedir + 'xz_plane//' + filename
        plt.savefig(savepath)
        print('xz plot {} saved'.format(kk))
        plt.close('all')
    
    return


def smart_plot_3D():
    '''
    Doesn't work, and would probably be mega slow anyway
    '''
    from mpl_toolkits.mplot3d import Axes3D
    
    x_axis, y_axis, z_axis, B0_grid = smart_interrogate_B0()
    
    X,Y,Z = np.meshgrid(x_axis, y_axis, z_axis)
    U = B0_grid[:, :, :, 0]
    V = B0_grid[:, :, :, 1]
    W = B0_grid[:, :, :, 2]

    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    
    ax.quiver(X, Y, Z, U, V, W)
    return


def check_directions():
    v_perp = 1.
    angles = np.linspace(0, 2 * np.pi, 5)
    
    for theta in angles:
        vy =   v_perp * np.sin(theta)
        vz = - v_perp * np.cos(theta)
        
        print('theta {:5.1f} : vy = {:4.1f}, vz = {:4.1f}'.format(theta * 180 / np.pi, vy, vz))
    return


def test_particle_open_boundary_fluxes():
    '''
    Full particle pusher
    Random initializes one species of Np particles
    Just for testing boundary fluxes
    '''   
    print('Testing boundary fluxes')
    
    # Initialize arrays
    Np     = 10
    N_init = 10*Np
    
    idx      = np.ones(N_init, dtype=int) * -1
    W_elec   = np.zeros((3, N_init))
    W_mag    = np.zeros((3, N_init))
    Ep       = np.zeros((3, N_init))
    Bp       = np.zeros((3, N_init))
    v_prime  = np.zeros((3, N_init))
    S        = np.zeros((3, N_init))
    T        = np.zeros((3, N_init))
    Ie       = np.zeros(N_init, dtype=int)
    Ib       = np.zeros(N_init, dtype=int)
    qmi      = np.zeros(N_init)
    
    B_test   = np.zeros((const.NC + 1, 3), dtype=np.float64) 
    E_test   = np.zeros((const.NC, 3),     dtype=np.float64) 
    
    vel      = np.zeros((3, N_init), dtype=np.float64)
    pos      = np.zeros((3, N_init), dtype=np.float64)
        
# =============================================================================
#     vel[0, :Np] = np.random.normal(0.0, const.vth_par[ jj], Np)
#     vel[1, :Np] = np.random.normal(0.0, const.vth_perp[jj], Np)
#     vel[2, :Np] = np.random.normal(0.0, const.vth_perp[jj], Np)
#     
#     pos[0, :Np] = np.random.uniform(const.xmin, const.xmax, Np)
# =============================================================================

    pos[0, 0] = const.xmin + const.dx / 2.0
    vel[0, 0] = const.va * 10.0
    
    pos[0, 1] = const.xmin + const.dx * 2.0
    vel[0, 1] = const.va * -10.0
    
    pos[0, 2] = const.xmax - const.dx * 2.0
    vel[0, 2] = const.va * 10.0
    
    pos[0, 3] = const.xmax - const.dx / 2.0
    vel[0, 3] = const.va * -10.0
    
    idx[0] = 1
    idx[1] = 1
    idx[2] = 1
    idx[3] = 1

    # Initial quantities
    ion_ts   = const.orbit_res / const.gyfreq
    vel_ts   = 0.5 * const.dx / np.max(np.abs(vel[0, :]))
    DT       = min(ion_ts, vel_ts)
    
    # Target: 25000 cyclotron periods (~1hrs)
    max_t    = const.max_rev / const.gyfreq
    max_inc  = int(max_t / DT) + 1

    # Retard velocity for stability
    particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B_test, E_test, v_prime, S, T, qmi, -0.5*DT)
    
    # Record initial values
    tt = 0; t_total = 0; max_inc=400
    while tt < max_inc - 1:

        plot_scatter_and_save(pos, vel, idx, tt)

        tt      += 1
        t_total += DT
        
# =============================================================================
#         db_pos = pos.T.copy()/const.dx
#         db_vel = vel.T.copy()/const.va
#         db_idx = idx.copy()
# =============================================================================
            
        save_boundary_diagnostics(pos[0]/const.dx, vel[0]/const.va, idx, tt)
# =============================================================================
#         if tt > 87:
#             print(db_idx[6: 8], db_pos[6:8, 0], db_vel[6:8, 0])
#             pdb.set_trace()
# =============================================================================
            
        particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B_test, E_test, v_prime, S, T, qmi, DT)
        particles.position_update(pos, vel, idx, Ep, DT, Ie, W_elec)
        
        
        
# =============================================================================
#         if tt > 50:
#             pdb.set_trace()
# =============================================================================
# =============================================================================
#             db_pos = pos.T.copy()/const.dx
#             db_vel = vel.T.copy()/const.va
#             db_idx = idx.copy()
#             pdb.set_trace()
# =============================================================================
    return

def save_boundary_diagnostics(pos, vel, idx, ii):
    print('Saving to file')
    save_dir = const.drive + '//' + const.save_path + '//DIAGNOSTICS_TEXT//'
    
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
        
    with open(save_dir + 'pos.txt', 'a') as pos_file:  
        pos_file.write('{:<5}'.format(ii))
        for val in pos:
            pos_file.write('{:6.2f}'.format(val))
        pos_file.write('\n')
        
    with open(save_dir + 'vel.txt', 'a') as vel_file:  
        vel_file.write('{:<5}'.format(ii))
        for val in vel:
            vel_file.write('{:6.2f}'.format(val))
        vel_file.write('\n')
        
    with open(save_dir + 'idx.txt', 'a') as idx_file:  
        idx_file.write('{:<5}'.format(ii))
        for val in idx:
            idx_file.write('{:3}'.format(val))
        idx_file.write('\n')
    return

def plot_scatter_and_save(pos, vel, idx, ii):   
    print('Saving phase space plot {}'.format(ii))
    save_dir = const.drive + '//' + const.save_path + '//DIAGNOSTICS//'
    
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
        
    n_active   = (idx >= 0).sum()
    n_disabled = (idx < 0).sum()
        
    plt.ioff()
    fig, ax = plt.subplots(figsize=(15, 10))
    
    plot_pos = np.ma.masked_where(idx < 0, pos[0])
    plot_vel = np.ma.masked_where(idx < 0, vel[0])

    ax.scatter(plot_pos/const.dx, plot_vel/const.va, c='k')
    ax.set_xlim(const.xmin/const.dx, const.xmax/const.dx)
    ax.set_ylim(-20, 20)
    ax.set_title('Open Boundary Test :: Active {} :: Disabled {}'.format(n_active, n_disabled))
    
    for node in const.B_nodes:
        ax.axvline(node / const.dx, c='k', alpha=0.5, ls=':')
    
    fig.savefig(save_dir + 'pos_vel_{:05}'.format(ii))
    plt.close('all')
    return


def plot_distributions_and_save(pos, vel, idx, ii, jj):   
    print('Saving phase space plot {}'.format(ii))
    save_dir = const.drive + '//' + const.save_path + '//DIAGNOSTICS//'

    lt = ['x', 'y', 'z']
              
    cfac = 10 if const.temp_type[jj] == 1 else 5
    vlim = 10 if const.temp_type[jj] == 1 else 5
    
    # Manually specify bin edges for histogram
    vbins = np.linspace(-vlim, vlim, 101, endpoint=True)
    xbins = np.linspace(const.xmin/const.dx, const.xmax/const.dx, const.NX + 1, endpoint=True)

    plot_pos = np.ma.masked_where(idx < 0, pos[0])
    plot_vel = np.ma.masked_where(idx < 0, vel[0])

    # Do the plotting
    plt.ioff()
    
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title('v{} vs. x :: {} :: Time {:.2f}s'.format(lt[comp], cf.species_lbl[jj], ptime))
        
    counts, xedges, yedges, im1 = ax.hist2d(pos[0, st:en]/cf.dx, vel[0, st:en]/cf.va, 
                                            bins=[xbins, vbins],
                                            vmin=0, vmax=cf.nsp_ppc[jj] / cfac)

    cb = fig.colorbar(im1, ax=ax)
    cb.set_label('Counts')
    
    ax.set_xlim(cf.xmin/cf.dx, cf.xmax/cf.dx)
    ax.set_ylim(-vlim, vlim)
    
    ax.set_xlabel('Position (cell)')
    ax.set_ylabel('Velocity ($v_A$)')

    save_dir = cf.anal_dir + '//Particle Distribution Histograms//For Every Time//Species {}//v{}//'.format(jj, lt[comp])
    filename = 'v{}_vs_x_species_{}_{:05}'.format(lt[comp], jj, ii)
    
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    
    plt.savefig(save_dir + filename, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close('all')

    return


def test_cubic_spine_and_B0():
    '''
    Works fine. No values set in Bx because B0x is always expected to be zero
    b1 too.
    '''
    Ex = const.E_nodes
    Bx = const.B_nodes
    L  = 2*const.xmax
    
    signal = np.zeros((Bx.shape[0], 3))
    interp = np.zeros((Bx.shape[0], 3))
    
    signal[:, 1] = 1.0 * np.sin(2 * np.pi * 1.0 * Bx / L)
    signal[:, 2] = 1.0 * np.sin(2 * np.pi * 2.0 * Bx / L)
    aux.interpolate_edges_to_center(signal, interp)
    
    plt.plot(Bx, signal[:, 0], marker='o', c='b')
    plt.plot(Ex, interp[:-1, 0], marker='x', c='r')
    return


def test_source_term_filling():
    return


if __name__ == '__main__':
    #check_position_distribution()
    #animate_moving_weight()
    #test_particle_orbit()
    #test_curl_B()
    #test_curl_E()
    #test_grad_P()
    #test_grad_P_with_init_loading()
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
    #test_mirror_motion()
    #test_B0_analytic()
    #plot_B0_function()
    #save_B0_map_3D()
    #return_2D_slice_rtheta()
    #interpolate_2D_slice()
    #calculate_all_2D_slices()
    #smart_plot_2D_planes()
    #smart_plot_3D()
    #check_directions()
    #plot_dipole_field_line(length=True)
    #test_particle_open_boundary_fluxes()
    test_cubic_spine_and_B0()
    
    
    
# =============================================================================
#     init_pos, init_vel, time, pos_history, vel_history, mag_history,\
#         DT, max_t, POS_gphase, VEL_gphase = do_particle_run(max_rev=1, v_mag=10.0, pitch=41.0, dt_mult=1.0)
#         
#     if False:
#         fig, ax = plt.subplots()
#         ax.set_title('Velocity history')
#         ax.scatter(vel_history[:, 0, 1], vel_history[:, 0, 2], c='r', marker='o')
#         ax.axis('equal')
#         
#         fig, ax = plt.subplots()
#         ax.set_title('Position history')
#         ax.scatter(pos_history[:, 0, 1], pos_history[:, 0, 2], c='b', marker='o')
#         ax.axis('equal')
#         
#         fig, ax = plt.subplots(2, sharex=True)
#         
#         ax[0].plot(time[:-1], POS_gphase, marker='o')
#         ax[1].plot(time[:-1], VEL_gphase, marker='o')
#         
#     if False:
#         larmor = np.sqrt(pos_history[:, 0, 1] ** 2 + pos_history[:, 0, 2] ** 2) * 1e-3
#         
#         fig, axes = plt.subplots(2, sharex=True)
#         
#         for jj, comp in zip(range(3), ['x', 'y', 'z']):
#             axes[0].plot(time, pos_history[:, 0, jj]* 1e-3, label= '{}'.format(comp))
#             axes[1].plot(time, vel_history[:, 0, jj]* 1e-3, label='v{}'.format(comp))
#         
#         axes[0].plot(time, larmor, c='k', label='r_L')
#         
#         axes[0].set_title('Test :: Single Particle :: NX = {}'.format(const.NX))
#         axes[0].set_ylabel('x (km)')
#         axes[1].set_ylabel('v (km/s)')
#         axes[1].set_xlabel('t (s)')    
#         axes[1].set_xlim(0, None)
#         axes[0].legend()
#         axes[1].legend()
#         
#     if False:
#         fig, axes = plt.subplots(2, sharex=True)
#         
#         for ii, lbl in zip(range(2), ['fast', 'slow']):
#             axes[0].plot(time, pos_history[:, ii, 0]* 1e-3, label='{}'.format(lbl), marker='o')
#             axes[1].plot(time, vel_history[:, ii, 0]* 1e-3, label='{}'.format(lbl), marker='o')
#         
#         axes[0].set_title('Test :: Two particles, one with half vz, DT = {:05f}s'.format(DT))
#         axes[0].set_ylabel('x (m)')
#         axes[1].set_ylabel('v (km)')
#         axes[1].set_xlabel('t (s)')    
#         axes[1].set_xlim(0, None)
#         axes[0].legend()
#         axes[1].legend()
#             
#     elif False:
#         # 3D position plot
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         
#         pos_history *= 1e-3
#         ax.plot(pos_history[:, 0, 0], pos_history[:, 0, 1], pos_history[:, 0, 2], marker='o')
#             
#         ax.set_title('Particle Position in 3D')
#         ax.set_xlabel('x (km)')
#         ax.set_ylabel('y (km)')
#         ax.set_zlabel('z (km)')
#             
#         fig = plt.figure()
#         ax  = fig.add_subplot(111, projection='3d')
#         
#         vel_history *= 1e-3
#         ax.plot(pos_history[:, 0, 0], vel_history[:, 0, 1], pos_history[:, 0, 2], marker='o')
#         
#         ax.set_title('Particle Velocity in 2D (along x)')
#         ax.set_xlabel('x (km)')
#         ax.set_ylabel('vy (km/s)')
#         ax.set_zlabel('vz (km/s)')
# =============================================================================
    