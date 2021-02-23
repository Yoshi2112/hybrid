# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:42:13 2017

@author: iarey
"""
import math, os, pdb
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib import animation
from timeit import default_timer as timer

import main_1D




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


def collect_macroparticle_moments(pos, vel, idx): #
    import sources_1D   as sources
    import particles_1D as particles
    
    I = np.zeros(pos.shape[1], dtype=int)
    W = np.zeros(pos.shape   , dtype=float)
    
    ni = np.zeros((main_1D.NC, main_1D.Nj),    dtype=nb.float64)
    nu = np.zeros((main_1D.NC, main_1D.Nj, 3), dtype=nb.float64)
    
    # Zeroth and First moment calculations
    main_1D.assign_weighting_TSC(pos, I, W, E_nodes=True)
    main_1D.deposit_moments_to_grid(vel, I, W, idx, ni, nu)
    
    fig, [ax1, ax2] = plt.subplots(2, figsize=(15, 10))
    
    x_arr = np.arange(main_1D.NC)
    
    ax1.plot(x_arr, ni)
    ax2.plot(x_arr, nu)
    return


def check_cell_velocity_distribution(pos, vel, node_number, j=0): #
    '''Checks the velocity distribution of a particle species within a specified cell
    '''
    # Account for damping nodes. Node_number should be "real" node count.
    node_number += main_1D.ND
    x_node = main_1D.E_nodes[node_number]
    f      = np.zeros((1, 3))
    
    count = 0
    for ii in np.arange(main_1D.idx_start[j], main_1D.idx_end[j]):
        if (abs(pos[0, ii] - x_node) <= 0.5*main_1D.dx):
            f = np.append(f, [vel[0:3, ii]], axis=0)
            count += 1

    print('{} particles counted for diagnostic'.format(count))
    fig = plt.figure(figsize=(12,10))
    fig.suptitle('Particle velocity distribution of species {} in cell {}'.format(j, node_number))
    fig.patch.set_facecolor('w')
    num_bins = main_1D.nsp_ppc[j] // 20

    ax_x = plt.subplot2grid((2, 3), (0,0), colspan=2, rowspan=2)
    ax_y = plt.subplot2grid((2, 3), (0,2))
    ax_z = plt.subplot2grid((2, 3), (1,2))
    
    for ax in [ax_x, ax_y, ax_z]:
        ax.axvline(0, c='k', ls=':', alpha=0.25)

    xs, BinEdgesx = np.histogram((f[:, 0] - main_1D.drift_v[j]) / main_1D.va, bins=num_bins)
    bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
    ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
    ax_x.set_xlabel(r'$v_x / v_A$')
    #ax_x.set_xlim(-2, 2)

    ys, BinEdgesy = np.histogram(f[:, 1] / main_1D.va, bins=num_bins)
    by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
    ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
    ax_y.set_xlabel(r'$v_y / v_A$')
    #ax_y.set_xlim(-2, 2)

    zs, BinEdgesz = np.histogram(f[:, 2] / main_1D.va, bins=num_bins)
    bz = 0.5 * (BinEdgesz[1:] + BinEdgesz[:-1])
    ax_z.plot(bz, zs, '-', c='c', drawstyle='steps')
    ax_z.set_xlabel(r'$v_z / v_A$')
    #ax_z.set_xlim(-2, 2)

    plt.show()
    return


def check_cell_velocity_distribution_2D(pos, vel, node_number, jj=0, show_cone=False, save=False, qq=None): #
    '''
    Checks the velocity distribution of a particle species within a specified cell
    
    Starts at "cell 0" - i.e. only counts real cells. 
    Damping cell offset is handled by function
    '''
    if node_number is None:
        node_number = np.arange(main_1D.NX)
        
    # Account for damping nodes. Node_number should be "real" node count.
    for node in node_number:
        print('Collecting particle info in cell {}'.format(node))
        node += main_1D.ND
        x_node = main_1D.E_nodes[node]
        f      = np.zeros((1, 3))
        
        count = 0
        for ii in np.arange(main_1D.idx_start[jj], main_1D.idx_end[jj]):
            if (abs(pos[0, ii] - x_node) <= 0.5*main_1D.dx):
                f = np.append(f, [vel[0:3, ii]], axis=0)
                count += 1
    
        plt.ioff()
        #print('{} particles counted for diagnostic'.format(count))
        fig, ax1 = plt.subplots(figsize=(15,10))
        fig.suptitle('Particle velocity distribution of species {} in cell {}'.format(jj, node))
        fig.patch.set_facecolor('w')
    
        V_PERP = np.sign(f[:, 2]) * np.sqrt(f[:, 1] ** 2 + f[:, 2] ** 2) / main_1D.va
        V_PARA = f[:, 0] / main_1D.va
    
        ax1.scatter(V_PERP, V_PARA, s=1, c=main_1D.temp_color[jj])
    
        ax1.set_ylabel('$v_\parallel (/v_A)$')
        ax1.set_xlabel('$v_\perp (/v_A)$')
        ax1.axis('equal')
        
        # Plot loss cone lines 
        B_at_nodes     = main_1D.eval_B0x(np.array([x_node + main_1D.dx, x_node - main_1D.dx]))
        end_loss_cones = np.arcsin(np.sqrt(B_at_nodes/main_1D.B_A)).max()                    # Calculate max loss cone
        
        yst, yend = ax1.get_ylim()                  # Get V_PARA lims
        yarr      = np.linspace(yst, yend, 100)     # V_PARA values on axis
        xarr      = yarr * np.tan(end_loss_cones)   # Calculate V_PERP values
        ax1.plot(xarr,  yarr, c='k', label='Loss Cone: {:.1f} deg'.format(end_loss_cones * 180. / np.pi))
        ax1.plot(xarr, -yarr, c='k')
        ax1.legend()
        ax1.set_xlim(yst, yend)
        ax1.set_ylim(yst, yend)
        
        if save == True:
            save_path = main_1D.drive + '//' + main_1D.save_path + '//' + 'run_{}//LCD_plot_sp_{}//'.format(main_1D.run, jj)
            
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
    for j in range(main_1D.Nj):
        fig = plt.figure(figsize=(12,10))
        fig.suptitle('Configuration space distribution of {}'.format(main_1D.species_lbl[j]))
        fig.patch.set_facecolor('w')
        
        if num_bins is None:
            num_bins = main_1D.NX
    
        ax_x = plt.subplot()
    
        xs, BinEdgesx = np.histogram(pos[0, main_1D.idx_start[j]: main_1D.idx_end[j]], bins=num_bins)
        bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
        ax_x.plot(bx, xs, '-', c=main_1D.temp_color[main_1D.temp_type[j]], drawstyle='steps')
        ax_x.set_xlabel(r'$x_p (m)$')
        ax_x.set_xlim(main_1D.xmin, main_1D.xmax)

    plt.show()
    return

def check_velocity_distribution(vel):
    '''Checks the velocity distribution of an entire species across the simulation domain
    '''
    for j in range(main_1D.Nj):
        fig = plt.figure(figsize=(12,10))
        fig.suptitle('Velocity distribution of species {} in simulation domain'.format(j))
        fig.patch.set_facecolor('w')
        num_bins = main_1D.nsp_ppc[j] // 5
    
        ax_x = plt.subplot2grid((2, 3), (0,0), colspan=2, rowspan=2)
        ax_y = plt.subplot2grid((2, 3), (0,2))
        ax_z = plt.subplot2grid((2, 3), (1,2))
    
        xs, BinEdgesx = np.histogram(vel[0, main_1D.idx_start[j]: main_1D.idx_end[j]] / main_1D.va, bins=num_bins)
        bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
        ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
        ax_x.set_xlabel(r'$v_x / v_A$')
    
        ys, BinEdgesy = np.histogram(vel[1, main_1D.idx_start[j]: main_1D.idx_end[j]] / main_1D.va, bins=num_bins)
        by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
        ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
        ax_y.set_xlabel(r'$v_y / v_A$')
    
        zs, BinEdgesz = np.histogram(vel[2, main_1D.idx_start[j]: main_1D.idx_end[j]] / main_1D.va, bins=num_bins)
        bz = 0.5 * (BinEdgesz[1:] + BinEdgesz[:-1])
        ax_z.plot(bz, zs, '-', c='c', drawstyle='steps')
        ax_z.set_xlabel(r'$v_z / v_A$')

    plt.show()
    return


def check_velocity_components_vs_space(pos, vel, jj=1):
    '''
    For each point in time
     - Collect particle information for particles near cell, plus time component
     - Store in array
     - Plot using either hexbin or hist2D
    '''
    print('Calculating velocity distributions vs. space for species {},...'.format(jj))
    comp = ['x', 'y', 'z']
    cfac = 10 if main_1D.temp_type[jj] == 1 else 5
    vlim = 15 if main_1D.temp_type[jj] == 1 else 5
    
    V_PERP = np.sign(vel[2]) * np.sqrt(vel[1] ** 2 + vel[2] ** 2) / main_1D.va
    V_PARA = vel[0] / main_1D.va
    
    # Manually specify bin edges for histogram
    vbins = np.linspace(-vlim, vlim, 101, endpoint=True)
    xbins = np.linspace(main_1D.xmin/main_1D.dx, main_1D.xmax/main_1D.dx, main_1D.NX + 1, endpoint=True)
        
    # Do the plotting
    plt.ioff()
    
    fig, axes = plt.subplots(3, figsize=(15, 10), sharex=True)
    axes[0].set_title('f(v) vs. x :: {}'.format(main_1D.species_lbl[jj]))
    
    st = main_1D.idx_start[jj]
    en = main_1D.idx_end[jj]
    
    for ii in range(3):
        counts, xedges, yedges, im1 = axes[ii].hist2d(pos[0, st:en]/main_1D.dx, vel[ii, st:en]/main_1D.va, 
                                                bins=[xbins, vbins],
                                                vmin=0, vmax=main_1D.nsp_ppc[jj] / cfac)
    
        cb = fig.colorbar(im1, ax=axes[ii])
        cb.set_label('Counts')
        
        axes[ii].set_xlim(main_1D.xmin/main_1D.dx, main_1D.xmax/main_1D.dx)
        axes[ii].set_ylim(-vlim, vlim)
        
        axes[ii].set_ylabel('v{}\n($v_A$)'.format(comp[ii]), rotation=0)
        
    axes[2].set_xlabel('Position (cell)')
    fig.subplots_adjust(hspace=0)
    plt.show()
    
    print('\n')
    return


def plot_temperature_extremes():
    '''
    Calculate what the max/min temperatures for the distribution are at eq/boundary
    For each species, plot these two distributions
    
    Two positions: eq/xmax, Two components, par/per, multiple species Nj
    '''
    vlim = 10
    jj   = 1
    
    from scipy.stats import norm
    
    T_eq_par = main_1D.beta_par[jj] * main_1D.B_eq ** 2 / (2 * main_1D.mu0 * main_1D.ne * main_1D.kB)
    T_eq_per = main_1D.beta_per[jj] * main_1D.B_eq ** 2 / (2 * main_1D.mu0 * main_1D.ne * main_1D.kB)
    
    T_xmax_par = main_1D.beta_par[jj] * main_1D.B_xmax ** 2 / (2 * main_1D.mu0 * main_1D.ne * main_1D.kB)
    T_xmax_per = main_1D.beta_per[jj] * main_1D.B_xmax ** 2 / (2 * main_1D.mu0 * main_1D.ne * main_1D.kB)
    
    sf_eq_par = np.sqrt(main_1D.kB *  T_eq_par /  main_1D.mass[jj])
    sf_eq_per = np.sqrt(main_1D.kB *  T_eq_per /  main_1D.mass[jj])
    
    sf_xmax_par = np.sqrt(main_1D.kB *  T_xmax_par /  main_1D.mass[jj])
    sf_xmax_per = np.sqrt(main_1D.kB *  T_xmax_per /  main_1D.mass[jj])
    
    x = np.linspace(-vlim*main_1D.va, vlim*main_1D.va, 100)
    
    prob_eq_par   = norm.pdf(x, 0.0, sf_eq_par)
    prob_eq_per   = norm.pdf(x, 0.0, sf_eq_per)
    
    prob_xmax_par = norm.pdf(x, 0.0, sf_xmax_par)
    prob_xmax_per = norm.pdf(x, 0.0, sf_xmax_per)
    
    plt.ioff()
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(x, prob_eq_par  , c='b')
    ax[0].plot(x, prob_xmax_par, c='r')
    ax[0].set_xlabel('$v_\parallel$')
    
    ax[1].plot(x, prob_eq_per, c='b')
    ax[1].plot(x, prob_xmax_per, c='r')
    ax[1].set_xlabel('$v_\perp$')
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
    
    main_1D.velocity_update(vp, Ie, We, Ib, Wb, idx, B, E, -0.5*dt)
    
    xy    = np.zeros((maxtime+1, 2))
    
    for ii in range(maxtime+1):
        xy[ii, 0] = xp[1]
        xy[ii, 1] = xp[2]

        position_update(xp, vp, dt)
        main_1D.velocity_update(vp, Ie, We, Ib, Wb, idx, B, E, dt)
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
    nspace        = 100

    positions     = np.zeros((3, nspace), dtype=float)
    positions[0]  = np.linspace(main_1D.xmin, main_1D.xmax, nspace, endpoint=True)
    normal_weight = np.zeros(positions.shape[1])
    left_nodes    = np.zeros(positions.shape[1], dtype=int)
    weights       = np.zeros((3, positions.shape[1]))
    
    main_1D.assign_weighting_TSC(positions, left_nodes, weights, E_nodes=False)
    
    for ii in range(positions.shape[1]):
        normal_weight[ii] = weights[:, ii].sum()

    plt.plot(positions[0], normal_weight)
    plt.xlim(main_1D.xmin, main_1D.xmax)
    return


def test_weight_shape_and_alignment():
    plt.ion()
    
    positions  = np.array([-1.5]) * main_1D.dx
    
    XMIN       = main_1D.xmin
    XMAX       = main_1D.xmax
    E_nodes    = main_1D.E_nodes
    B_nodes    = main_1D.B_nodes    
    
    E_grid     = False
    arr_offset = 0       if E_grid is True else 1
    X_grid     = E_nodes if E_grid is True else B_nodes
    X_color    = 'r'     if E_grid is True else 'b'
    
    Np         = positions.shape[0]
    Nc         = main_1D.NX + 2*main_1D.ND + arr_offset
    
    W_test     = np.zeros(Nc) 
    left_nodes = np.zeros(Np, dtype=int)
    weights    = np.zeros((3, Np))

    main_1D.assign_weighting_TSC(positions, left_nodes, weights, E_nodes=E_grid)
    
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
    E_nodes  = (np.arange(main_1D.NX + 2*main_1D.ND    ) - main_1D.ND  + 0.5) * main_1D.dx / 1e3
    B_nodes  = (np.arange(main_1D.NX + 2*main_1D.ND + 1) - main_1D.ND  - 0.0) * main_1D.dx / 1e3
    
    plt.figure()
    plt.plot(E_nodes, qn, marker='o')
    
    for ii in range(E_nodes.shape[0]):
        plt.axvline(E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[ii], linestyle='--', c='b', alpha=0.2)
     
    plt.axvline(B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    
    plt.axvline(main_1D.xmin / 1e3, linestyle=':', c='k', alpha=0.5)
    plt.axvline(main_1D.xmax / 1e3, linestyle=':', c='k', alpha=0.5)
    plt.ylabel('Charge density')
    plt.xlabel('x (km)')
    
    plt.figure()
    plt.plot(E_nodes, ji, marker='o')
    
    for ii in range(E_nodes.shape[0]):
        plt.axvline(E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[ii], linestyle='--', c='b', alpha=0.2)
     
    plt.axvline(B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    
    plt.axvline(main_1D.xmin / 1e3, linestyle=':', c='k', alpha=0.5)
    plt.axvline(main_1D.xmax / 1e3, linestyle=':', c='k', alpha=0.5)
    plt.ylabel('Current density')
    plt.xlabel('x (km)')
    
    
    return


def test_density_and_velocity_deposition():
    # Change dx to 1 and NX/ND/ppc to something reasonable to make this nice
    # Use the sim_params with only one species
    POS, VEL, IE, W_ELEC, IB, W_MAG, IDX  = main_1D.initialize_particles()
    Q_DENS, Q_DENS_ADV, JI, NI, NU        = main_1D.initialize_source_arrays()
    temp1D                                = np.zeros(main_1D.NC, dtype=np.float64) 
    
    main_1D.collect_moments(VEL, IE, W_ELEC, IDX, Q_DENS, JI, NI, NU, temp1D) 
    
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
    plt.plot(main_1D.E_nodes, Q_DENS / Q_DENS.max(), marker='o')
        
    for ii in range(main_1D.E_nodes.shape[0]):
        plt.axvline(main_1D.E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(main_1D.B_nodes[ii], linestyle='--', c='b', alpha=0.2)
     
    plt.axvline(main_1D.xmin, color='k')
    plt.axvline(main_1D.xmax, color='k')
    plt.axvline(main_1D.B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(main_1D.B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    return


def check_density_deposition():
    # Change dx to 1 and NX/ND to reasonable values to make this nice
    # Don't forget to comment out the density floor AND re-enable it when done.
    #positions   = np.array([2.00])
    positions = np.arange(-main_1D.NX/2, main_1D.NX/2 + dy, dy)

    Np  = positions.shape[0]
    pos = np.zeros((3, Np))
    
    for ii in range(Np):
        pos[0, ii] = positions[ii]
    
    vel        = np.zeros((3, Np))
    idx        = np.zeros(Np, dtype=np.uint8)
    Ie         = np.zeros(Np, dtype=np.uint16)
    W_elec     = np.zeros((3, Np), dtype=np.float64)
    
    q_dens, q_dens_adv, Ji, ni, nu = main_1D.initialize_source_arrays()
    temp1D                         = np.zeros(main_1D.NC, dtype=np.float64) 
    
    main_1D.assign_weighting_TSC(pos, Ie, W_elec)
    main_1D.collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu, temp1D) 
    
    # Plot normalized charge density
    q_dens /= (main_1D.q * main_1D.n_contr[0])
    #pdb.set_trace()
    #print(q_dens.sum())
    ypos = np.ones(Np) * q_dens.max() * 1.1
    plt.plot(main_1D.E_nodes, q_dens, marker='o')
    plt.scatter(pos[0], ypos, marker='x', c='k')
        
    for ii in range(main_1D.E_nodes.shape[0]):
        plt.axvline(main_1D.E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(main_1D.B_nodes[ii], linestyle='--', c='b', alpha=0.2)
     
    plt.axvline(main_1D.xmin, linestyle=':', c='k'       , alpha=0.7)
    plt.axvline(main_1D.xmax, linestyle=':', c='k'       , alpha=0.7)
    plt.axvline(main_1D.B_nodes[ 0], linestyle='-', c='darkblue', alpha=0.7)
    plt.axvline(main_1D.B_nodes[-1], linestyle='-', c='darkblue', alpha=0.7)
    return



def test_velocity_deposition():
    E_nodes = (np.arange(main_1D.NX + 3) - 0.5) #* main_1D.dx
    B_nodes = (np.arange(main_1D.NX + 3) - 1.0) #* main_1D.dx
    
    dt       = 0.1
    velocity = np.array([[0.3 * main_1D.dx / dt, 0.0],
                         [ 0., 0.0],
                         [ 0., 0.0]])
    
    position = np.array([16.5, 16.5]) * main_1D.dx
    idx      = np.array([0, 0]) 
    
    left_nodes, weights = main_1D.assign_weighting_TSC(position)
    n_i, nu_i = main_1D.deposit_velocity_moments(velocity, left_nodes, weights, idx)

    for jj in range(main_1D.Nj):
        normalized_density = (main_1D.cellpart / main_1D.Nj)*n_i[:, jj] / main_1D.density[jj]
        species_color = main_1D.temp_color[jj]
        plt.plot(E_nodes, normalized_density, marker='o', c=species_color)
        
        print('Normalized total density contribution of species {} is {}'.format(jj, normalized_density.sum()))

    for ii in range(main_1D.NX + 3):
        plt.axvline(E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(B_nodes[ii], linestyle='--', c='b', alpha=0.2)
        
    plt.axvline(main_1D.xmin/main_1D.dx, linestyle='-', c='k', alpha=0.2)
    plt.axvline(main_1D.xmax/main_1D.dx, linestyle='-', c='k', alpha=0.2)
    return


def test_force_interpolation():
    E = np.zeros((main_1D.NX + 3, 3))
    B = np.zeros((main_1D.NX + 3, 3))
    
    E_nodes = (np.arange(main_1D.NX + 3) - 0.5) #* main_1D.dx
    B_nodes = (np.arange(main_1D.NX + 3) - 1.0) #* main_1D.dx
    
    B[:, 0] = np.arange(main_1D.NX + 3) * 5e-9            # Linear
    B[:, 1] = np.sin(0.5*np.arange(main_1D.NX + 3) + 5)   # Sinusoidal
    B[:, 2] = 4e-9                                      # Constant
    
    E[:, 0] = np.arange(main_1D.NX + 3) * 1e-5        # Linear
    E[:, 1] = np.sin(0.5*np.arange(main_1D.NX + 3))   # Sinusoidal
    E[:, 2] = 3e-5                                  # Constant
    
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid((3, 3), (0,0), colspan=3)
    ax2 = plt.subplot2grid((3, 3), (1,0), colspan=3)
    ax3 = plt.subplot2grid((3, 3), (2,0), colspan=3)
    #plt.tight_layout(pad=1.0, w_pad=1.8)
    fig.subplots_adjust(hspace=0)
    
    which_field = 'B'
    for ii in np.arange(0, main_1D.NX + 2, 0.5):
        position   = np.array([ii]) * main_1D.dx
        
        Ie, W_elec = main_1D.assign_weighting_TSC(position, E_nodes=True)
        Ib, W_mag  = main_1D.assign_weighting_TSC(position, E_nodes=False)
    
        Ep, Bp     = main_1D.interpolate_forces_to_particle(E, B, Ie[0], W_elec[:, 0], Ib[0], W_mag[:, 0])

        for ax, jj in zip([ax1, ax2, ax3], list(range(3))):
            ax.clear()
            ax.set_xlim(-1.5, main_1D.NX + 2)
            
            if which_field == 'E':
                ax1.set_title('Electric field interpolation to Particle')
                ax.plot(E_nodes, E[:, jj])
                ax.scatter(ii, Ep[jj])
            elif which_field == 'B':
                ax1.set_title('Magnetic field interpolation to Particle')
                ax.plot(B_nodes, B[:, jj])
                ax.scatter(ii, Bp[jj])
 
            for kk in range(main_1D.NX + 3):
                ax.axvline(E_nodes[kk], linestyle='--', c='r', alpha=0.2)
                ax.axvline(B_nodes[kk], linestyle='--', c='b', alpha=0.2)
                
                ax.axvline(main_1D.xmin/main_1D.dx, linestyle='-', c='k', alpha=0.2)
                ax.axvline(main_1D.xmax/main_1D.dx, linestyle='-', c='k', alpha=0.2)
    
        plt.pause(0.01)
    
    plt.show()

    return


def test_curl_B():
    '''
    Confirmed
    '''
    NC   = main_1D.NC   
    k    = 2 * np.pi / (2 * main_1D.xmax)

    # Inputs and analytic solutions
    B_input       = np.zeros((NC + 1, 3))
    B_input[:, 0] = np.cos(1.0*k*main_1D.B_nodes)
    B_input[:, 1] = np.cos(1.5*k*main_1D.B_nodes)
    B_input[:, 2] = np.cos(2.0*k*main_1D.B_nodes)
    
    curl_B_anal       = np.zeros((NC, 3))
    curl_B_anal[:, 1] =  2.0 * k * np.sin(2.0*k*main_1D.E_nodes)
    curl_B_anal[:, 2] = -1.5 * k * np.sin(1.5*k*main_1D.E_nodes)
    
    curl_B_FD = np.zeros((NC, 3))
    main_1D.curl_B_term(B_input, curl_B_FD)
    
    curl_B_FD *= main_1D.mu0
    
    ## DO THE PLOTTING ##
    plt.figure(figsize=(15, 15))
    marker_size = None
        
    plt.scatter(main_1D.E_nodes, curl_B_anal[:, 1], marker='o', c='k', s=marker_size, label='By Node Solution')
    plt.scatter(main_1D.E_nodes, curl_B_FD[  :, 1], marker='x', c='b', s=marker_size, label='By Finite Difference')
      
    plt.scatter(main_1D.E_nodes, curl_B_anal[:, 2], marker='o', c='k', s=marker_size, label='Bz Node Solution')
    plt.scatter(main_1D.E_nodes, curl_B_FD[  :, 2], marker='x', c='r', s=marker_size, label='Bz Finite Difference')   
    plt.title(r'Test of $\nabla \times B$')

    for ii in range(main_1D.E_nodes.shape[0]):
        plt.axvline(main_1D.E_nodes[ii], linestyle='--', c='r', alpha=0.2)
        plt.axvline(main_1D.B_nodes[ii], linestyle='--', c='b', alpha=0.2)
     
    plt.axvline(main_1D.B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(main_1D.B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    
    plt.axvline(main_1D.xmin, linestyle=':', c='k', alpha=0.5)
    plt.axvline(main_1D.xmax, linestyle=':', c='k', alpha=0.5)
    plt.xlabel('x (km)')
    
    plt.legend()
    return


def test_curl_E():
    '''
    Confirmed with parabolic B0 code
    '''
    NC   = main_1D.NC   
    k    = 2 * np.pi / (2 * main_1D.xmax)

    # Inputs and analytic solutions
    E_input       = np.zeros((NC, 3))
    E_input[:, 0] = np.cos(1.0*k*main_1D.E_nodes)
    E_input[:, 1] = np.cos(1.5*k*main_1D.E_nodes)
    E_input[:, 2] = np.cos(2.0*k*main_1D.E_nodes)

    curl_E_FD = np.zeros((NC + 1, 3))
    main_1D.get_curl_E(E_input, curl_E_FD)
    
    curl_E_anal       = np.zeros((NC + 1, 3))
    curl_E_anal[:, 1] =  2.0 * k * np.sin(2.0*k*main_1D.B_nodes)
    curl_E_anal[:, 2] = -1.5 * k * np.sin(1.5*k*main_1D.B_nodes)
    
    
    ## PLOT
    plt.figure(figsize=(15, 15))
    marker_size = None

    if False:
        plt.scatter(main_1D.E_nodes, E_input[:, 0], marker='o', c='k', s=marker_size, label='Ex Node Solution')
        plt.scatter(main_1D.E_nodes, E_input[:, 1], marker='o', c='k', s=marker_size, label='Ey Node Solution')
        plt.scatter(main_1D.E_nodes, E_input[:, 2], marker='o', c='k', s=marker_size, label='Ez Node Solution')

    if True:
        plt.scatter(main_1D.B_nodes, curl_E_anal[:, 1], marker='o', c='k', s=marker_size, label='By Node Solution')
        plt.scatter(main_1D.B_nodes, curl_E_FD[  :, 1], marker='x', c='b', s=marker_size, label='By Finite Difference')
          
        plt.scatter(main_1D.B_nodes, curl_E_anal[:, 2], marker='o', c='k', s=marker_size, label='Bz Node Solution')
        plt.scatter(main_1D.B_nodes, curl_E_FD[  :, 2], marker='x', c='r', s=marker_size, label='Bz Finite Difference')   
    
    plt.title(r'Test of $\nabla \times E$')

    ## Add node markers and boundaries
    for kk in range(NC):
        plt.axvline(main_1D.E_nodes[kk], linestyle='--', c='r', alpha=0.2)
        plt.axvline(main_1D.B_nodes[kk], linestyle='--', c='b', alpha=0.2)
    
    plt.axvline(main_1D.B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(main_1D.B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    
    plt.axvline(main_1D.xmin, linestyle='-', c='k', alpha=0.2)
    plt.axvline(main_1D.xmax, linestyle='-', c='k', alpha=0.2)
    
    plt.legend()
    return


def test_grad_P():
    '''
    Verified for parabolic B0 :: Analytic solutions are a pain, but these come
    out looking sinusoidal as expected
    '''
    k    = 2 * np.pi / (2 * main_1D.xmax)
    
    # Set analytic solutions (input/output)
    if False:
        q_dens    = np.cos(1.0 * k * main_1D.E_nodes)  * main_1D.q * main_1D.ne
        te_input  = np.ones(main_1D.NC)*main_1D.Te0_scalar
    elif False:
        q_dens    = np.ones(main_1D.NC) * main_1D.q * main_1D.ne
        te_input  = np.cos(1.0 * k * main_1D.E_nodes)*main_1D.Te0_scalar
    else:
        q_dens    = np.cos(1.0 * k * main_1D.E_nodes)* main_1D.q * main_1D.ne
        te_input  = np.cos(1.0 * k * main_1D.E_nodes)*main_1D.Te0_scalar
    
    gp_diff   = np.zeros(main_1D.NC)
    temp      = np.zeros(main_1D.NC + 1)
    
    # Finite differences
    main_1D.get_grad_P(q_dens, te_input, gp_diff, temp)

    ## PLOT ##
    plt.figure(figsize=(15, 15))
    marker_size = None

    plt.scatter(main_1D.E_nodes, gp_diff, marker='x', c='r', s=marker_size, label='Finite Difference')
    
    plt.title(r'Test of $\nabla p_e$')

    for kk in range(main_1D.NC):
        plt.axvline(main_1D.E_nodes[kk], linestyle='--', c='r', alpha=0.2)
        plt.axvline(main_1D.B_nodes[kk], linestyle='--', c='b', alpha=0.2)
    
    plt.axvline(main_1D.B_nodes[ 0], linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(main_1D.B_nodes[-1], linestyle='-', c='darkblue', alpha=1.0)
    
    plt.axvline(main_1D.xmin, linestyle='-', c='k', alpha=0.2)
    plt.axvline(main_1D.xmax, linestyle='-', c='k', alpha=0.2)
    
    plt.legend()
    return


def test_grad_P_with_init_loading():
    '''
    Init density is homogenous :: Confirmed
    
    '''
    pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, temp_N = main_1D.initialize_particles()
    q_dens, q_dens2, Ji, ni, nu                          = main_1D.initialize_source_arrays()
    
    main_1D.collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu)
    
    te_input  = np.ones(main_1D.NC)*main_1D.Te0_scalar
    gp_diff   = np.zeros(main_1D.NC)
    temp      = np.zeros(main_1D.NC + 1)
    
    main_1D.get_grad_P(q_dens, te_input, gp_diff, temp)
    
    fig, ax = plt.subplots(4, figsize=(15, 10), sharex=True)
    ax[0].set_title('Particle Moments')
    ax[0].plot(main_1D.E_nodes, q_dens)
    ax[1].plot(main_1D.E_nodes, Ji[:, 0])
    ax[2].plot(main_1D.E_nodes, Ji[:, 1])
    ax[3].plot(main_1D.E_nodes, Ji[:, 2])
    
    ax[0].set_ylabel(r'$\rho_c$')
    ax[1].set_ylabel(r'$J_x$')
    ax[2].set_ylabel(r'$J_y$')
    ax[3].set_ylabel(r'$J_z$')
    
    fig2, ax2 = plt.subplots(figsize=(15, 10))
    
    ax2.set_title('E-field contribution of initial density loading')
    ax2.plot(main_1D.E_nodes, gp_diff / q_dens)
    
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
    main_1D.cross_product(A, B, test_result)
    
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
    xmin = 0.0     #main_1D.xmin
    xmax = 2*np.pi #main_1D.xmax
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
    
    E_FD = main_1D.calculate_E(B_input, J_input, qn_input, DX=dx)
    
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
    xmin = 0.0     #main_1D.xmin
    xmax = 2*np.pi #main_1D.xmax
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
    
    E_FD  = main_1D.calculate_E(       B_input, J_input, qn_input, DX=dx)
    E_FD2 = main_1D.calculate_E_w_exel(B_input, J_input, qn_input, DX=dx)
    
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
    NX   = 32      #main_1D.NX
    xmin = 0.0     #main_1D.xmin
    xmax = 2*np.pi #main_1D.xmax
    
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
    B_center       = main_1D.interpolate_to_center_cspline3D(B_input, DX=dx)
    
    
    ## TEST CURL B (AGAIN JUST TO BE SURE)
    curl_B_FD   = main_1D.get_curl_B(B_input, DX=dx)
    curl_B_anal = np.zeros((NX + 3, 3))
    curl_B_anal[:, 1] = -dBz
    curl_B_anal[:, 2] =  dBy


    ## ELECTRIC FIELD CALCULATION ## 
    E_FD         =   main_1D.calculate_E(       B_input, np.zeros((NX + 3, 3)), np.ones(NX + 3), DX=dx)
    E_FD2        =   main_1D.calculate_E_w_exel(B_input, np.zeros((NX + 3, 3)), np.ones(NX + 3), DX=dx)
    
    E_anal       = np.zeros((NX + 3, 3))
    E_anal[:, 0] = - (Bye * dBy + Bze * dBz)
    E_anal[:, 1] = Bxe * dBy
    E_anal[:, 2] = Bxe * dBz
    E_anal      /= main_1D.mu0
        

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



def test_E2C_interpolation():
    '''
    Tests Edge-to-Center (B to E) cubic spline interpolation. 
    Only tests y component, since x is constant offset and z should be identical.
    '''
    marker_size = 20
    LENGTH      = main_1D.xmax - main_1D.xmin
    k           = 1.0 / LENGTH

    # Interpolation
    B_input       = np.zeros((main_1D.NC + 1, 3))
    B_input[:, 1] = np.cos(2*np.pi*k*main_1D.B_nodes)    
    
    # Analytic solution
    B_anal        = np.zeros((main_1D.NC, 3))
    B_anal[:, 1]  = np.cos(2*np.pi*k*main_1D.E_nodes)
    
    ## TEST INTERPOLATION ##
    B_cent        = np.zeros((main_1D.NC, 3))
    main_1D.interpolate_edges_to_center(B_input, B_cent)

    fig, ax = plt.subplots()
    ax.scatter(main_1D.B_nodes, B_input[:, 1], s=marker_size, c='k', marker='o', label='Input')
    ax.scatter(main_1D.E_nodes, B_anal[ :, 1], s=marker_size, c='b', marker='x', label='Analytic Soln')
    ax.scatter(main_1D.E_nodes, B_cent[ :, 1], s=marker_size, c='r', marker='x', label='Numerical Soln')
    
    for kk in range(main_1D.NC):
        ax.axvline(main_1D.E_nodes[kk], linestyle='--', c='r', alpha=0.2)
        ax.axvline(main_1D.B_nodes[kk], linestyle='--', c='b', alpha=0.2)
    ax.axvline(main_1D.B_nodes[kk+1], linestyle='--', c='b', alpha=0.2)
        
    ax.axvline(main_1D.xmin, linestyle='-', c='k', alpha=0.2)
    ax.axvline(main_1D.xmax, linestyle='-', c='k', alpha=0.2)

    ax.legend()
    plt.show()
    return


def test_C2E_interpolation():
    '''
    Tests Center-to-Edge (E to B) cubic spline interpolation. 
    Only tests x component, as the rest should be fine
    
    y2 tested and validated, although its not great near edges. May also invalidated
    how '4th order' the cubic spline is (derivative uses 2nd order finite difference)
    '''
    marker_size = 20
    LENGTH      = main_1D.xmax - main_1D.xmin
    k           = 1.0 / LENGTH

    # Interpolation
    E_input       = np.zeros((main_1D.NC, 3))
    E_input[:, 1] = np.cos(2*np.pi*k*main_1D.E_nodes)    
    
    # Analytic solution
    E_anal        = np.zeros((main_1D.NC + 1, 3))
    E_anal[:, 1]  = np.cos(2*np.pi*k*main_1D.B_nodes)
    
    # Test y2 Analytic solution (needs return after y2 calculation)
    if False:
        y2_anal       = np.zeros((main_1D.NC, 3))
        y2_anal[:, 1] = (-(2*np.pi*k) ** 2) * np.cos(2*np.pi*k*main_1D.E_nodes)
       
        E_edge        = np.zeros((main_1D.NC + 1, 3))
        y2 = main_1D.interpolate_centers_to_edge(E_input, E_edge)
    
        fig, ax = plt.subplots()
        ax.scatter(main_1D.E_nodes, y2_anal[:, 1], s=marker_size, c='k', marker='o', label='Analytic  y2')
        ax.scatter(main_1D.E_nodes, y2[     :, 1], s=marker_size, c='r', marker='x', label='Numerical y2')
    
    ## TEST INTERPOLATION ##
    E_edge        = np.zeros((main_1D.NC + 1, 3))
    main_1D.interpolate_centers_to_edge(E_input, E_edge)

    fig, ax = plt.subplots()
    ax.scatter(main_1D.E_nodes, E_input[:, 1], s=marker_size, c='k', marker='o', label='Input')
    ax.scatter(main_1D.B_nodes, E_anal[ :, 1], s=marker_size, c='b', marker='x', label='Analytic Soln')
    ax.scatter(main_1D.B_nodes, E_edge[ :, 1], s=marker_size, c='r', marker='x', label='Numerical Soln')
    
    for kk in range(main_1D.NC):
        ax.axvline(main_1D.E_nodes[kk], linestyle='--', c='r', alpha=0.2)
        ax.axvline(main_1D.B_nodes[kk], linestyle='--', c='b', alpha=0.2)
    ax.axvline(main_1D.B_nodes[kk+1], linestyle='--', c='b', alpha=0.2)
        
    ax.axvline(main_1D.xmin, linestyle='-', c='k', alpha=0.2)
    ax.axvline(main_1D.xmax, linestyle='-', c='k', alpha=0.2)

    ax.legend()
    plt.show()
    return


def test_cspline_order():
    '''
    dx not used in these interpolations, so solution order is calculated by
    having the same waveform sampled with different numbers of cells across
    LENGTH.
    '''
    marker_size = 20
    grids       = [128, 256]
    E2C_errors  = np.zeros(len(grids))
    C2E_errors  = np.zeros(len(grids))
    
    for mx, ii in zip(grids, list(range(len(grids)))):
        print('Testing interpolation :: {} points'.format(mx))
        xmax   = 2*np.pi
        dx     = xmax / mx
        k      = 2.0 / xmax
        
        # Physical location of nodes
        B_nodes  = (np.arange(mx + 1) - mx // 2)       * dx      # B grid points position in space
        E_nodes  = (np.arange(mx)     - mx // 2 + 0.5) * dx      # E grid points position in space
    
        edge_soln       = np.zeros((mx + 1, 3))                    # Input at B nodes
        edge_soln[:, 1] = np.cos(2*np.pi*k*B_nodes)
        
        center_soln        = np.zeros((mx, 3))
        center_soln[:, 1]  = np.cos(2*np.pi*k*E_nodes)                # Solution at E nodes
    
        ## TEST INTERPOLATIONs ##
        center_interp = np.zeros((mx, 3))
        main_1D.interpolate_edges_to_center(edge_soln, center_interp, zero_boundaries=False)

        edge_interp  = np.zeros((mx + 1, 3))
        main_1D.interpolate_centers_to_edge(center_soln, edge_interp, zero_boundaries=False)

        E2C_errors[ii] = abs(center_interp[:, 1] - center_soln[:, 1]).max()
        C2E_errors[ii] = abs(edge_interp[  :, 1] - edge_soln[  :, 1]).max()
        
        if ii == 0:
            plt.figure()
            plt.title('E2C Interpolation :: {} points'.format(mx))
            plt.scatter(B_nodes, edge_soln[    :, 1], s=marker_size, c='k', marker='o', label='Input')
            plt.scatter(E_nodes, center_soln[  :, 1], s=marker_size, c='b', marker='x', label='Analytic  Soln')
            plt.scatter(E_nodes, center_interp[:, 1], s=marker_size, c='r', marker='x', label='Numerical Soln')
            plt.legend()
            
            plt.figure()
            plt.title('C2E Interpolation :: {} points'.format(mx))
            plt.scatter(E_nodes, center_soln[:, 1], s=marker_size, c='k', marker='o', label='Input')
            plt.scatter(B_nodes, edge_soln[  :, 1], s=marker_size, c='b', marker='x', label='Analytic  Soln')
            plt.scatter(B_nodes, edge_interp[:, 1], s=marker_size, c='r', marker='x', label='Numerical Soln')
            plt.legend()

    for ii in range(len(grids) - 1):
        E2C_order = np.log(E2C_errors[ii] / E2C_errors[ii + 1]) / np.log(2)
        C2E_order = np.log(C2E_errors[ii] / C2E_errors[ii + 1]) / np.log(2)
        print(E2C_order, C2E_order)
    return


def test_interp_cross_manual():
    '''
    Test order of cross product with interpolation, separate from hall term calculation (double check)
    '''
    grids  = [16, 32, 64, 128, 256, 512, 1024]
    errors = np.zeros(len(grids))
    
    #NX   = 32      #main_1D.NX
    xmin = 0.0     #main_1D.xmin
    xmax = 2*np.pi #main_1D.xmax
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
        
        B_inter      = main_1D.interpolate_to_center_cspline3D(B)
        
        ## RESULTS (AxB) ##
        anal_result       = np.ones((NX + 3, 3))
        anal_result[:, 0] = Ay*Bze - Az*Bye
        anal_result[:, 1] = Az*Bxe - Ax*Bze
        anal_result[:, 2] = Ax*Bye - Ay*Bxe
        
        test_result  = main_1D.cross_product(A, Be)
        inter_result = main_1D.cross_product(A, B_inter)

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
        Ls     = [main_1D.L]
        lat_st = 90 - main_1D.theta_xmax * 180./np.pi
        lat_en = 90 + main_1D.theta_xmax * 180./np.pi
    
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
    pos, vel, idx = main_1D.uniform_gaussian_distribution_quiet()
    
    plt.figure()
    for jj in range(main_1D.Nj):
        st = main_1D.idx_start[jj]
        en = main_1D.idx_end[  jj]    
        Np = en - st
        plt.scatter(pos[st:en]/main_1D.dx, np.ones(Np)*jj, color=main_1D.temp_color[jj])
        
    ## Add node markers and boundaries
    for kk in range(main_1D.NC):
        plt.axvline(main_1D.E_nodes[kk]/main_1D.dx, linestyle='--', c='r', alpha=0.2)
        plt.axvline(main_1D.B_nodes[kk]/main_1D.dx, linestyle='--', c='b', alpha=0.2)
    
    plt.axvline(main_1D.B_nodes[ 0]/main_1D.dx, linestyle='-', c='darkblue', alpha=1.0)
    plt.axvline(main_1D.B_nodes[-1]/main_1D.dx, linestyle='-', c='darkblue', alpha=1.0)
    
    plt.axvline(main_1D.xmin/main_1D.dx, linestyle='-', c='k', alpha=0.2)
    plt.axvline(main_1D.xmax/main_1D.dx, linestyle='-', c='k', alpha=0.2)
    plt.ylim(-0.1, 2)
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
    B0        = main_1D.B_xmax
    v0_perp   = main_1D.va
    
    gyfreq    = main_1D.gyfreq / (2 * np.pi)
    orbit_res = 0.05
    max_rev   = 1000
    dt        = orbit_res / gyfreq 
    max_t     = max_rev   / gyfreq
    max_inc   = int(max_t / dt)
    
    pos       = np.array([0])
    vel       = np.array([main_1D.va, main_1D.va, main_1D.va]).reshape((3, 1))
    idx       = np.array([0])
    
    # Dummy arrays so the functions work
    E       = np.zeros((main_1D.NC    , 3))
    B       = np.zeros((main_1D.NC + 1, 3))
    
    W_mag   = np.array([0, 1, 0]).reshape((3, 1))
    W_elec  = np.array([0, 1, 0]).reshape((3, 1))
    Ie      = np.zeros(pos.shape[0], dtype=int)
    Ib      = np.zeros(pos.shape[0], dtype=int)
    
    pos_history = np.zeros( max_inc)
    vel_history = np.zeros((max_inc, 3))
    
    B[:, 0]+= B0; tt = 0; t_total = 0
    
    main_1D.assign_weighting_TSC(pos, Ie, W_elec)
    main_1D.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, -0.5*dt)
    while tt < max_inc:
        pos_history[tt] = pos
        vel_history[tt] = vel[:, 0]
        
        main_1D.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, dt)
        main_1D.position_update(pos, vel, dt, Ie, W_elec)  
        
        if pos[0] < main_1D.xmin:
            pos[0] += main_1D.xmax
        elif pos[0] > main_1D.xmax:
            pos[0] -= main_1D.xmax
                
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
    B_test = np.zeros((main_1D.NC + 1, 3), dtype=np.float64) 
    E_test = np.zeros((main_1D.NC, 3),     dtype=np.float64) 
    
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
        v_par  = v_mag * main_1D.va * np.cos(pitch * np.pi / 180.)
        v_perp = v_mag * main_1D.va * np.sin(pitch * np.pi / 180.)
        
        #initial_gyrophase  = 270            # degrees
        #initial_gyrophase *= np.pi / 180.   # convert to radians
        
        vel[0, :] =  v_par
        vel[1, :] = 0.0  #- v_perp * np.sin(initial_gyrophase)
        vel[2, :] = -v_perp  #v_perp * np.cos(initial_gyrophase)
                
        rL = main_1D.mp * v_perp / (main_1D.q * main_1D.B_eq)
        
        pos[0, :] = 0.0  
        pos[1, :] = rL   #rL * np.cos(initial_gyrophase)
        pos[2, :] = 0.0  #rL * np.sin(initial_gyrophase)

    # Initial quantities
    init_pos = pos.copy() 
    init_vel = vel.copy()
    gyfreq   = main_1D.gyfreq / (2 * np.pi)
    ion_ts   = main_1D.orbit_res / gyfreq
    vel_ts   = 0.5 * main_1D.dx / np.max(np.abs(vel[0, :]))

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
    main_1D.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B_test, E_test, v_prime, S, T, qmi, -0.5*DT)
    
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
        
        main_1D.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B_test, E_test, v_prime, S, T, qmi, DT)
        main_1D.position_update(pos, vel, idx, DT, Ie, W_elec)
        
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
        qmi = main_1D.q / main_1D.mp
        vn  = vel[:, ii]
        
        B_p = main_1D.eval_B0_particle(pos[ii], vel[:, ii], qmi, np.array([0., 0., 0.]))
        
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
            #init_KE    = 0.5 * main_1D.mp * init_vel ** 2
            #init_pitch = np.arctan(init_vperp / init_vpara) * 180. / np.pi
            init_mu    = 0.5 * main_1D.mp * init_vperp ** 2 / main_1D.B_eq
            
            #vel_perp      = np.sqrt(vel_history[:, 1] ** 2 + vel_history[:, 2] ** 2)
            #vel_para      = vel_history[:, 0]
            #vel_perp      = np.sqrt(vel_history[:, 1] ** 2 + vel_history[:, 2] ** 2)
            #vel_magnitude = np.sqrt(vel_history[:, 0] ** 2 + vel_history[:, 1] ** 2 + vel_history[:, 2] ** 2)
            
            #B_para      = mag_history[:, 0]
            #B_perp      = np.sqrt(mag_history[:, 1] ** 2 + mag_history[:, 2] ** 2)
            #B_magnitude = np.sqrt(mag_history[:, 0] ** 2 + mag_history[:, 1] ** 2 + mag_history[:, 2] ** 2)
            
            KE_perp = 0.5 * main_1D.mp * (vel_history[:, 1] ** 2 + vel_history[:, 2] ** 2)
            #KE_para = 0.5 * main_1D.mp *  vel_history[:, 0] ** 2
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
        #axes[0].set_title(r'Velocity/Magnetic Field at Particle, v0 = [%4.1f, %4.1f, %4.1f]km/s, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg' % (init_vel[0, 0], init_vel[1, 0], init_vel[2, 0], main_1D.loss_cone, init_pitch))
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
        #axes[0].set_title(r'First Invariant $\mu$ for single trapped particle, v0 = [%3.1f, %3.1f, %3.1f]$v_{A,eq}^{-1}$, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg' % (init_vel[0, 0]/main_1D.va, init_vel[1, 0]/main_1D.va, init_vel[2, 0]/main_1D.va, main_1D.loss_cone, init_pitch))
        axes[0].set_title(r'First Invariant $\mu$ for single trapped particle :: TEST DT = %7.5fs :: Max $\delta \mu = $%6.4f%%' % (DT, mu_percent))
        axes[0].set_ylabel(r'$(\times 10^{-10})$', rotation=0, labelpad=30)
        axes[0].get_yaxis().get_major_formatter().set_useOffset(False)
        axes[0].axhline(init_mu*1e10, c='k', ls=':')
        axes[0].set_ylim(3.84, 3.99)
        # Check mu calculation against different sources of B: Source of oscillation? Nope
        mag_interp           = (mag_history_x[1:] + mag_history_x[:-1]) / 2
        mag_interp_magnitude = np.sqrt(mag_interp ** 2 + mag_history[1:, 1] ** 2 + mag_history[1:, 2] ** 2)
        
        mu_interp  = 0.5*main_1D.mp*(vel_history[1:, 1] ** 2 + vel_history[1:, 2] ** 2) / mag_interp_magnitude

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
        axes[0].set_title(r'Velocity/Magnetic Field at Particle, v0 = [%4.1f, %4.1f, %4.1f]km/s, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg' % (init_vel[0, 0], init_vel[1, 0], init_vel[2, 0], main_1D.loss_cone, init_pitch))
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
        axes[0].set_title(r'Position/Velocity of Particle : v0 = [%3.1f, %3.1f, %3.1f]$v_A$, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg' % (init_vel[0, 0]/main_1D.va, init_vel[1, 0]/main_1D.va, init_vel[2, 0]/main_1D.va, main_1D.loss_cone, init_pitch))

        axes[0].plot(time, pos_history*1e-3)
        axes[0].set_xlabel('t (s)')
        axes[0].set_ylabel(r'x (km)')
        axes[0].axhline(main_1D.xmin*1e-3, color='k', ls=':')
        axes[0].axhline(main_1D.xmax*1e-3, color='k', ls=':')
        axes[0].legend()
        
        axes[1].plot(time, vel_history[:, 0]/main_1D.va, label='$v_\parallel$')
        axes[1].plot(time,            vel_perp/main_1D.va, label='$v_\perp$')
        axes[1].set_xlabel('t (s)')
        axes[1].set_ylabel(r'$v_\parallel$ ($v_{A,eq}^{-1}$)')
        axes[1].legend()
        
        axes[2].plot(time, vel_history[:, 1]/main_1D.va, label='vy')
        axes[2].plot(time, vel_history[:, 2]/main_1D.va, label='vz')
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
        plt.plot(time, KE_para/main_1D.q, c='b', label=r'$KE_\parallel$')
        plt.plot(time, KE_perp/main_1D.q, c='r', label=r'$KE_\perp$')
        plt.plot(time, KE_tot /main_1D.q, c='k', label=r'$KE_{total}$')
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
        ax.set_xlim(main_1D.xmin*1e-3, main_1D.xmax*1e-3)
        ax.legend()

    if False:
        # Invariant and parameters vs. space
        fig, axes = plt.subplots(3, sharex=True)
        axes[0].plot(pos_history*1e-3, mu*1e10)
        axes[0].set_title(r'First Invariant $\mu$ for single trapped particle, v0 = [%3.1f, %3.1f, %3.1f]$v_{A,eq}^{-1}$, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg, $t_{max} = %5.0fs$' % (init_vel[0, 0]/main_1D.va, init_vel[1, 0]/main_1D.va, init_vel[2, 0]/main_1D.va, main_1D.loss_cone, init_pitch, max_t))
        axes[0].set_ylabel(r'$\mu (\times 10^{-10})$', rotation=0, labelpad=20)
        axes[0].get_yaxis().get_major_formatter().set_useOffset(False)
        axes[0].axhline(init_mu*1e10, c='k', ls=':')
        
        axes[1].plot(pos_history*1e-3, KE_perp/main_1D.q)
        axes[1].set_ylabel(r'$KE_\perp (eV)$', rotation=0, labelpad=20)

        axes[2].plot(pos_history*1e-3, B_magnitude*1e9)
        axes[2].set_ylabel(r'$|B|$ (nT)', rotation=0, labelpad=20)
        
        axes[2].set_xlabel('Position (km)')
        axes[2].set_xlim(main_1D.xmin*1e-3, main_1D.xmax*1e-3)
        
    if False:
        fig, axes = plt.subplots(2, sharex=True)
        axes[0].plot(pos_history*1e-3, mag_history[:, 0] * 1e9, lw=0.25, c='k')
        axes[0].set_title(r'Magnetic fields taken at v, x times :: v0 = [%3.1f, %3.1f, %3.1f]$v_{A,eq}^{-1}$, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg, $t_{max} = %5.0fs$' % (init_vel[0, 0]/main_1D.va, init_vel[1, 0]/main_1D.va, init_vel[2, 0]/main_1D.va, main_1D.loss_cone, init_pitch, max_t))
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
        
        ax[1].plot(time, KE_perp/main_1D.q, label='$KE_\perp$',     c='k', marker='o')
        ax[1].plot(time, KE_av/main_1D.q,   label='$KE_\perp$ av.', c='r', marker='x', ms=1.0)
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


def test_CAM_CL():
    
    grids  = [16, 32, 64, 128, 256, 512, 1024]
    errors = np.zeros(len(grids))
    
    NX   = 32      #const.NX
    xmin = 0.0     #const.xmin
    xmax = 2*np.pi #const.xmax
    k    = 1.0
    marker_size = 20
    
    #for NX, ii in zip(grids, range(len(grids))):
    dx      = xmax / NX
    x       = np.arange(xmin, xmax, dx/100.)
    E_nodes = (np.arange(NX + 3) - 0.5) * dx
    B_nodes = (np.arange(NX + 3) - 1.0) * dx
        
    B       = np.zeros((NX + 3, 3))
    n_i     = np.ones((NX + 3)) 
    J_i     = np.ones((NX + 3, 3))
    
    DT        = 1.0
    subcycles = 25
    
    new_B = main_1D.cyclic_leapfrog(B, n_i, J_i, DT, subcycles)
    
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
    
    J_out = main_1D.push_current(J_in, E_in, B, L_in, G_in, DT)
    
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


if __name__ == '__main__':
    #test_particle_orbit()
    #animate_moving_weight()
    #test_curl_E()
    #test_grad_P_varying_qn()
    #test_cross_product()
    #test_cspline_interpolation()
    #test_E_convective()
    #test_E_hall()
    #test_interp_cross_manual()
    #test_CAM_CL()
    #test_current_push()
    #test_E_convective_exelectron()
    
    test_weight_shape_and_alignment()
    