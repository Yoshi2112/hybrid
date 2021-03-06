# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numba as nb
import numpy as np

import pdb
from simulation_parameters_1D  import N, NX, dx, xmax, xmin, Nj, charge, mass, n_contr, do_parallel, smooth_sources, min_dens, ne, q
import auxilliary_1D as aux

def init_and_run():
    return


@nb.njit(parallel=True)
def advance_particles_and_moments(pos, vel, idx, B, E, DT):
    '''
    Helper function to group the particle advance and moment collection functions.

    Recode to do single loop over particle table. Don't use vectorised array operations... yet
    '''
    n_i    = np.zeros((E.shape[0], Nj))
    nu_i   = np.zeros((E.shape[0], Nj, 3))
    q_dens = np.zeros(E.shape[0])
    Ji     = np.zeros(E.shape)

    # Code for each particle
    for ii in nb.prange(N):
        Ie, We       = assign_weighting(pos[ii], E_nodes=True)
        Ib, Wb       = assign_weighting(pos[ii], E_nodes=False)
        vel[:, ii]   = velocity_update(vel[:, ii], Ie, We, Ib, Wb, idx[ii], B, E, DT)
        position_update(pos[ii], vel[:, ii], DT)

        deposit_moments(n_i, nu_i, vel[:, ii], Ie, We, idx[ii])

    transform_moments_to_densities(q_dens, Ji, n_i, nu_i)

    return q_dens, Ji


@nb.njit(parallel=do_parallel)
def sync_velocities(pos, vel, idx, B, E, DT):
    for ii in nb.prange(N):
        Ie, We       = assign_weighting(pos[ii], E_nodes=True)
        Ib, Wb       = assign_weighting(pos[ii], E_nodes=False)

        vel[:, ii]   = velocity_update(vel[:, ii], Ie, We, Ib, Wb, idx[ii], B, E, DT)
    return


@nb.njit()
def assign_weighting(pos, E_nodes=True):
    '''Triangular-Shaped Cloud (TSC) weighting scheme used to distribute particle densities to
    nodes and interpolate field values to particle positions.

    INPUT:
        pos  -- particle position (x)
        BE   -- Flag: Weighting factor for Magnetic (0) or Electric (1) field node

    OUTPUT:
        weights -- 3-array consisting of leftmost (to the nearest) node, and weights for -1, 0 TSC nodes

    Note: Designed for a single particle (as part of multi-threading)
    '''
    weights    = np.zeros(3, dtype=np.float64)

    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 1.0

    left_node   = int(round(pos / dx + grid_offset) - 1.0)
    delta_left  = left_node - pos / dx - grid_offset

    weights[0] = 0.5  * (1.5 - abs(delta_left)) ** 2
    weights[1] = 0.75 - (delta_left + 1.) ** 2
    weights[2] = 1.0  - weights[0] - weights[1]
    return left_node, weights


@nb.njit()
def velocity_update(vel, Ie, W_elec, Ib, W_mag, idx, B, E, dt):
    '''
    Interpolates the fields to the particle positions using TSC weighting, then
    updates velocities using a Boris particle pusher.
    Based on Birdsall & Langdon (1985), pp. 59-63.

    INPUT:
        part -- Particle array containing velocities to be updated
        B    -- Magnetic field on simulation grid
        EIb, W_mag = assign_weighting_TSC(pos, E_nodes=False)    -- Electric field on simulation grid
        dt   -- Simulation time cadence
        W    -- Weighting factor of particles to rightmost node

    OUTPUT:
        vel  -- Returns particle array with updated velocities
    '''
    Ep = E[Ie    , 0:3] * W_elec[0]                 \
       + E[Ie + 1, 0:3] * W_elec[1]                 \
       + E[Ie + 2, 0:3] * W_elec[2]                 # E-field at particle location

    Bp = B[Ib    , 0:3] * W_mag[0]                  \
       + B[Ib + 1, 0:3] * W_mag[1]                  \
       + B[Ib + 2, 0:3] * W_mag[2]                  # B-field at particle location

    T = (charge[idx] * Bp / mass[idx]) * dt / 2.                        # Boris variable
    S = 2.*T / (1. + T[0] ** 2 + T[1] ** 2 + T[2] ** 2)                 # Boris variable

    v_minus    = vel + charge[idx] * Ep * dt / (2. * mass[idx])
    v_prime    = v_minus + aux.cross_product_single(v_minus, T)
    v_plus     = v_minus + aux.cross_product_single(v_prime, S)

    vel        = v_plus + charge[idx] * Ep * dt / (2. * mass[idx])
    return vel





@nb.njit()
def deposit_moments(n_i, nu_i, vel, I, W_elec, sp):
    '''
    Moment (charge/current) collection function.

    INPUT:
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier

    OUTPUT:
        rho_c  -- Charge  density
        Ji     -- Current density
    '''
    for kk in range(3):
        nu_i[I,     sp, kk] += W_elec[0] * vel[kk]
        nu_i[I + 1, sp, kk] += W_elec[1] * vel[kk]
        nu_i[I + 2, sp, kk] += W_elec[2] * vel[kk]

    n_i[I,     sp] += W_elec[0]
    n_i[I + 1, sp] += W_elec[1]
    n_i[I + 2, sp] += W_elec[2]
    return


@nb.njit()
def transform_moments_to_densities(q_dens, Ji, n_i, nu_i):
    for jj in range(Nj):
        q_dens  += n_i[:, jj] * n_contr[jj] * charge[jj]

        for kk in range(3):
            Ji[:, kk] += nu_i[:, jj, kk] * n_contr[jj] * charge[jj]

    if smooth_sources == 1:
        for jj in range(Nj):
            n_i[:, jj]  = smooth(n_i[:, jj])

            for kk in range(3):
                nu_i[ :, jj, kk] = smooth(nu_i[:,  jj, kk])

    for ii in range(q_dens.shape[0]):
        if q_dens[ii] < min_dens * ne * q:
            q_dens[ii] = min_dens * ne * q
    return


@nb.njit()
def manage_ghost_cells(arr):
    '''Deals with ghost cells: Moves their contributions and mirrors their counterparts.
       Works like a charm if spatial dimensions always come first in an array.'''

    arr[NX]     += arr[0]                 # Move contribution: Start to end
    arr[1]      += arr[NX + 1]            # Move contribution: End to start

    arr[NX + 1]  = arr[1]                 # Fill ghost cell: End
    arr[0]       = arr[NX]                # Fill ghost cell: Start

    arr[NX + 2]  = arr[2]                 # This one doesn't get used, but prevents nasty nan's from being in array.
    return arr


@nb.njit()
def smooth(function):
    '''Smoothing function: Applies Gaussian smoothing routine across adjacent cells.
    Assummes no contribution from ghost cells.'''
    size         = function.shape[0]
    new_function = np.zeros(size)

    for ii in nb.prange(1, size - 1):
        new_function[ii - 1] = 0.25*function[ii] + new_function[ii - 1]
        new_function[ii]     = 0.50*function[ii] + new_function[ii]
        new_function[ii + 1] = 0.25*function[ii] + new_function[ii + 1]

    # Move Ghospost Cell Contributions: Periodic Boundary Condition
    new_function[1]        += new_function[size - 1]
    new_function[size - 2] += new_function[0]

    # Set ghost cell values to mirror corresponding real cell
    new_function[0]        = new_function[size - 2]
    new_function[size - 1] = new_function[1]
    return new_function


#@nb.njit()
def position_update(pos, vel, dt):
    '''
    Single particle thing
    '''
    pos += vel[0] * dt

    if pos < xmin:
        pos += xmax

    if pos > xmax:
        pos -= xmax
    return pos



@nb.guvectorize(["void(float64[:], float64[:,:], float64)"], "(n),(t,n),()", target='cpu')
def position_update_vectorize(pos, vel, dt):
    for ii in range(pos.shape[0]):
        pos[ii] += vel[0, ii] * dt

        if pos[ii] < xmin:
            pos[ii] += xmax

        if pos[ii] > xmax:
            pos[ii] -= xmax
    return


if __name__ == '__main__':

    @nb.njit(parallel=False)
    def test_call_function(pos, vel, dt, vec=False):

        if vec == True:
            position_update_vectorize(pos, vel, dt)
        else:
            for ii in nb.prange(pos.shape[0]):
                pos[ii] = position_update(pos[ii], vel[:, ii], dt)

        return

    N        = 320000000
    pos_test = np.linspace(xmin, xmax, N)
    vel_test = np.array([np.random.normal(0, 1, N),
                         np.random.normal(0, 1, N),
                         np.random.normal(0, 1, N)])
    dt_test  = 0.001

# =============================================================================
#     idx_test = np.ones(N, dtype=int)
#     B_test   = np.ones((NX + 3, 3)) * 4e-9
#     E_test   = np.zeros((NX + 3, 3))
#
# =============================================================================
# =============================================================================
#     advance_particles_and_moments(pos_test, vel_test, idx_test,  B_test, E_test, dt_test)
#
#     advance_particles_and_moments.parallel_diagnostics(level=4)
# =============================================================================
    from timeit import default_timer as timer

    print 'Calling particle push'

    start_time = timer()
    test_call_function(pos_test, vel_test, dt_test, vec=False)
    end_time = timer()

    print 'Execution time: {}s'.format(round(end_time - start_time, 3))
