# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import numba as nb
from timeit import default_timer as timer


def advance_particles_and_moments(pos, vel, idx, B, E, q_dens, Ji, DT):
    '''
    Helper function to group the particle advance and moment collection functions.

    Recode to do single loop over particle table. Don't use vectorised array operations... yet
    '''
    # Specify some temp arrays
    n_i    = np.zeros((E.shape[0], Nj))
    nu_i   = np.zeros((E.shape[0], Nj, 3))

    Ie      = np.random.randint(0, NX, N, dtype=np.uint8)
    We      = np.random.uniform(0, 1, (3, N))
    Ib      = np.random.randint(0, NX, N, dtype=np.uint8)
    Wb      = np.random.uniform(0, 1, (3, N))

    Ep      = np.zeros(N); Bp = np.zeros(N)

    assign_weighting(pos, Ie, We, True)
    assign_weighting(pos, Ib, Wb, False)

    interpolate_fields(B, E, Ie, We, Ib, Wb, Bp, Ep)
    velocity_update(vel, Ep, Bp, idx, DT, ctm_test)

    position_advance(pos, vel, DT)

    deposit_moments(n_i, nu_i, vel, Ie, We, idx)
    transform_moments(q_dens, Ji, n_i, nu_i)
    return



@nb.guvectorize([(nb.float64[:,:], nb.float64[:,:], nb.float64[:,:])],
                 '(n,m),(n,m)->(n,m)', target='cpu')
def cross_product_guvec(A, B, output):
    for ii in range(A.shape[0]):
        output[ii, 0] = A[ii, 1] * B[ii, 2] - A[ii, 2] * B[ii, 1]
        output[ii, 1] = A[ii, 2] * B[ii, 0] - A[ii, 0] * B[ii, 2]
        output[ii, 2] = A[ii, 0] * B[ii, 1] - A[ii, 1] * B[ii, 0]
    return


@nb.njit()
def cross_product_single(A, B):
    output = np.zeros(3)
    output[0] = A[1] * B[2] - A[2] * B[1]
    output[1] = A[2] * B[0] - A[0] * B[2]
    output[2] = A[0] * B[1] - A[1] * B[0]
    return output


@nb.guvectorize([(nb.float64[:], nb.float64[:,:], nb.float64)],
                 '(m),(n,m),()', target='cpu')
def position_advance(P, V, DT):
    for ii in nb.prange(P.shape[0]):
        P[ii] += V[0, ii] * DT
    return


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:,:], nb.boolean)],
                 '(n),(n),(m, n), ()', target='cpu')
def assign_weighting(P, I, W, E_nodes):
    DX = 0.1

    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 1.0

    for ii in nb.prange(P.shape[0]):
        I[ii]       = int(round(P[ii] / DX + grid_offset) - 1.0)
        delta_left  = I[ii] - P[ii] / DX - grid_offset

        W[0, ii] = 0.5  * (1.5 - abs(delta_left)) ** 2
        W[1, ii] = 0.75 - (delta_left + 1.) ** 2
        W[2, ii] = 1.0  - W[0, ii] - W[1, ii]
    return


@nb.guvectorize([(nb.float64[:,:], nb.uint8[:], nb.float64[:,:], nb.float64[:,:])],
                 '(i,m),(n),(m,n),(n,m)', target='cpu')
def interpolate_one_field(field, nodes, weights, interp):
    for ii in nb.prange(nodes.shape[0]):    # For each particle
        for jj in nb.prange(3):             # For each node
            for kk in nb.prange(3):         # For each direction
                interp[ii, kk] = field[nodes[ii] + jj, kk] * weights[jj, ii]
    return


#@nb.njit()
def interpolate_fields(B, E, Ie, We, Ib, Wb, Bp, Ep):
    '''
    Seems to have issues calling guvec's when in njit()?
    '''
    interpolate_one_field(E, Ie, We, Ep)
    interpolate_one_field(B, Ib, Wb, Bp)
    return


@nb.njit(parallel=False)
def velocity_update(vel, Ep, Bp, idx, dt, ctm_rat):
    for ii in nb.prange(vel.shape[1]):
        T = 0.5 * ctm_rat[idx[ii]] * Bp[ii] * dt                    # Boris variable
        S = 2.*T / (1. + T[0] ** 2 + T[1] ** 2 + T[2] ** 2)     # Boris variable

        v_minus    = vel[:, ii] + 0.5 * ctm_rat[idx[ii]] * Ep[ii] * dt
        v_prime    = v_minus + cross_product_single(v_minus, T)
        v_plus     = v_minus + cross_product_single(v_prime, S)

        vel[:, ii] = v_plus + 0.5 * ctm_rat[idx[ii]] * Ep[ii] * dt
    return


@nb.guvectorize([(nb.float64[:,:], nb.float64[:,:, :], nb.float64[:,:],
                  nb.uint8[:]    , nb.float64[:,:],    nb.uint8[:])],
                 '(i, j),(i, j, k),(k, n),(n), (k, n), (n)', target='cpu')
def deposit_moments(n_i, nu_i, vel, I, W, sp):
    for ii in nb.prange(I.shape[0]):              # For each particle
        for jj in nb.prange(3):                   # For each node
            n_i[I[ii] + jj, sp[ii]] += W[jj, ii]  # Deposit density
            for kk in range(3):                   # For each direction, deposit velocity
                nu_i[I[ii] + jj, sp[ii], kk] += W[jj, ii] * vel[kk, ii]
    return


@nb.guvectorize([(nb.float64[:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:, :])],
                 '(i),(i, k),(i, j),(i, j, k)', target='cpu')
def transform_moments(q_dens, Ji, n_i, nu_i):
    for ii in nb.prange(q_dens.shape[0]):
        for jj in nb.prange(Nj):
            q_dens[ii]  += n_i[ii, jj] * n_contr[jj] * charge[jj]

            for kk in nb.prange(3):
                Ji[ii, kk] += nu_i[ii, jj, kk] * n_contr[jj] * charge[jj]

    for ii in nb.prange(q_dens.shape[0]):
        if q_dens[ii] < min_dens * ne * q:
            q_dens[ii] = min_dens * ne * q
    return


if __name__ == '__main__':
    xmin = 0.; xmax = 10.; NX = 100 ; DX = xmax / NX
    N    = 10000000       ; Nj = 1

    ne = 10e6
    q  = 16.02e-19
    min_dens = 0.05
    n_contr = np.array([1])
    charge  = np.array([1])
    ctm_test = np.array([1])

    dt_test  = 0.001

    q_dens_test = np.zeros(NX + 3)
    Ji_test     = np.zeros((NX + 3, 3))

    pos_test = np.random.uniform(xmin, xmax, N)
    vel_test = np.random.normal(0, 0.2, (3, N))
    idx_test = np.zeros(N, dtype=np.uint8)


    B_test      = np.ones((NX + 3, 3)) * 4e-9
    E_test      = np.zeros((NX + 3, 3))

    ni_test     = np.zeros((NX + 3, Nj))
    nui_test    = np.zeros((NX + 3, Nj, 3))


    #memory_usage = (B_test.nbytes + E_test.nbytes
    #              + B_part.nbytes + E_part.nbytes
    #              + Ie_test.nbytes + We_test.nbytes
    #              + Ib_test.nbytes + Wb_test.nbytes
    #              ) / (1024. * 1024.)

    #print 'Memory usage: {}MiB'.format(round(memory_usage, 3))

#%% TEST LOOP

    start_time = timer()

    advance_particles_and_moments(pos_test, vel_test, idx_test, B_test, E_test,
                                  q_dens_test, Ji_test, dt_test)

    #transform_moments(q_dens_test, Ji_test, ni_test, nui_test)
    #transform_moments_guvec(q_dens_test, Ji_test, ni_test, nui_test)

    #deposit_moments(ni_test, nui_test, vel_test, Ie_test, We_test, idx_test)

    #velocity_update(vel_test, E_part, B_part, idx_test, dt_test, ctm_test)

    #interpolate_fields_to_particle_guvec(B_test, E_test, Ie_test, We_test, Ib_test, Wb_test, B_part, E_part)
    #interpolate_fields(B_test, E_test, Ie_test, We_test, Ib_test, Wb_test, B_part, E_part)

    #interpolate_one_field_guvec(     E_test, Ie_test, We_test, E_part)
    #interpolate_one_field_guvec_loop(E_test, Ie_test, We_test, E_part)

    #interpolate_fields_to_particle_guvec(B_test, E_test, Ie_test, We_test, Ib_test, Wb_test, B_part, E_part)
    #interpolate_fields_to_particle_prange(B_test, E_test, Ie_test, We_test, Ib_test, Wb_test, B_part, E_part)
    #interpolate_fields_to_particle_fancy(B_test, E_test, Ie_test, We_test, Ib_test, Wb_test, B_part, E_part)


    #assign_weighting_E_guvec(pos_test, Ie_test, We_test, True)
    #assign_weighting_E_prange(pos_test, I_test, W_test)

    #cross_product(arr1, arr2, ans)
    #cross_product_npvec(arr1, arr2, ans)
    #cross_product_guvec(arr1, arr2, ans)

    #position_advance_guvec(pos, vel, dt)
    #position_advance_prange(pos, vel, dt)
    #position_advance_npvec(pos, vel, dt)

    end_time   = timer()

    print('Execution time: {}s'.format(round(end_time - start_time, 5)))